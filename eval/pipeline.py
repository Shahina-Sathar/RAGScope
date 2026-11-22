"""Configurable RAG pipeline for the ablation study.

Built from RAGScope's own components (backend/app/...) so the numbers describe
this codebase: VectorRetriever (Pinecone + text-embedding-3-small), BM25Retriever,
HybridRetriever (RRF, k=60), CrossEncoderReranker (ms-marco-MiniLM-L-6-v2),
LLMQueryRewriter and AnswerGenerator (same system prompt for every config).

Differences from the FastAPI /query path, all deliberate and documented in
REPORT.md:
- Vector search reads a dedicated Pinecone namespace; BM25 reads
  eval/data/corpus.jsonl (not the app's bm25_chunks.json).
- The answer is always generated for the *original* question; a rewritten
  query is only used for retrieval and reranking.
- C4 is a LangGraph corrective loop wrapped around the C3 pipeline (rewrite ->
  hybrid -> rerank) with an LLM relevance grader. The app's existing
  CorrectiveRAG (vector-only, cosine >= 0.7 grader, its own prompt) is
  measured separately as C4_legacy.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.generation.generator import AnswerGenerator  # noqa: E402
from app.query.corrective_rag import CorrectiveRAG  # noqa: E402
from app.query.rewriter import LLMQueryRewriter  # noqa: E402
from app.retrieval.bm25 import BM25Result, BM25Retriever  # noqa: E402
from app.retrieval.hybrid import HybridChunk, HybridRetriever  # noqa: E402
from app.retrieval.reranker import CrossEncoderReranker, RerankedChunk  # noqa: E402
from app.retrieval.vector import RetrievedChunk, VectorRetriever  # noqa: E402

from eval.build_benchmark import CORPUS, NAMESPACE  # noqa: E402

LLM_SEED = 42
FINAL_K = 5
CANDIDATE_K = 20


@dataclass(frozen=True)
class Config:
    name: str
    description: str
    retrieval: Literal["vector", "hybrid"]
    rerank: bool = False
    rewrite: bool = False
    corrective: bool = False
    legacy_crag: bool = False


CONFIGS: dict[str, Config] = {
    c.name: c
    for c in [
        Config("C0", "Dense vector only, top-5", "vector"),
        Config("C1", "C0 + BM25 hybrid (RRF)", "hybrid"),
        Config("C2", "C1 + cross-encoder rerank (20 -> 5)", "hybrid", rerank=True),
        Config("C3", "C2 + LLM query rewriting", "hybrid", rerank=True, rewrite=True),
        Config(
            "C4",
            "C3 + LangGraph corrective loop (LLM grader)",
            "hybrid",
            rerank=True,
            rewrite=True,
            corrective=True,
        ),
        Config(
            "C4_legacy",
            "Rewrite + app's existing CorrectiveRAG (vector-only, cosine grader)",
            "vector",
            rewrite=True,
            legacy_crag=True,
        ),
    ]
}
MAIN_ORDER = ["C0", "C1", "C2", "C3", "C4"]


def seeded_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
        seed=LLM_SEED,
    )


class NamespacedVectorRetriever(VectorRetriever):
    """App VectorRetriever, pointed at the benchmark namespace."""

    def __init__(self, namespace: str):
        super().__init__()
        self.namespace = namespace

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_vector, top_k=top_k, include_metadata=True, namespace=self.namespace
        )
        return [
            RetrievedChunk(
                chunk_id=m.metadata.get("chunk_id", m.id),
                text=m.metadata.get("text", ""),
                document_name=m.metadata.get("document_name", ""),
                page_number=int(m.metadata.get("page_number", 0)),
                similarity_score=m.score,
            )
            for m in results.matches
        ]


GRADER_PROMPT = """You are grading retrieved documents for a question-answering system.
For each numbered document, decide whether it contains information that is relevant to
answering the question (even partially). Respond with JSON only, in the form
{"relevant": [true, false, ...]} with exactly one boolean per document, in order."""


class CorrectiveState(TypedDict):
    question: str
    query: str
    queries: list[str]
    docs: list
    relevant: list[bool]
    retry_count: int


class Components:
    """Shared, expensive objects (created once per process)."""

    def __init__(self):
        self.vector = NamespacedVectorRetriever(NAMESPACE)
        self.bm25 = BM25Retriever()
        with open(CORPUS, encoding="utf-8") as f:
            self.bm25.build_index([json.loads(line) for line in f])
        self.hybrid = HybridRetriever(self.vector, self.bm25)
        self.reranker = CrossEncoderReranker()
        _ = self.reranker.model  # load weights now, not inside a timed query
        self.rewriter = LLMQueryRewriter()
        self.rewriter._llm = seeded_llm()
        self.generator = AnswerGenerator()
        self.generator.llm = seeded_llm()
        self.grader = seeded_llm()
        self.legacy_crag = CorrectiveRAG(retriever=self.vector, max_retries=2,
                                         relevance_threshold=0.7)
        self.legacy_crag.llm = seeded_llm()
        self.legacy_crag.query_rewriter = self.rewriter


@dataclass
class Trace:
    stage_ms: dict = field(default_factory=dict)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_by_stage: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def add_ms(self, stage: str, ms: float):
        self.stage_ms[stage] = self.stage_ms.get(stage, 0.0) + ms


class _Stage:
    """Context manager: wall-clock time + LLM token usage for one stage."""

    def __init__(self, trace: Trace, name: str):
        self.trace, self.name = trace, name

    def __enter__(self):
        self._cb_cm = get_usage_metadata_callback()
        self._cb = self._cb_cm.__enter__()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        ms = (time.perf_counter() - self._t0) * 1000
        self._cb_cm.__exit__(*exc)
        self.trace.add_ms(self.name, ms)
        for usage in self._cb.usage_metadata.values():
            self.trace.input_tokens += usage.get("input_tokens", 0)
            self.trace.output_tokens += usage.get("output_tokens", 0)
            s = self.trace.tokens_by_stage.setdefault(self.name, [0, 0])
            s[0] += usage.get("input_tokens", 0)
            s[1] += usage.get("output_tokens", 0)
        return False


def _to_retrieved(c) -> RetrievedChunk:
    if isinstance(c, RetrievedChunk):
        return c
    score = (
        c.reranker_score if isinstance(c, RerankedChunk)
        else c.rrf_score if isinstance(c, HybridChunk)
        else c.bm25_score if isinstance(c, BM25Result)
        else 0.0
    )
    return RetrievedChunk(c.chunk_id, c.text, c.document_name, c.page_number, float(score))


class CountingLLM:
    """Counts .invoke() calls on a wrapped LangChain chat model."""

    def __init__(self, llm, trace_getter):
        self._llm, self._get = llm, trace_getter

    def invoke(self, *a, **kw):
        self._get().llm_calls += 1
        return self._llm.invoke(*a, **kw)


class AblationPipeline:
    def __init__(self, cfg: Config, comp: Components, max_retries: int = 2):
        self.cfg, self.c, self.max_retries = cfg, comp, max_retries
        self._trace: Trace | None = None
        get = lambda: self._trace  # noqa: E731
        # Wrap every LLM so calls are counted per query.
        self.rewriter = LLMQueryRewriter()
        self.rewriter._llm = CountingLLM(comp.rewriter._llm, get)
        self.generator = AnswerGenerator.__new__(AnswerGenerator)
        self.generator.llm = CountingLLM(comp.generator.llm, get)
        self.grader = CountingLLM(comp.grader, get)
        if cfg.legacy_crag:
            self.legacy = CorrectiveRAG(retriever=comp.vector, max_retries=2,
                                        relevance_threshold=0.7)
            self.legacy.llm = CountingLLM(comp.legacy_crag.llm, get)
            self.legacy.query_rewriter = self.rewriter
        if cfg.corrective:
            self.graph = self._build_graph()

    # ---- building blocks -------------------------------------------------
    def _retrieve(self, query: str, stage_prefix: str = "") -> list:
        t = self._trace
        k = CANDIDATE_K if self.cfg.rerank else FINAL_K
        with _Stage(t, stage_prefix + "retrieval"):
            if self.cfg.retrieval == "vector":
                chunks = self.c.vector.retrieve(query, top_k=k)
            else:
                chunks = self.c.hybrid.retrieve(query, top_k=k, candidate_k=CANDIDATE_K)
        if self.cfg.rerank and chunks:
            with _Stage(t, stage_prefix + "reranking"):
                chunks = self.c.reranker.rerank(query, chunks, top_k=FINAL_K)
        return [_to_retrieved(c) for c in chunks[:FINAL_K]]

    def _grade(self, question: str, docs: list[RetrievedChunk]) -> list[bool]:
        listing = "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(docs))
        msg = [
            SystemMessage(content=GRADER_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nDocuments:\n{listing}"),
        ]
        raw = self.grader.invoke(msg).content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            flags = [bool(x) for x in json.loads(raw)["relevant"]]
        except Exception:
            self._trace.extra["grader_parse_errors"] = (
                self._trace.extra.get("grader_parse_errors", 0) + 1
            )
            flags = [True] * len(docs)  # fail open: do not retry on parse errors
        return (flags + [False] * len(docs))[: len(docs)]

    # ---- C4: corrective loop around C3 -----------------------------------
    def _build_graph(self):
        def retrieve(state: CorrectiveState) -> dict:
            prefix = "corrective_" if state["retry_count"] > 0 else ""
            return {"docs": self._retrieve(state["query"], prefix)}

        def grade(state: CorrectiveState) -> dict:
            with _Stage(self._trace, "corrective_grading"):
                return {"relevant": self._grade(state["question"], state["docs"])}

        def route(state: CorrectiveState) -> str:
            if any(state["relevant"]) or state["retry_count"] >= self.max_retries:
                return "done"
            return "retry"

        def rewrite(state: CorrectiveState) -> dict:
            # Same semantics as app.query.corrective_rag: rewrite the current query.
            with _Stage(self._trace, "corrective_rewrite"):
                q = self.rewriter.rewrite(state["query"])
            return {"query": q, "queries": state["queries"] + [q],
                    "retry_count": state["retry_count"] + 1}

        g = StateGraph(CorrectiveState)
        g.add_node("retrieve", retrieve)
        g.add_node("grade", grade)
        g.add_node("rewrite", rewrite)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", route, {"retry": "rewrite", "done": END})
        g.add_edge("rewrite", "retrieve")
        return g.compile()

    # ---- run one question ------------------------------------------------
    def run(self, question: str) -> dict:
        t = self._trace = Trace()
        t0 = time.perf_counter()
        query = question
        out: dict = {}
        if self.cfg.legacy_crag:
            with _Stage(t, "rewriting"):
                query = self.rewriter.rewrite(question)
            with _Stage(t, "corrective_loop_total"):
                res = self.legacy.run(query)
            docs = res["documents"][:FINAL_K]
            answer = res["final_answer"]
            t.extra.update(retries=res["retry_count"], final_query=res["rewritten_query"] or query)
        else:
            if self.cfg.rewrite:
                with _Stage(t, "rewriting"):
                    query = self.rewriter.rewrite(question)
            if self.cfg.corrective:
                st = self.graph.invoke(
                    {"question": question, "query": query, "queries": [query], "docs": [],
                     "relevant": [], "retry_count": 0}
                )
                docs = st["docs"]
                t.extra.update(
                    retries=st["retry_count"],
                    queries=st["queries"],
                    final_relevant=st["relevant"],
                    retry_changed_query=len(set(st["queries"])) > 1,
                )
            else:
                docs = self._retrieve(query)
            with _Stage(t, "generation"):
                answer, _ = self.generator.generate(question, docs)
        total_ms = (time.perf_counter() - t0) * 1000
        out.update(
            answer=answer,
            retrieval_query=query,
            contexts=[{"chunk_id": d.chunk_id, "title": d.document_name, "text": d.text,
                       "score": float(d.similarity_score)} for d in docs],
            total_ms=total_ms,
            stage_ms=t.stage_ms,
            llm_calls=t.llm_calls,
            input_tokens=t.input_tokens,
            output_tokens=t.output_tokens,
            tokens_by_stage=t.tokens_by_stage,
            **t.extra,
        )
        return out
