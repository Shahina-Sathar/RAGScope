import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import Trace
from app.retrieval.vector import RetrievedChunk
from app.retrieval.bm25 import BM25Result
from app.retrieval.hybrid import HybridChunk
from app.retrieval.reranker import RerankedChunk


class QueryTracer:
    def __init__(
        self,
        original_query: str,
        retrieval_mode: str = "vector",
        reranker_enabled: bool = False,
        query_rewriting_enabled: bool = False,
        query_rewriting_strategy: str | None = None,
        corrective_rag_enabled: bool = False,
        evaluation_enabled: bool = False,
        grounding_check_enabled: bool = False,
    ):
        self.trace_id = str(uuid.uuid4())
        self.original_query = original_query
        self.retrieval_mode = retrieval_mode
        self.reranker_enabled = reranker_enabled
        self.query_rewriting_enabled = query_rewriting_enabled
        self.query_rewriting_strategy = query_rewriting_strategy
        self.corrective_rag_enabled = corrective_rag_enabled
        self.corrective_rag_retries = 0
        self.evaluation_enabled = evaluation_enabled
        self.grounding_check_enabled = grounding_check_enabled
        self.rewritten_query: str | None = None
        self.retrieval_results: list[dict] = []
        self.reranker_results: list[dict] = []
        self.generation_result: dict = {}
        self.evaluation_result: dict = {}
        self.grounding_result: dict = {}
        self.retrieval_start: float | None = None
        self.retrieval_ms: int = 0
        self.reranker_start: float | None = None
        self.reranker_ms: int = 0
        self.generation_start: float | None = None
        self.generation_ms: int = 0
        self.evaluation_start: float | None = None
        self.evaluation_ms: int = 0

    def start_retrieval(self):
        self.retrieval_start = time.time()

    def record_retrieval(self, chunks: list[RetrievedChunk | BM25Result | HybridChunk]):
        if self.retrieval_start is not None:
            self.retrieval_ms = int((time.time() - self.retrieval_start) * 1000)

        self.retrieval_results = []
        for i, chunk in enumerate(chunks):
            result = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "vector_score": None,
                "bm25_score": None,
                "reranker_score": None,
                "original_rank": i + 1,
                "final_rank": i + 1,
            }

            if isinstance(chunk, RetrievedChunk):
                result["vector_score"] = chunk.similarity_score
            elif isinstance(chunk, BM25Result):
                result["bm25_score"] = chunk.bm25_score
            elif isinstance(chunk, HybridChunk):
                result["vector_score"] = chunk.vector_score
                result["bm25_score"] = chunk.bm25_score

            self.retrieval_results.append(result)

    def start_reranker(self):
        self.reranker_start = time.time()

    def record_reranker(self, chunks: list[RerankedChunk]):
        if self.reranker_start is not None:
            self.reranker_ms = int((time.time() - self.reranker_start) * 1000)

        self.reranker_results = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "original_score": chunk.original_score,
                "reranker_score": chunk.reranker_score,
                "original_rank": chunk.original_rank,
                "reranker_rank": chunk.reranker_rank,
            }
            for chunk in chunks
        ]

    def start_generation(self):
        self.generation_start = time.time()

    def record_generation(
        self,
        answer: str,
        citations: list[str],
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ):
        if self.generation_start is not None:
            self.generation_ms = int((time.time() - self.generation_start) * 1000)

        self.generation_result = {
            "answer": answer,
            "citations": citations,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def start_evaluation(self):
        self.evaluation_start = time.time()

    def record_evaluation(
        self,
        faithfulness_score: float | None = None,
        answer_relevancy_score: float | None = None,
        context_precision_score: float | None = None,
        context_recall_score: float | None = None,
        error: str | None = None,
    ):
        if self.evaluation_start is not None:
            self.evaluation_ms = int((time.time() - self.evaluation_start) * 1000)

        self.evaluation_result = {
            "faithfulness_score": faithfulness_score,
            "answer_relevancy_score": answer_relevancy_score,
            "context_precision_score": context_precision_score,
            "context_recall_score": context_recall_score,
            "error": error,
            "evaluation_ms": self.evaluation_ms,
        }

    def record_grounding(
        self,
        is_grounded: bool,
        grounding_score: float,
        supported_claims: list[str],
        hallucinated_claims: list[str],
        explanation: str | None = None,
    ):
        self.grounding_result = {
            "is_grounded": is_grounded,
            "grounding_score": grounding_score,
            "supported_claims": supported_claims,
            "hallucinated_claims": hallucinated_claims,
            "explanation": explanation,
        }

    async def persist(self, session: AsyncSession) -> str:
        total_ms = self.retrieval_ms + self.reranker_ms + self.generation_ms + self.evaluation_ms

        trace = Trace(
            id=self.trace_id,
            original_query=self.original_query,
            rewritten_query=self.rewritten_query,
            retrieval_mode=self.retrieval_mode,
            reranker_enabled=self.reranker_enabled,
            query_rewriting_enabled=self.query_rewriting_enabled,
            query_rewriting_strategy=self.query_rewriting_strategy,
            corrective_rag_enabled=self.corrective_rag_enabled,
            corrective_rag_retries=self.corrective_rag_retries,
            evaluation_enabled=self.evaluation_enabled,
            grounding_check_enabled=self.grounding_check_enabled,
            retrieval_results={"chunks": self.retrieval_results},
            reranker_results={"chunks": self.reranker_results},
            generation_result=self.generation_result,
            evaluation_result=self.evaluation_result,
            grounding_result=self.grounding_result,
            latency={
                "retrieval_ms": self.retrieval_ms,
                "reranker_ms": self.reranker_ms,
                "generation_ms": self.generation_ms,
                "evaluation_ms": self.evaluation_ms,
                "total_ms": total_ms,
            },
        )

        session.add(trace)
        await session.commit()

        return self.trace_id
