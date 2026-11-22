"""RAGAS judging, run *after* the timed pipeline pass (so judge latency never
contaminates pipeline latency) and concurrently for throughput."""
from __future__ import annotations

import math
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from app.config.settings import settings  # noqa: E402

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _judge_objects(model: str):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # bypass_n=True: AnswerRelevancy asks for n=3 generations (strictness=3);
    # through the LangChain wrapper ChatOpenAI returned only 1, silently
    # degrading the metric. bypass_n sends n separate requests instead.
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0, seed=42),
        bypass_n=True,
    )
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    )
    return llm, emb


def ragas_judge(records: list[dict], questions: dict, model: str, max_workers: int = 16):
    """Score records with RAGAS. Returns (per-record score dicts, token usage dict).

    Metrics: Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference,
    LLMContextRecall (reference = HotpotQA gold answer). Errors -> NaN, never
    imputed.
    """
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    llm, emb = _judge_objects(model)
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    ok = [r for r in records if not r.get("error")]
    scores = {r["qid"]: {m: float("nan") for m in METRICS} for r in records}
    if not ok:
        return scores, {"input_tokens": 0, "output_tokens": 0}
    samples = [
        SingleTurnSample(
            user_input=questions[r["qid"]]["question"],
            response=r["answer"],
            retrieved_contexts=[c["text"] for c in r["contexts"]] or [""],
            reference=questions[r["qid"]]["reference"],
        )
        for r in ok
    ]
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=max_workers, timeout=180, max_retries=6, seed=42),
        raise_exceptions=False,
        token_usage_parser=get_token_usage_for_openai,
        show_progress=False,
    )
    df = result.to_pandas()
    name_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "llm_context_precision_with_reference": "context_precision",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
    }
    for r, (_, row) in zip(ok, df.iterrows()):
        for col, ours in name_map.items():
            if col in row and row[col] is not None:
                v = float(row[col])
                scores[r["qid"]][ours] = v if not math.isinf(v) else float("nan")
    usage = result.total_tokens()
    return scores, {"input_tokens": int(usage.input_tokens), "output_tokens": int(usage.output_tokens)}


# ---- deterministic metrics (no LLM) --------------------------------------

def retrieval_metrics(contexts: list[dict], gold_titles: list[str], k: int = 5) -> dict:
    titles = [c["title"] for c in contexts[:k]]
    gold = set(gold_titles)
    first = next((i for i, t in enumerate(titles, 1) if t in gold), None)
    found = gold & set(titles)
    return {
        "hit_rate@5": 1.0 if first else 0.0,
        "mrr@5": 1.0 / first if first else 0.0,
        "recall@5": len(found) / len(gold) if gold else float("nan"),
        "all_gold@5": 1.0 if gold and found == gold else 0.0,
    }


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_contains_gold(answer: str, reference: str) -> float:
    """1 if the normalized gold answer appears as a whole-word span in the
    response (SQuAD-style normalization). Supplementary, not a RAGAS metric."""
    a, g = _norm(answer), _norm(reference)
    if not g:
        return float("nan")
    return 1.0 if re.search(rf"(^| ){re.escape(g)}( |$)", a) else 0.0
