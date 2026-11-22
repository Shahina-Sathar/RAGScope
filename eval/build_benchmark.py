"""Build the frozen benchmark (test set + corpus) from HotpotQA and index it.

Source: HotpotQA dev set, "distractor" setting (Yang et al., 2018, CC BY-SA 4.0).
Questions, answers and supporting-fact labels were written by human crowd
workers. No synthetic / LLM-generated questions are used.

Design
- Test set: N_EVAL questions sampled with a fixed seed, stratified by HotpotQA
  question type (bridge = multi-hop chain, comparison = multi-hop compare).
- Corpus: every context paragraph (2 gold + 8 adversarial distractors) of the
  test questions, plus the paragraphs of N_POOL extra dev questions, so the
  retriever must search a few thousand Wikipedia paragraphs rather than 10.
  Paragraphs are deduplicated by Wikipedia title.
- Chunking: the app's own chunker (RecursiveCharacterTextSplitter, 1000/200).
- Index: a dedicated Pinecone namespace, so the app's default namespace is
  untouched. BM25 corpus is written to eval/data/corpus.jsonl.

Usage:
    python -m eval.build_benchmark            # build testset + corpus + index
    python -m eval.build_benchmark --no-index # files only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "backend"))

RAW = ROOT / "data" / "raw" / "hotpot_qa_distractor_validation.parquet"
RAW_URL = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/"
    "distractor/validation/0.parquet"
)
RAW_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
TESTSET = ROOT / "testset.json"
CORPUS = ROOT / "data" / "corpus.jsonl"
NAMESPACE = "ablation-hotpotqa-v1"

SEED = 42
N_BRIDGE = 70
N_COMPARISON = 30
N_POOL = 400  # extra questions whose paragraphs are added as corpus noise
WARMUP_ID_KEY = "warmup"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows():
    import pyarrow.parquet as pq

    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Download it with:\n  curl -L -o {RAW} {RAW_URL}")
    got = sha256(RAW)
    if got != RAW_SHA256:
        raise SystemExit(f"Checksum mismatch for {RAW}: {got}")
    return pq.read_table(RAW).to_pylist()


def paragraph_text(title: str, sentences: list[str]) -> str:
    return f"{title}\n" + "".join(sentences).strip()


def build():
    rows = load_rows()
    rows.sort(key=lambda r: r["id"])  # deterministic order independent of file layout
    rng = random.Random(SEED)

    bridge = [r for r in rows if r["type"] == "bridge"]
    comparison = [r for r in rows if r["type"] == "comparison"]
    picked = rng.sample(bridge, N_BRIDGE) + rng.sample(comparison, N_COMPARISON)
    picked_ids = {r["id"] for r in picked}
    rest = [r for r in rows if r["id"] not in picked_ids]
    pool = rng.sample(rest, N_POOL + 1)
    warmup, pool = pool[0], pool[1:]

    # Corpus: test-question paragraphs take precedence over pool versions.
    paragraphs: dict[str, str] = {}
    for r in picked + [warmup] + pool:
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            paragraphs.setdefault(title, paragraph_text(title, sents))

    questions = []
    for i, r in enumerate(picked, 1):
        ctx = dict(zip(r["context"]["title"], r["context"]["sentences"]))
        gold_titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
        facts = [
            ctx[t][s]
            for t, s in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])
            if t in ctx and s < len(ctx[t])
        ]
        questions.append(
            {
                "id": f"HQ{i:03d}",
                "hotpot_id": r["id"],
                "question": r["question"],
                "reference": r["answer"],
                "type": r["type"],
                "level": r["level"],
                "gold_titles": gold_titles,
                "reference_contexts": [paragraphs[t] for t in gold_titles],
                "supporting_facts": [s.strip() for s in facts],
            }
        )

    testset = {
        "name": "hotpotqa-dev-distractor-100",
        "labelling": "human (HotpotQA crowd-written questions, answers and supporting facts)",
        "synthetic": False,
        "source": "HotpotQA dev set, distractor setting (Yang et al., 2018), CC BY-SA 4.0",
        "source_url": RAW_URL,
        "source_sha256": RAW_SHA256,
        "sampling": {
            "seed": SEED,
            "bridge": N_BRIDGE,
            "comparison": N_COMPARISON,
            "corpus_pool_questions": N_POOL,
        },
        "warmup_question": warmup["question"],
        "questions": questions,
    }
    TESTSET.write_text(json.dumps(testset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "testset.sha256").write_text(sha256(TESTSET) + "  testset.json\n")

    # Chunk with the app's own chunker and settings.
    from app.config.settings import settings
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.parser import PDFPage

    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for title in sorted(paragraphs):
        page = PDFPage(page_number=1, text=paragraphs[title])
        for c in chunk_pages([page], title, settings.chunk_size, settings.chunk_overlap):
            chunks.append(
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "document_name": c.document_name,
                    "page_number": c.page_number,
                }
            )
    with open(CORPUS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"test questions: {len(questions)}  paragraphs: {len(paragraphs)}  chunks: {len(chunks)}")
    return chunks


def index(chunks: list[dict], batch: int = 100):
    import hashlib as _h

    from langchain_openai import OpenAIEmbeddings
    from pinecone import Pinecone

    from app.config.settings import settings

    pc = Pinecone(api_key=settings.pinecone_api_key)
    idx = pc.Index(settings.pinecone_index_name)
    stats = idx.describe_index_stats()
    existing = (stats.namespaces or {}).get(NAMESPACE)
    if existing and existing.vector_count == len(chunks):
        print(f"namespace {NAMESPACE} already holds {existing.vector_count} vectors; skipping")
        return
    if existing:
        print(f"namespace {NAMESPACE} has {existing.vector_count} vectors (expected "
              f"{len(chunks)}); clearing it")
        idx.delete(delete_all=True, namespace=NAMESPACE)

    emb = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vecs = emb.embed_documents([c["text"] for c in part])
        idx.upsert(
            vectors=[
                {
                    # Pinecone ids must be ASCII; titles can be unicode.
                    "id": _h.sha1(c["chunk_id"].encode()).hexdigest(),
                    "values": v,
                    "metadata": c,
                }
                for c, v in zip(part, vecs)
            ],
            namespace=NAMESPACE,
        )
        print(f"  upserted {i + len(part)}/{len(chunks)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args()
    chunks = build()
    if not args.no_index:
        index(chunks)


if __name__ == "__main__":
    main()
