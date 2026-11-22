"""Aggregate judged runs into per-config jsonl, summary.csv, deltas.csv and REPORT.md.

Every number in the outputs is computed here from the raw per-question files;
nothing is typed in by hand. Hand-written interpretation lives in
eval/report_notes.md and is included verbatim.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from eval.stats import bootstrap_ci, holm, paired_test, percentile

ROOT = Path(__file__).resolve().parent

# OpenAI standard-tier prices, USD per 1M tokens, read from
# https://developers.openai.com/api/docs/pricing on 2026-11-21.
# Cached-input discounts are ignored (all input billed at full price), which
# can only over-state cost.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

QUALITY = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
DET = ["hit_rate@5", "mrr@5", "recall@5", "all_gold@5", "answer_contains_gold"]
METRICS = QUALITY + DET
STAGES = ["rewriting", "retrieval", "reranking", "corrective_grading", "corrective_rewrite",
          "corrective_retrieval", "corrective_reranking", "corrective_loop_total", "generation"]
STEPS = [("C0", "C1"), ("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C0", "C4")]
EXTRA_STEPS = [("C3", "C4_legacy"), ("C0", "C4_legacy")]


def _read(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _val(rec, m):
    if rec.get("error"):
        return float("nan")
    v = rec["scores"].get(m) if m in QUALITY else rec["det"].get(m)
    return float("nan") if v is None else float(v)


def load(configs, runs, results: Path, judge_model: str):
    data = {}
    for c in configs:
        recs = []
        for r in range(1, runs + 1):
            p = results / "raw" / f"{c}_run{r}.judged_{judge_model}.jsonl"
            if p.exists():
                recs += _read(p)
        if recs:
            data[c] = recs
    return data


def per_question(recs, qids, m):
    """Per-question value = mean over runs (NaN if no run produced a value)."""
    out = []
    for q in qids:
        vals = [_val(r, m) for r in recs if r["qid"] == q]
        vals = [v for v in vals if not math.isnan(v)]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return np.array(out)


def run_means(recs, m):
    runs = sorted({r["run"] for r in recs})
    means = []
    for r in runs:
        v = np.array([_val(x, m) for x in recs if x["run"] == r])
        v = v[~np.isnan(v)]
        if v.size:
            means.append(float(v.mean()))
    return means


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def summarize(configs, runs, ts, results: Path, judge_model: str):
    from eval.pipeline import CONFIGS

    qids = [q["id"] for q in ts["questions"]]
    data = load(configs, runs, results, judge_model)
    if not data:
        print("[summarize] no judged results found")
        return
    gen_price = PRICES[__import__("app.config.settings", fromlist=["settings"]).settings.openai_llm_model]
    judge_usage_p = results / "raw" / f"judge_usage_{judge_model}.json"
    judge_usage = json.loads(judge_usage_p.read_text()) if judge_usage_p.exists() else {}

    # per-config jsonl (one line per question x run)
    for c, recs in data.items():
        with open(results / f"{c}.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(recs, key=lambda x: (x["run"], x["qid"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pq = {c: {m: per_question(recs, qids, m) for m in METRICS} for c, recs in data.items()}
    rows = []
    for c, recs in data.items():
        ok = [r for r in recs if not r.get("error")]
        row = {"config": c, "description": CONFIGS[c].description,
               "runs": len({r["run"] for r in recs}), "question_runs": len(recs),
               "errors": len(recs) - len(ok)}
        for m in METRICS:
            mean, lo, hi, n = bootstrap_ci(pq[c][m])
            rm = run_means(recs, m)
            row.update({m: mean, f"{m}_ci_lo": lo, f"{m}_ci_hi": hi, f"{m}_n": n,
                        f"{m}_run_sd": float(np.std(rm, ddof=1)) if len(rm) > 1 else float("nan"),
                        f"{m}_nan": sum(1 for r in recs if math.isnan(_val(r, m)))})
        lat = [r["total_ms"] for r in ok]
        row["latency_p50_ms"] = percentile(lat, 50)
        row["latency_p95_ms"] = percentile(lat, 95)
        row["latency_mean_ms"] = float(np.mean(lat)) if lat else float("nan")
        for s in STAGES:
            v = [r["stage_ms"].get(s, 0.0) for r in ok]
            if any(x > 0 for x in v):
                row[f"stage_{s}_mean_ms"] = float(np.mean(v))
                row[f"stage_{s}_p95_ms"] = percentile(v, 95)
        row["llm_calls_per_query"] = float(np.mean([r["llm_calls"] for r in ok])) if ok else float("nan")
        row["input_tokens_per_query"] = float(np.mean([r["input_tokens"] for r in ok])) if ok else float("nan")
        row["output_tokens_per_query"] = float(np.mean([r["output_tokens"] for r in ok])) if ok else float("nan")
        row["cost_per_query_usd"] = (row["input_tokens_per_query"] * gen_price[0]
                                     + row["output_tokens_per_query"] * gen_price[1]) / 1e6
        if "retries" in (ok[0] if ok else {}):
            row["mean_corrective_retries"] = float(np.mean([r.get("retries", 0) for r in ok]))
            row["share_with_retry"] = float(np.mean([r.get("retries", 0) > 0 for r in ok]))
        ju = [v for k, v in judge_usage.items() if k.startswith(c + "_run")]
        if ju:
            jp = PRICES.get(judge_model, (float("nan"), float("nan")))
            tin = sum(x["input_tokens"] for x in ju)
            tout = sum(x["output_tokens"] for x in ju)
            row["judge_cost_per_query_usd"] = (tin * jp[0] + tout * jp[1]) / 1e6 / len(recs)
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(results / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # deltas
    deltas = []
    for m in METRICS:
        block = []
        for a, b in STEPS + EXTRA_STEPS:
            if a in pq and b in pq:
                t = paired_test(pq[a][m], pq[b][m])
                block.append({"metric": m, "from": a, "to": b, **t,
                              "family": "main" if (a, b) in STEPS else "extra"})
        main = [d for d in block if d["family"] == "main"]
        for d, adj in zip(main, holm([d["p"] for d in main])):
            d["p_holm"] = adj
        for d in block:
            d.setdefault("p_holm", float("nan"))
        deltas += block
    if deltas:
        with open(results / "deltas.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(deltas[0]))
            w.writeheader()
            w.writerows(deltas)

    write_report(rows, deltas, ts, data, results, judge_model, runs)
    print(f"[summarize] wrote {results / 'summary.csv'}, deltas.csv, REPORT.md")


def write_report(rows, deltas, ts, data, results: Path, judge_model, runs):
    from app.config.settings import settings

    corpus_n = sum(1 for _ in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8"))
    qs = ts["questions"]
    n_bridge = sum(q["type"] == "bridge" for q in qs)
    by = {r["config"]: r for r in rows}
    L = []
    L.append("# RAGScope component ablation study\n")
    L.append("_All tables below are generated by `eval/summarize.py` from the raw per-question "
             "files in `eval/results/`. Nothing in them is typed by hand._\n")
    L.append("## Setup\n")
    L.append(f"- **Test set:** {len(qs)} questions from {ts['source']}. **Human-labelled** "
             f"(crowd-written questions, answers and supporting-fact labels); **not synthetic**. "
             f"Sampled with seed {ts['sampling']['seed']}: {n_bridge} bridge (multi-hop chain) + "
             f"{len(qs) - n_bridge} comparison (multi-hop comparison/yes-no) questions. Frozen in "
             f"`eval/testset.json` (sha256 checked on every run).")
    L.append(f"- **Corpus:** {corpus_n} chunks = all context paragraphs (2 gold + 8 adversarial "
             f"distractors each) of the test questions plus those of "
             f"{ts['sampling']['corpus_pool_questions']} other dev questions, deduplicated by "
             f"Wikipedia title. Chunked with the app's chunker ({settings.chunk_size}/"
             f"{settings.chunk_overlap}). Pinecone namespace `ablation-hotpotqa-v1`.")
    L.append(f"- **Models:** generator, query rewriter and corrective-loop grader = "
             f"`{settings.openai_llm_model}` (temperature 0, seed 42); embeddings = "
             f"`{settings.openai_embedding_model}`; reranker = `cross-encoder/ms-marco-MiniLM-L-6-v2` "
             f"(local CPU); RAGAS judge = `{judge_model}` (temperature 0) with "
             f"`{settings.openai_embedding_model}` for answer relevancy.")
    L.append(f"- **Repetitions:** {runs} full runs per config; per-question scores are averaged "
             "over runs before bootstrapping. One untimed warm-up query per config per run; "
             "config order shuffled per run.")
    L.append("- **Held fixed:** corpus, chunking, embedding model, generator model and prompt, "
             "judge model, final top-k = 5.\n")
    L.append("| Config | What it adds |\n|---|---|")
    from eval.pipeline import CONFIGS
    for c in by:
        L.append(f"| {c} | {CONFIGS[c].description} |")
    L.append("")

    def table(metrics, title):
        L.append(f"## {title}\n")
        L.append("Mean over questions, 95% bootstrap CI (10,000 resamples over questions) in brackets; "
                 "± = SD of the per-run means across runs.\n")
        L.append("| Config | " + " | ".join(metrics) + " |")
        L.append("|---|" + "---|" * len(metrics))
        for c, r in by.items():
            cells = []
            for m in metrics:
                sd = r.get(f"{m}_run_sd", float("nan"))
                sd_s = "" if math.isnan(sd) else f" ±{sd:.3f}"
                cells.append(f"{fmt(r[m])} [{fmt(r[m + '_ci_lo'])}, {fmt(r[m + '_ci_hi'])}]{sd_s}")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")

    table(QUALITY, f"RAGAS quality (judge: {judge_model})")
    table(["hit_rate@5", "mrr@5", "recall@5", "all_gold@5"],
          "Retrieval (exact, against HotpotQA gold supporting paragraphs)")
    table(["answer_contains_gold"],
          "Answer correctness proxy (normalized gold answer appears in response; no LLM)")

    L.append("## Deltas between configurations\n")
    L.append("Paired over questions. Δ = to − from. CI = 95% paired bootstrap of the mean "
             "difference. p = two-sided sign-flip permutation test (10,000 permutations); "
             "p_holm = Holm-adjusted across the 5 main steps for that metric. "
             "**Bold** = significant at p_holm < 0.05; everything else should be read as "
             "*no detectable difference* on this benchmark.\n")
    L.append("| Metric | Step | From | To | Δ abs | Δ rel | 95% CI of Δ | p | p_holm |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in deltas:
        sig = d["family"] == "main" and not math.isnan(d["p_holm"]) and d["p_holm"] < 0.05
        rel = "n/a" if math.isnan(d["rel"]) else f"{d['rel'] * 100:+.1f}%"
        step = f"{d['from']}→{d['to']}"
        line = (f"| {d['metric']} | {step} | {fmt(d['mean_a'])} | {fmt(d['mean_b'])} | "
                f"{d['delta']:+.3f} | {rel} | [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | "
                f"{d['p']:.4f} | {fmt(d['p_holm'], 4)} |")
        L.append(line.replace(f"| {step} |", f"| **{step}** |") if sig else line)
    L.append("")

    L.append("## Latency\n")
    L.append("End-to-end pipeline wall-clock per query (judge excluded), all question-runs pooled; "
             "per-stage columns are means in ms. Corrective-loop stages are the work *beyond* the "
             "first retrieval pass. Measured from a home WSL2 machine to OpenAI/Pinecone over "
             "the public internet, so absolute values include network latency.\n")
    stages = [s for s in STAGES if any(f"stage_{s}_mean_ms" in r for r in rows)]
    L.append("| Config | p50 ms | p95 ms | " + " | ".join(stages) + " |")
    L.append("|---|---|---|" + "---|" * len(stages))
    for c, r in by.items():
        L.append(f"| {c} | {fmt(r['latency_p50_ms'], 0)} | {fmt(r['latency_p95_ms'], 0)} | "
                 + " | ".join(fmt(r.get(f'stage_{s}_mean_ms', float('nan')), 0) for s in stages)
                 + " |")
    L.append("")

    L.append("## Cost and reliability\n")
    L.append("Pipeline LLM usage only (generator, rewriter, grader); embeddings and Pinecone "
             "excluded. Cost uses OpenAI standard list prices read on 2026-11-21, all input "
             "billed as uncached.\n")
    L.append("| Config | LLM calls/query | input tok/query | output tok/query | est. $/query | "
             "$/1k queries | corrective retries/query | queries with ≥1 retry | errored queries | "
             "RAGAS NaNs (F/AR/CP/CR) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c, r in by.items():
        nans = "/".join(str(r[f"{m}_nan"]) for m in QUALITY)
        L.append(f"| {c} | {fmt(r['llm_calls_per_query'], 2)} | {fmt(r['input_tokens_per_query'], 0)} | "
                 f"{fmt(r['output_tokens_per_query'], 0)} | {r['cost_per_query_usd']:.6f} | "
                 f"{r['cost_per_query_usd'] * 1000:.3f} | {fmt(r.get('mean_corrective_retries', float('nan')), 2)} | "
                 f"{fmt(r.get('share_with_retry', float('nan')) * 100, 1) + '%' if 'share_with_retry' in r else 'n/a'} | "
                 f"{r['errors']} / {r['question_runs']} | {nans} |")
    L.append("")
    jc = [r.get("judge_cost_per_query_usd") for r in rows if r.get("judge_cost_per_query_usd")]
    if jc:
        L.append(f"RAGAS judging cost (not part of the pipeline): mean ${np.mean(jc):.5f} per "
                 f"question-run with `{judge_model}`.\n")

    notes = ROOT / "report_notes.md"
    if notes.exists():
        L.append(notes.read_text(encoding="utf-8").strip() + "\n")

    L.append("## Reproduce\n")
    L.append("```bash\ncd <repo root>\n"
             "(cd backend && uv sync)                       # exact locked dependencies\n"
             "mkdir -p eval/data/raw && curl -L -o eval/data/raw/hotpot_qa_distractor_validation.parquet \\\n"
             "  https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet\n"
             "backend/.venv/bin/python -m eval.build_benchmark   # verifies sha256, rebuilds testset/corpus, indexes Pinecone namespace\n"
             f"backend/.venv/bin/python -m eval.run_ablation --configs all --runs {runs} --judge-model {judge_model}\n"
             "```\n")
    L.append("Needs `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_HOST` in `.env` at the repo "
             "root. `build_benchmark` refuses to run if the parquet checksum differs; "
             "`run_ablation` refuses to run if `testset.json` differs from its frozen sha256. "
             "Runs are resumable: delete `eval/results/raw/` to start over.\n")
    # Only the default results directory may overwrite eval/REPORT.md; smoke tests
    # with --results elsewhere write their report next to their own results.
    target = ROOT / "REPORT.md" if results.resolve() == (ROOT / "results").resolve() \
        else results / "REPORT.md"
    target.write_text("\n".join(L), encoding="utf-8")
