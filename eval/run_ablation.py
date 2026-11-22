"""Run the RAGScope component ablation study end to end.

    python -m eval.run_ablation --configs all --runs 3

Phases (each is resumable; finished config/run files are skipped):
  1. pipeline  - every config on every question, sequentially, one untimed
                 warm-up query per config/run first. Config order is shuffled
                 per run (seeded) so slow drift in API latency does not
                 systematically favour one config.
  2. judge     - RAGAS scoring of the saved answers (concurrent; judge time is
                 never part of pipeline latency) + deterministic metrics.
  3. summarize - eval/results/<config>.jsonl, summary.csv, deltas.csv, REPORT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

TESTSET = ROOT / "testset.json"


def load_testset(limit: int | None = None) -> dict:
    expected = (ROOT / "testset.sha256").read_text().split()[0]
    got = hashlib.sha256(TESTSET.read_bytes()).hexdigest()
    if got != expected:
        raise SystemExit(f"testset.json changed (sha256 {got} != frozen {expected}); refusing to run")
    ts = json.loads(TESTSET.read_text(encoding="utf-8"))
    if limit:
        ts["questions"] = ts["questions"][:limit]
    return ts


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def phase_pipeline(configs, runs, ts, raw_dir: Path):
    from eval.pipeline import CONFIGS, AblationPipeline, Components

    todo = [
        (r, c) for r in range(1, runs + 1) for c in configs
        if len(read_jsonl(raw_dir / f"{c}_run{r}.pipeline.jsonl")) != len(ts["questions"])
    ]
    if not todo:
        print("[pipeline] all config/run files complete")
        return
    print("[pipeline] loading components ...", flush=True)
    comp = Components()
    for r in range(1, runs + 1):
        order = [c for c in configs if (r, c) in todo]
        random.Random(1000 + r).shuffle(order)
        for c in order:
            pipe = AblationPipeline(CONFIGS[c], comp)
            try:
                pipe.run(ts["warmup_question"])  # warm-up, discarded
            except Exception as e:  # warm-up failure is reported, not hidden
                print(f"[pipeline] warm-up failed for {c}: {e}", flush=True)
            rows = []
            t0 = time.time()
            for q in ts["questions"]:
                rec = {"config": c, "run": r, "qid": q["id"]}
                try:
                    rec.update(pipe.run(q["question"]))
                    rec["error"] = None
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    rec["traceback"] = traceback.format_exc(limit=5)
                rows.append(rec)
            write_jsonl(raw_dir / f"{c}_run{r}.pipeline.jsonl", rows)
            errs = sum(1 for x in rows if x["error"])
            print(f"[pipeline] run {r} {c}: {len(rows)} q in {time.time() - t0:.0f}s, "
                  f"{errs} errors", flush=True)


def phase_judge(configs, runs, ts, raw_dir: Path, judge_model: str, workers: int):
    from eval.judge import answer_contains_gold, ragas_judge, retrieval_metrics

    qmap = {q["id"]: q for q in ts["questions"]}
    usage_file = raw_dir / f"judge_usage_{judge_model}.json"
    usage = json.loads(usage_file.read_text()) if usage_file.exists() else {}
    for r in range(1, runs + 1):
        for c in configs:
            out = raw_dir / f"{c}_run{r}.judged_{judge_model}.jsonl"
            recs = read_jsonl(raw_dir / f"{c}_run{r}.pipeline.jsonl")
            if not recs or len(read_jsonl(out)) == len(recs):
                continue
            t0 = time.time()
            scores, tok = ragas_judge(recs, qmap, judge_model, max_workers=workers)
            for rec in recs:
                q = qmap[rec["qid"]]
                rec["judge_model"] = judge_model
                rec["scores"] = scores[rec["qid"]]
                if rec.get("error"):
                    rec["det"] = {}
                else:
                    rec["det"] = retrieval_metrics(rec["contexts"], q["gold_titles"])
                    rec["det"]["answer_contains_gold"] = answer_contains_gold(
                        rec["answer"], q["reference"])
            write_jsonl(out, recs)
            usage[f"{c}_run{r}"] = tok
            usage_file.write_text(json.dumps(usage, indent=2))
            nan = {m: sum(1 for x in recs if x["scores"][m] != x["scores"][m])
                   for m in recs[0]["scores"]}
            print(f"[judge] run {r} {c}: {time.time() - t0:.0f}s tokens={tok} NaNs={nan}",
                  flush=True)


def main():
    from eval.pipeline import CONFIGS, MAIN_ORDER

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--configs", default="all",
                    help="'all' or comma list, e.g. C0,C1 (all = C0..C4 + C4_legacy)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke tests)")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--judge-workers", type=int, default=16)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--phases", default="pipeline,judge,summarize")
    args = ap.parse_args()

    configs = MAIN_ORDER + ["C4_legacy"] if args.configs == "all" else args.configs.split(",")
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; choose from {list(CONFIGS)}")
    ts = load_testset(args.limit)
    results = Path(args.results)
    raw_dir = results / "raw"
    phases = args.phases.split(",")
    if "pipeline" in phases:
        phase_pipeline(configs, args.runs, ts, raw_dir)
    if "judge" in phases:
        phase_judge(configs, args.runs, ts, raw_dir, args.judge_model, args.judge_workers)
    if "summarize" in phases:
        from eval.summarize import summarize
        summarize(configs, args.runs, ts, results, args.judge_model)


if __name__ == "__main__":
    main()
