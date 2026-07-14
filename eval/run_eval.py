"""Evaluate every adapter iteration with auto-demo sessions (Phase 4).

For each served checkpoint (PTO/GRPO iteration 1-10, plus optionally the base
model), runs N simulated practice sessions (fixed thesis patient persona,
sampled at the app's default temperatures) and scores them with the thesis
judges: per-turn Q1, end-of-session Q2 + MITI globals (gpt-4o-mini).

Outputs under eval/results/: raw per-session JSON, an aggregated markdown
table, and docs/eval_scores.png (score vs. training iteration, PTO vs GRPO).

Usage:
    # vLLM server must be running with all adapters (scripts/serve.sh)
    .venv/bin/python eval/run_eval.py --sessions 3 --workers 4

Honest-methodology notes (also in README): the judge is an LLM (gpt-4o-mini),
sessions are short (3 patient turns) and sampled, and N is small — error bars
are wide; this reproduces the *shape* of the thesis evaluation in the deployed
system, it is not a re-run of the thesis experiments.
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.graph import run_demo  # noqa: E402

TURN_QS = ["Q1"]
REPORT_QS = ["Q2", "MITI"]


def run_one(model: str, session_idx: int, max_patient_turns: int) -> dict:
    t0 = time.time()
    state = run_demo(model, max_patient_turns=max_patient_turns,
                     turn_questionnaires=TURN_QS, report_questionnaires=REPORT_QS)
    q1_turns = [t["means"]["Q1"] for t in state["turn_scores"]]
    report = state["report"]
    return {
        "model": model,
        "session": session_idx,
        "q1_turn_means": q1_turns,
        "q1_session_mean": round(statistics.mean(q1_turns), 3) if q1_turns else None,
        "q2_mean": report["means"]["Q2"],
        "miti_global_mean": report["means"]["MITI"],
        "n_messages": len(state["messages"]) - 1,
        "wall_s": round(time.time() - t0, 1),
    }


def aggregate(rows: list[dict]) -> dict:
    def mstd(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return round(statistics.mean(vals), 2), round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0

    out = {}
    for metric in ("q1_session_mean", "q2_mean", "miti_global_mean"):
        mean, std = mstd([r[metric] for r in rows])
        out[metric] = {"mean": mean, "std": std}
    out["n"] = len(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3, help="sessions per checkpoint")
    parser.add_argument("--max-patient-turns", type=int, default=3)
    parser.add_argument("--iters", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--methods", nargs="*", default=["pto", "grpo"])
    parser.add_argument("--include-base", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    models = [f"mi-coach-{m}-iter{i}" for m in args.methods for i in args.iters]
    if args.include_base:
        import httpx
        served = httpx.get("http://localhost:8000/v1/models", timeout=10).json()["data"]
        base = next(m["id"] for m in served if not m["id"].startswith("mi-coach-"))
        models.append(base)

    jobs = [(model, s) for model in models for s in range(args.sessions)]
    print(f"{len(models)} checkpoints x {args.sessions} sessions = {len(jobs)} sessions "
          f"(~{len(jobs) * (2 * args.max_patient_turns + 3)} gpt-4o-mini calls)")

    results = []
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, model, s, args.max_patient_turns): (model, s) for model, s in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            model, s = futures[future]
            try:
                row = future.result()
                results.append(row)
                print(f"[{i}/{len(jobs)}] {model} #{s}: Q1={row['q1_session_mean']} "
                      f"Q2={row['q2_mean']} MITI={row['miti_global_mean']} ({row['wall_s']}s)")
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {model} #{s}: FAILED — {e}")

    out_dir = REPO / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    agg = {model: aggregate([r for r in results if r["model"] == model]) for model in models}
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps({"raw": results, "aggregate": agg}, indent=2))

    # Markdown table
    lines = ["| Checkpoint | Q1 (per-turn) | Q2 (17 items) | MITI globals | n |",
             "|---|---|---|---|---|"]
    for model in models:
        a = agg[model]
        def fmt(m):
            v = a[m]
            return f"{v['mean']} ± {v['std']}" if v["mean"] is not None else "—"
        lines.append(f"| {model} | {fmt('q1_session_mean')} | {fmt('q2_mean')} | {fmt('miti_global_mean')} | {a['n']} |")
    table = "\n".join(lines)
    (out_dir / f"eval-{stamp}.md").write_text(table + "\n")
    print("\n" + table)

    # Plot: score vs iteration per method; base model as dashed reference.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("q1_session_mean", "Q1 per-turn mean"), ("q2_mean", "Q2 mean"),
               ("miti_global_mean", "MITI globals mean")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {"pto": "#d62728", "grpo": "#1f77b4"}
    for ax, (metric, title) in zip(axes, metrics):
        for method in args.methods:
            xs, ys, es = [], [], []
            for i in args.iters:
                a = agg.get(f"mi-coach-{method}-iter{i}", {}).get(metric)
                if a and a["mean"] is not None:
                    xs.append(i); ys.append(a["mean"]); es.append(a["std"] or 0)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method.upper(),
                        color=colors.get(method))
        if args.include_base and agg.get(base, {}).get(metric, {}).get("mean") is not None:
            ax.axhline(agg[base][metric]["mean"], ls="--", c="gray", label="base model")
        ax.set_title(title); ax.set_xlabel("training iteration"); ax.grid(alpha=0.3)
        ax.set_xticks(args.iters)
    axes[0].set_ylabel("judge score (1-5)"); axes[0].legend()
    fig.suptitle("MI Coach: judge scores vs training iteration "
                 f"(auto-demo, n={args.sessions}/checkpoint, gpt-4o-mini judge)")
    fig.tight_layout()
    plot_path = REPO / "docs" / "eval_scores.png"
    fig.savefig(plot_path, dpi=130)
    print(f"\nSaved: {out_dir}/eval-{stamp}.{{json,md}} and {plot_path}")


if __name__ == "__main__":
    main()
