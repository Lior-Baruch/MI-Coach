"""Evaluate adapter iterations with auto-demo sessions (Phase 4).

For each served checkpoint (PTO/GRPO iterations, plus optionally the base
model), runs N simulated practice sessions and scores them with the thesis
judges: per-turn Q1, end-of-session Q2 + MITI globals (gpt-4o-mini). With
--personas, sweeps several thesis patient personas per checkpoint and reports
per-persona plus aggregate scores.

Outputs under eval/results/: raw per-session JSON, an aggregated markdown
table, docs/eval_scores.png (score vs. training iteration) and, for
multi-persona runs, docs/eval_personas.png (score by persona per model).

Usage:
    # vLLM server must be running with all adapters (scripts/serve.sh)
    .venv/bin/python eval/run_eval.py --sessions 3 --workers 4
    # persona sweep over the thesis-best checkpoints only:
    .venv/bin/python eval/run_eval.py --models mi-coach-pto-iter10 mi-coach-grpo-iter8 \
        --personas all --sessions 2

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.graph import build_patient_persona, run_demo  # noqa: E402

TURN_QS = ["Q1"]
REPORT_QS = ["Q2", "MITI"]

# Named thesis-permutation personas for the sweep. "emma-smoking" is the app
# default persona and matches earlier single-persona eval runs.
PERSONAS = {
    "emma-smoking": dict(gender="Female", age=61, problem="Smoking", problem_time="ManyYears",
                         tried_to_solve="ManyTimes", cooperation="StartLowAndChangesToHigh"),
    "noah-smoking-resistant": dict(gender="Male", age=27, problem="Smoking", problem_time="FewMonths",
                                   tried_to_solve="Never", cooperation="Low"),
    "ava-obesity-eager": dict(gender="Female", age=27, problem="Obesity", problem_time="FewMonths",
                              tried_to_solve="Never", cooperation="High"),
    "liam-obesity": dict(gender="Male", age=61, problem="Obesity", problem_time="ManyYears",
                         tried_to_solve="ManyTimes", cooperation="StartLowAndChangesToHigh"),
}


def run_one(model: str, persona_name: str, session_idx: int, max_patient_turns: int) -> dict:
    t0 = time.time()
    persona = build_patient_persona(**PERSONAS[persona_name])
    state = run_demo(model, max_patient_turns=max_patient_turns, patient_system_prompt=persona,
                     turn_questionnaires=TURN_QS, report_questionnaires=REPORT_QS)
    q1_turns = [t["means"]["Q1"] for t in state["turn_scores"]]
    report = state["report"]
    usage = state.get("openai_usage") or {}
    return {
        "model": model,
        "persona": persona_name,
        "session": session_idx,
        "q1_turn_means": q1_turns,
        "q1_session_mean": round(statistics.mean(q1_turns), 3) if q1_turns else None,
        "q2_mean": report["means"]["Q2"],
        "miti_global_mean": report["means"]["MITI"],
        "n_messages": len(state["messages"]) - 1,
        "cost_usd": usage.get("cost_usd", 0.0),
        "wall_s": round(time.time() - t0, 1),
    }


METRICS = ("q1_session_mean", "q2_mean", "miti_global_mean")


def aggregate(rows: list[dict]) -> dict:
    def mstd(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return round(statistics.mean(vals), 2), round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0

    out = {}
    for metric in METRICS:
        mean, std = mstd([r[metric] for r in rows])
        out[metric] = {"mean": mean, "std": std}
    out["n"] = len(rows)
    out["cost_usd"] = round(sum(r.get("cost_usd", 0.0) for r in rows), 4)
    return out


def fmt_cell(agg: dict, metric: str) -> str:
    v = agg[metric]
    return f"{v['mean']} ± {v['std']}" if v["mean"] is not None else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3, help="sessions per checkpoint x persona")
    parser.add_argument("--max-patient-turns", type=int, default=3)
    parser.add_argument("--iters", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--methods", nargs="*", default=["pto", "grpo"])
    parser.add_argument("--models", nargs="*", default=None,
                        help="explicit served model names; overrides --methods/--iters")
    parser.add_argument("--personas", nargs="*", default=["emma-smoking"],
                        help=f"named personas to sweep, or 'all'; choices: {list(PERSONAS)}")
    parser.add_argument("--include-base", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    personas = list(PERSONAS) if args.personas == ["all"] else args.personas
    unknown = [p for p in personas if p not in PERSONAS]
    if unknown:
        parser.error(f"unknown personas {unknown}; choices: {list(PERSONAS)}")

    if args.models:
        models = args.models
    else:
        models = [f"mi-coach-{m}-iter{i}" for m in args.methods for i in args.iters]
    base = None
    if args.include_base:
        import httpx
        served = httpx.get("http://localhost:8000/v1/models", timeout=10).json()["data"]
        base = next(m["id"] for m in served if not m["id"].startswith("mi-coach-"))
        if base not in models:
            models.append(base)

    jobs = [(model, persona, s) for model in models for persona in personas
            for s in range(args.sessions)]
    print(f"{len(models)} checkpoints x {len(personas)} personas x {args.sessions} sessions "
          f"= {len(jobs)} sessions (~{len(jobs) * (2 * args.max_patient_turns + 3)} gpt-4o-mini calls)")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, model, persona, s, args.max_patient_turns): (model, persona, s)
                   for model, persona, s in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            model, persona, s = futures[future]
            try:
                row = future.result()
                results.append(row)
                print(f"[{i}/{len(jobs)}] {model} · {persona} #{s}: Q1={row['q1_session_mean']} "
                      f"Q2={row['q2_mean']} MITI={row['miti_global_mean']} "
                      f"(${row['cost_usd']:.4f}, {row['wall_s']}s)")
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {model} · {persona} #{s}: FAILED — {e}")

    out_dir = REPO / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    total_cost = round(sum(r.get("cost_usd", 0.0) for r in results), 4)

    # Aggregates: per model (across personas) and per model x persona.
    agg = {model: aggregate([r for r in results if r["model"] == model]) for model in models}
    agg_mp = {(model, persona): aggregate([r for r in results
                                           if r["model"] == model and r["persona"] == persona])
              for model in models for persona in personas}
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps({
        "raw": results,
        "aggregate": agg,
        "aggregate_by_persona": {f"{m}|{p}": a for (m, p), a in agg_mp.items()},
        "total_cost_usd": total_cost,
    }, indent=2))

    # Markdown: per-model table (aggregate over personas) + per-persona table when sweeping.
    lines = ["| Checkpoint | Q1 (per-turn) | Q2 (17 items) | MITI globals | n | cost |",
             "|---|---|---|---|---|---|"]
    for model in models:
        a = agg[model]
        lines.append(f"| {model} | {fmt_cell(a, 'q1_session_mean')} | {fmt_cell(a, 'q2_mean')} "
                     f"| {fmt_cell(a, 'miti_global_mean')} | {a['n']} | ${a['cost_usd']:.3f} |")
    if len(personas) > 1:
        lines += ["", "### By persona", "",
                  "| Checkpoint | Persona | Q1 (per-turn) | Q2 (17 items) | MITI globals | n |",
                  "|---|---|---|---|---|---|"]
        for model in models:
            for persona in personas:
                a = agg_mp[(model, persona)]
                lines.append(f"| {model} | {persona} | {fmt_cell(a, 'q1_session_mean')} "
                             f"| {fmt_cell(a, 'q2_mean')} | {fmt_cell(a, 'miti_global_mean')} | {a['n']} |")
    lines += ["", f"*Total judge/patient-sim cost: ${total_cost} "
                  f"({len(results)} sessions, gpt-4o-mini).*"]
    table = "\n".join(lines)
    (out_dir / f"eval-{stamp}.md").write_text(table + "\n")
    print("\n" + table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("q1_session_mean", "Q1 per-turn mean"), ("q2_mean", "Q2 mean"),
               ("miti_global_mean", "MITI globals mean")]
    colors = {"pto": "#d62728", "grpo": "#1f77b4"}

    # Plot 1: score vs iteration per method (only when sweeping iterations).
    if not args.models:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, (metric, title) in zip(axes, metrics):
            for method in args.methods:
                xs, ys, es = [], [], []
                for i in args.iters:
                    a = agg.get(f"mi-coach-{method}-iter{i}", {}).get(metric)
                    if a and a["mean"] is not None:
                        xs.append(i); ys.append(a["mean"]); es.append(a["std"] or 0)
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method.upper(),
                            color=colors.get(method))
            if base and agg.get(base, {}).get(metric, {}).get("mean") is not None:
                ax.axhline(agg[base][metric]["mean"], ls="--", c="gray", label="base model")
            ax.set_title(title); ax.set_xlabel("training iteration"); ax.grid(alpha=0.3)
            ax.set_xticks(args.iters)
        axes[0].set_ylabel("judge score (1-5)"); axes[0].legend()
        fig.suptitle("MI Coach: judge scores vs training iteration "
                     f"(auto-demo, n={args.sessions}/checkpoint, gpt-4o-mini judge)")
        fig.tight_layout()
        fig.savefig(REPO / "docs" / "eval_scores.png", dpi=130)
        print(f"Saved: {REPO / 'docs' / 'eval_scores.png'}")

    # Plot 2: grouped bars per persona (only when sweeping personas).
    if len(personas) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        width = 0.8 / len(models)
        for ax, (metric, title) in zip(axes, metrics):
            for j, model in enumerate(models):
                xs = [i + j * width for i in range(len(personas))]
                ys = [agg_mp[(model, p)][metric]["mean"] or 0 for p in personas]
                es = [agg_mp[(model, p)][metric]["std"] or 0 for p in personas]
                label = model.removeprefix("mi-coach-")
                ax.bar(xs, ys, width=width, yerr=es, capsize=2, label=label)
            ax.set_xticks([i + 0.4 - width / 2 for i in range(len(personas))])
            ax.set_xticklabels(personas, rotation=15, ha="right", fontsize=8)
            ax.set_title(title); ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(1, 5.2)
        axes[0].set_ylabel("judge score (1-5)"); axes[0].legend(fontsize=8)
        fig.suptitle(f"MI Coach: judge scores by patient persona (n={args.sessions} each, "
                     "gpt-4o-mini judge)")
        fig.tight_layout()
        fig.savefig(REPO / "docs" / "eval_personas.png", dpi=130)
        print(f"Saved: {REPO / 'docs' / 'eval_personas.png'}")

    print(f"\nSaved: {out_dir}/eval-{stamp}.{{json,md}} — total cost ${total_cost}")


if __name__ == "__main__":
    main()
