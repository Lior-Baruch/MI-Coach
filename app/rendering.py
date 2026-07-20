"""Rendering: markdown for scores/reports/exports + the score-timeline plot.

Pure presentation — everything here turns session data into markdown strings
or a matplotlib Figure; nothing mutates state or calls a model.
"""

import matplotlib

matplotlib.use("Agg")  # server-side rendering; handlers run off the main thread
from matplotlib import ticker
from matplotlib.figure import Figure

from agent.judging import CUSTOM_QUESTIONNAIRES, QUESTIONNAIRES, known_questionnaires
from app import sessions

# CVD-validated categorical palette (dataviz default, light mode). Each built-in
# instrument keeps a fixed slot so its color follows it across sessions and tabs;
# custom instruments take a slot by registry position and draw dashed.
_PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
            "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
_INSTRUMENT_COLORS = dict(zip(QUESTIONNAIRES, _PALETTE))


def _series_style(name: str) -> dict:
    color = _INSTRUMENT_COLORS.get(name)
    if color:
        return {"color": color, "linestyle": "-"}
    customs = list(CUSTOM_QUESTIONNAIRES)
    idx = customs.index(name) if name in customs else 0
    # Walk the palette from the far end so a custom instrument doesn't share a
    # hue with the common built-ins (Q1=blue, Q2=green) it's plotted next to.
    return {"color": _PALETTE[(len(_PALETTE) - 1 - idx) % len(_PALETTE)], "linestyle": "--"}


def scores_plot(turn_scores: list[dict]) -> Figure | None:
    """Score-timeline figure (mean per instrument per therapist turn).
    Server-rendered matplotlib via gr.Plot — gr.LinePlot rendered blank on
    Gradio 6, and markers keep even a single scored turn visible."""
    series: dict[str, tuple[list[int], list[float]]] = {}
    for t in turn_scores:
        for name, mean in t["means"].items():
            xs, ys = series.setdefault(name, ([], []))
            xs.append(t["therapist_turns"] - 1)
            ys.append(mean)
    if not series:
        return None
    fig = Figure(figsize=(5.4, 2.6), dpi=100)
    ax = fig.add_subplot()
    order = {n: i for i, n in enumerate(known_questionnaires())}
    for name in sorted(series, key=lambda n: order.get(n, len(order))):
        xs, ys = series[name]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.5,
                label=name, **_series_style(name))
    ax.set_ylim(0.7, 5.3)
    ax.set_yticks(range(1, 6))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("therapist turn", fontsize=8)
    ax.set_ylabel("mean score (1–5)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        ax.legend(fontsize=7, frameon=False, loc="best")
    else:
        ax.set_title(next(iter(series)), fontsize=9)
    fig.tight_layout()
    return fig


def usage_line(usage: dict | None) -> str:
    if not usage or not usage.get("calls"):
        return ""
    return (f"*OpenAI usage: {usage['calls']} calls · {usage['prompt_tokens']:,} in / "
            f"{usage['completion_tokens']:,} out tokens · ~${usage['cost_usd']:.4f}*")


def scores_markdown(turn_scores: list[dict], report: dict | None = None,
                    usage: dict | None = None) -> str:
    if not sessions.SCORING_ENABLED:
        return "*Live scoring off (no `OPENAI_API_KEY`).*"
    lines = ["### Live scores (mean, 1-5)"]
    if turn_scores:
        for t in turn_scores:
            means = " · ".join(f"{name} **{mean}**" for name, mean in t["means"].items())
            lines.append(f"- Turn {t['therapist_turns'] - 1}: {means}")
            for name, r in t.get("results", {}).items():
                if r.get("rationale"):
                    lines.append(f"  - {name}: *{r['rationale']}*")
    else:
        lines.append("*No scored turns yet.*")
    if report:
        lines.append("\n### Session report")
        for name, r in report["results"].items():
            lines.append(f"- **{name}** mean: **{r['mean']}**")
            if "globals" in r:
                lines.append(f"  - globals: {r['globals']}")
                lines.append(f"  - behavior counts: {r['behaviors']}")
            if r.get("rationale"):
                lines.append(f"  - *{r['rationale']}*")
        a = report["assessment"]
        stars = "★" * a["overall_rating"] + "☆" * (5 - a["overall_rating"])
        lines += [
            f"\n### Overall assessment: {stars} ({a['overall_rating']}/5)",
            a["summary"],
            "\n**Strengths:**", *[f"- {s}" for s in a["strengths"]],
            "\n**Growth areas:**", *[f"- {g}" for g in a["growth_areas"]],
            f"\n**Tip:** {a['tip']}",
        ]
    line = usage_line(usage)
    if line:
        lines += ["", line]
    return "\n".join(lines)


def comparison_markdown(cmp: dict, model_a: str, model_b: str) -> str:
    """Render a compare_sessions() verdict as markdown."""
    preferred = {"A": f"A (`{model_a}`)", "B": f"B (`{model_b}`)", "tie": "Tie"}[cmp["preferred"]]
    return "\n".join([
        f"### ⚖️ Comparative review — A: `{model_a}` vs B: `{model_b}`",
        f"**Preferred: {preferred}**", "",
        cmp["summary"], "",
        "**Key differences:**", *[f"- {d}" for d in cmp["key_differences"]], "",
        f"**Where A (`{model_a}`) is stronger:**", *[f"- {s}" for s in cmp["a_strengths"]], "",
        f"**Where B (`{model_b}`) is stronger:**", *[f"- {s}" for s in cmp["b_strengths"]], "",
        f"**Recommendation:** {cmp['recommendation']}",
    ])


def session_markdown(session: dict) -> str:
    """Export a session (transcript + scores + report) as markdown."""
    lines = [f"# MI Coach session `{session['id']}`",
             f"*{session.get('created_at', '')} — {session.get('kind', 'practice')} — "
             f"therapist model: `{session['model']}`* — {sessions.DISCLAIMER}", "", "## Transcript"]
    for m in session["messages"]:
        if m["role"] == "user":
            lines.append(f"\n**Patient:** {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"\n**Therapist:** {m['content']}")
    if session.get("turn_scores"):
        lines += ["", "## Per-turn judge scores (mean, 1-5)"]
        for t in session["turn_scores"]:
            means = ", ".join(f"{k}: {v}" for k, v in t["means"].items())
            lines.append(f"- Therapist turn {t['therapist_turns'] - 1}: {means}")
            for name, r in t.get("results", {}).items():
                if r.get("rationale"):
                    lines.append(f"  - {name}: *{r['rationale']}*")
    report = session.get("report")
    if report:
        lines += ["", "## Session report"]
        for name, r in report["results"].items():
            lines.append(f"### {name} — mean {r['mean']}")
            lines.append(f"- scores: {r['scores']}")
            if "globals" in r:
                lines.append(f"- globals: {r['globals']}")
                lines.append(f"- behavior counts: {r['behaviors']}")
            if r.get("rationale"):
                lines.append(f"- judge rationale: *{r['rationale']}*")
        a = report["assessment"]
        lines += ["", f"## Overall assessment — {a['overall_rating']}/5", a["summary"],
                  "", "**Strengths:**", *[f"- {s}" for s in a["strengths"]],
                  "", "**Growth areas:**", *[f"- {g}" for g in a["growth_areas"]],
                  "", f"**Tip:** {a['tip']}"]
    cmp = session.get("comparison")
    if cmp:
        lines += ["", comparison_markdown(cmp, cmp["model_a"], cmp["model_b"])]
    line = usage_line(session.get("usage"))
    if line:
        lines += ["", line]
    return "\n".join(lines) + "\n"
