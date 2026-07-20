"""MI Coach app: FastAPI session API + Gradio practice UI in one service.

The user plays the *patient* (the served thesis model is the therapist) or the
*therapist* (a simulated patient responds and the judges score the human).
Every therapist turn can be scored live by an LLM judge (thesis questionnaires,
gpt-4o-mini) via the LangGraph agent in agent/graph.py; sessions end with a
full feedback report (Q2 + MITI + supervisor narrative). Auto-demo mode runs a
simulated patient; the Compare tab drives two checkpoints side by side.

Sessions are held in memory (practice tool, not a clinical record store).

Run:  .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
Env:  VLLM_URL        vLLM OpenAI endpoint   (default http://localhost:8000/v1)
      DEFAULT_MODEL   therapist model name   (default mi-coach-pto-iter10)
      OPENAI_API_KEY  enables judge/report/demo (loaded from repo .env too)
"""

import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import zip_longest
from queue import SimpleQueue
from threading import Thread

import gradio as gr
import httpx
import matplotlib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

matplotlib.use("Agg")  # server-side rendering; handlers run off the main thread
from matplotlib import ticker
from matplotlib.figure import Figure

from agent.config import DEFAULT_PARAMS, JUDGE_MODEL_CHOICES, VLLM_URL, empty_usage
from agent.graph import (
    compare_sessions,
    judge_turn,
    patient_node,
    run_demo,
    run_patient_turn,
    run_report,
    run_turn,
    stream_patient,
    stream_therapist,
    therapist_node,
)
from agent.judging import (
    CUSTOM_QUESTIONNAIRES,
    DEFAULT_REPORT_QUESTIONNAIRES,
    DEFAULT_TURN_QUESTIONNAIRES,
    QUESTIONNAIRES,
    add_custom_questionnaire,
    delete_custom_questionnaire,
    known_questionnaires,
    questionnaire_blurbs,
)
from agent.thesis import GREETING, PERSONA_OPTIONS, build_patient_persona, initial_messages

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mi-coach-pto-iter10")
SCORING_ENABLED = bool(os.environ.get("OPENAI_API_KEY"))

DISCLAIMER = (
    "MI Coach is a practice tool for Motivational Interviewing skills. "
    "It is not therapy and must not be used as a substitute for professional care."
)

SESSIONS: dict[str, dict] = {}


def _validate_questionnaires(names: list[str]) -> list[str]:
    known = known_questionnaires()
    unknown = [n for n in names if n not in known]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown questionnaires {unknown}; valid: {known}")
    return names


class AdvancedParams(BaseModel):
    """Generation knobs; None = use the app default (agent.graph.DEFAULT_PARAMS)."""
    therapist_temperature: float | None = Field(default=None, ge=0, le=2)
    therapist_max_tokens: int | None = Field(default=None, ge=16, le=1024)
    patient_temperature: float | None = Field(default=None, ge=0, le=2)
    judge_model: str | None = Field(default=None, description=f"one of {JUDGE_MODEL_CHOICES}")
    seed: int | None = Field(default=None, description="seeds vLLM + OpenAI calls (best-effort)")

    def as_dict(self) -> dict:
        if self.judge_model is not None and self.judge_model not in JUDGE_MODEL_CHOICES:
            raise HTTPException(status_code=422, detail=f"judge_model must be one of {JUDGE_MODEL_CHOICES}")
        return {k: v for k, v in self.model_dump().items() if v is not None}


class CreateSession(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, description="Served therapist model name (ignored when you play the therapist)")
    role: str = Field(default="patient", description="Which role YOU play: 'patient' (model is the therapist) or 'therapist' (simulated patient responds; the judges score YOU)")
    turn_questionnaires: list[str] = Field(
        default=DEFAULT_TURN_QUESTIONNAIRES,
        description=f"Judged every turn; any of {list(QUESTIONNAIRES)}; [] disables per-turn judging")
    report_questionnaires: list[str] = Field(
        default=DEFAULT_REPORT_QUESTIONNAIRES, description="Judged at session end")
    params: AdvancedParams = AdvancedParams()
    turn_rationale: bool = Field(default=False, description="judge adds a one-sentence rationale per turn")
    report_rationale: bool = Field(default=False, description="...and per report instrument")
    # Simulated-patient persona (used when role='therapist').
    gender: str = "Female"
    age: int = 61
    problem: str = "Smoking"
    problem_time: str = "ManyYears"
    tried_to_solve: str = "ManyTimes"
    cooperation: str = "StartLowAndChangesToHigh"


class PatientMessage(BaseModel):
    content: str


class CompareReview(BaseModel):
    session_a: str = Field(description="Session id of side A")
    session_b: str = Field(description="Session id of side B")


class CustomQuestionnaire(BaseModel):
    name: str = Field(description="Instrument name (must not clash with a built-in)")
    description: str = Field(default="", description="One-line blurb shown in pickers")
    items: list[str] = Field(description="Statements about the therapist, each judged 1-5")


class DemoRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    max_patient_turns: int = Field(default=4, ge=1, le=20)
    turn_questionnaires: list[str] = Field(default=DEFAULT_TURN_QUESTIONNAIRES)
    report_questionnaires: list[str] = Field(default=DEFAULT_REPORT_QUESTIONNAIRES)
    params: AdvancedParams = AdvancedParams()
    turn_rationale: bool = False
    report_rationale: bool = False
    # Simulated-patient persona (thesis permutation dimensions).
    gender: str = Field(default="Female", description=f"one of {PERSONA_OPTIONS['gender']}")
    age: int = Field(default=61)
    problem: str = Field(default="Smoking", description=f"one of {PERSONA_OPTIONS['problem']}")
    problem_time: str = Field(default="ManyYears", description=f"one of {PERSONA_OPTIONS['problem_time']}")
    tried_to_solve: str = Field(default="ManyTimes", description=f"one of {PERSONA_OPTIONS['tried_to_solve']}")
    cooperation: str = Field(default="StartLowAndChangesToHigh", description=f"one of {PERSONA_OPTIONS['cooperation']}")


def _new_session(model: str, turn_qs: list[str] | None = None, report_qs: list[str] | None = None,
                 role: str = "patient", patient_persona: str | None = None,
                 params: dict | None = None, turn_rationale: bool = False,
                 report_rationale: bool = False, kind: str = "practice") -> dict:
    session = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,  # practice | compare | demo
        "model": model,
        "role": role,  # which role the HUMAN plays
        # Patient mode: model opens with the standard greeting. Therapist mode:
        # the human therapist writes their own opener; transcript starts empty.
        "messages": initial_messages() if role == "patient" else [],
        "patient_persona": patient_persona or build_patient_persona(),
        "turn_scores": [],
        "turn_questionnaires": DEFAULT_TURN_QUESTIONNAIRES if turn_qs is None else turn_qs,
        "report_questionnaires": DEFAULT_REPORT_QUESTIONNAIRES if report_qs is None else report_qs,
        "params": params or {},
        "turn_rationale": turn_rationale,
        "report_rationale": report_rationale,
        "usage": empty_usage(),
    }
    SESSIONS[session["id"]] = session
    return session


def _store_demo_session(model: str, state: dict, persona: str, kind: str = "demo") -> dict:
    """Record a finished auto-demo run so it shows up in history/export."""
    session = _new_session(model, state.get("turn_questionnaires"), state.get("report_questionnaires"),
                           role="patient", patient_persona=persona,
                           params=state.get("params") or {},
                           turn_rationale=bool(state.get("turn_rationale")),
                           report_rationale=bool(state.get("report_rationale")), kind=kind)
    session["messages"] = state["messages"]
    session["turn_scores"] = state["turn_scores"]
    session["report"] = state["report"]
    session["usage"] = state.get("openai_usage") or empty_usage()
    return session


def _advance(session: dict, user_message: str) -> dict:
    """Append the human's turn, run the counterpart (+judge when enabled)."""
    params = session.get("params") or {}
    turn_qs = session.get("turn_questionnaires") or []
    usage = session.setdefault("usage", empty_usage())
    judge = SCORING_ENABLED and bool(turn_qs)
    if session["role"] == "therapist":
        # Human is the therapist; the simulated patient replies, judges score the human.
        if not SCORING_ENABLED:
            raise HTTPException(status_code=503, detail="therapist mode needs OPENAI_API_KEY (simulated patient)")
        session["messages"].append({"role": "assistant", "content": user_message})
        if judge:
            state = run_patient_turn(session["messages"], session["patient_persona"],
                                     session["turn_scores"], turn_qs, params,
                                     session.get("turn_rationale", False), usage)
        else:
            state = patient_node({"messages": session["messages"], "params": params,
                                  "patient_system_prompt": session["patient_persona"],
                                  "openai_usage": usage})
    else:
        # Human is the patient; the local model therapist replies.
        session["messages"].append({"role": "user", "content": user_message})
        if judge:
            state = run_turn(session["messages"], session["model"], session["turn_scores"],
                             turn_qs, params, session.get("turn_rationale", False), usage)
        else:
            state = therapist_node({"messages": session["messages"], "model": session["model"],
                                    "params": params})
    session["messages"] = state["messages"]
    session["turn_scores"] = state.get("turn_scores", session["turn_scores"])
    session["usage"] = state.get("openai_usage", usage)
    return session["turn_scores"][-1] if judge and session["turn_scores"] else {}


def _ensure_report(session: dict) -> dict:
    """Generate (once) and cache the end-of-session report."""
    if "report" not in session:
        session["report"] = run_report(session["messages"], session["turn_scores"],
                                       session["report_questionnaires"], session.get("params"),
                                       session.get("report_rationale", False),
                                       dict(session.get("usage") or empty_usage()))
        session["usage"] = session["report"]["usage"]
    return session["report"]


def _run_comparison(sess_a: dict, sess_b: dict) -> dict:
    """Comparative final review of two sessions: make sure both reports exist,
    then one judge call over both transcripts+reports. Cached on both sessions;
    the comparison call's cost is counted on side A."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa, fb = pool.submit(_ensure_report, sess_a), pool.submit(_ensure_report, sess_b)
        fa.result(), fb.result()
    cmp = compare_sessions(sess_a["model"], sess_a["messages"], sess_a["report"],
                           sess_b["model"], sess_b["messages"], sess_b["report"],
                           sess_a.get("params"), sess_a["usage"])
    comparison = {"model_a": sess_a["model"], "model_b": sess_b["model"], **cmp}
    sess_a["comparison"] = sess_b["comparison"] = comparison
    return comparison


async def _list_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{VLLM_URL}/models")
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


def _session_summary(session: dict) -> dict:
    return {
        "id": session["id"],
        "created_at": session["created_at"],
        "kind": session["kind"],
        "role": session["role"],
        "model": session["model"],
        "therapist_turns": sum(1 for m in session["messages"] if m["role"] == "assistant"),
        "scored_turns": len(session["turn_scores"]),
        "has_report": "report" in session,
        "cost_usd": (session.get("usage") or {}).get("cost_usd", 0.0),
    }


# -------------------------------------------------------------------------- api

app = FastAPI(
    title="MI Coach",
    description=f"Practice API for Motivational Interviewing skills. {DISCLAIMER}",
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict:
    try:
        models = await _list_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"vLLM unreachable: {e}")
    return {"status": "ok", "vllm_models": models, "live_scoring": SCORING_ENABLED}


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSession) -> dict:
    if body.role not in ("patient", "therapist"):
        raise HTTPException(status_code=422, detail="role must be 'patient' or 'therapist'")
    try:
        persona = build_patient_persona(body.gender, body.age, body.problem,
                                        body.problem_time, body.tried_to_solve, body.cooperation)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"invalid persona option: {e}; valid: {PERSONA_OPTIONS}")
    session = _new_session(
        body.model,
        _validate_questionnaires(body.turn_questionnaires),
        _validate_questionnaires(body.report_questionnaires),
        role=body.role,
        patient_persona=persona,
        params=body.params.as_dict(),
        turn_rationale=body.turn_rationale,
        report_rationale=body.report_rationale,
    )
    return {"session_id": session["id"], "model": session["model"], "role": session["role"],
            "greeting": GREETING if body.role == "patient" else None,
            "turn_questionnaires": session["turn_questionnaires"],
            "report_questionnaires": session["report_questionnaires"],
            "params": {**DEFAULT_PARAMS, **session["params"]}}


@app.get("/sessions")
def list_sessions() -> list[dict]:
    return [_session_summary(s) for s in SESSIONS.values()]


@app.post("/sessions/{session_id}/message")
def send_message(session_id: str, body: PatientMessage) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    turn_score = _advance(session, body.content)
    return {
        "reply": session["messages"][-1]["content"],
        "reply_role": "patient" if session["role"] == "therapist" else "therapist",
        "turns": len(session["messages"]),
        "turn_score": turn_score,
        "usage": session["usage"],
    }


@app.post("/sessions/{session_id}/report")
def session_report(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if not SCORING_ENABLED:
        raise HTTPException(status_code=503, detail="no OPENAI_API_KEY — judge disabled")
    return _ensure_report(session)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {k: session[k] for k in ("id", "created_at", "kind", "role", "model", "messages",
                                    "turn_scores", "usage", "report") if k in session}


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    if SESSIONS.pop(session_id, None) is None:
        raise HTTPException(status_code=404, detail="unknown session")


@app.post("/demo")
def demo(body: DemoRequest) -> dict:
    if not SCORING_ENABLED:
        raise HTTPException(status_code=503, detail="no OPENAI_API_KEY — demo needs the judge")
    try:
        persona = build_patient_persona(body.gender, body.age, body.problem,
                                        body.problem_time, body.tried_to_solve, body.cooperation)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"invalid persona option: {e}; valid: {PERSONA_OPTIONS}")
    state = run_demo(
        body.model,
        max_patient_turns=body.max_patient_turns,
        patient_system_prompt=persona,
        turn_questionnaires=_validate_questionnaires(body.turn_questionnaires),
        report_questionnaires=_validate_questionnaires(body.report_questionnaires),
        params=body.params.as_dict(),
        turn_rationale=body.turn_rationale,
        report_rationale=body.report_rationale,
    )
    session = _store_demo_session(body.model, state, persona)
    return {
        "session_id": session["id"],
        "model": body.model,
        "messages": state["messages"][1:],  # drop system prompt
        "turn_scores": state["turn_scores"],
        "report": state["report"],
        "patient_persona": persona,
        "usage": session["usage"],
    }


@app.post("/compare/review")
def compare_review(body: CompareReview) -> dict:
    """Comparative final review between two sessions (reports generated if missing)."""
    if not SCORING_ENABLED:
        raise HTTPException(status_code=503, detail="no OPENAI_API_KEY — judge disabled")
    sess_a, sess_b = SESSIONS.get(body.session_a), SESSIONS.get(body.session_b)
    if sess_a is None or sess_b is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if not sess_a["messages"] or not sess_b["messages"]:
        raise HTTPException(status_code=422, detail="both sessions need at least one turn")
    return _run_comparison(sess_a, sess_b)


@app.get("/questionnaires")
def list_questionnaires() -> dict:
    """Selectable judge instruments: thesis built-ins + user-defined custom ones."""
    return {"builtin": {name: blurb for name, (_, blurb) in QUESTIONNAIRES.items()},
            "custom": CUSTOM_QUESTIONNAIRES}


@app.post("/questionnaires", status_code=201)
def create_questionnaire(body: CustomQuestionnaire) -> dict:
    try:
        add_custom_questionnaire(body.name, body.items, body.description)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    name = body.name.strip()
    return {"name": name, **CUSTOM_QUESTIONNAIRES[name]}


@app.delete("/questionnaires/{name}", status_code=204)
def remove_questionnaire(name: str) -> None:
    try:
        delete_custom_questionnaire(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown custom questionnaire {name!r}")


def _usage_line(usage: dict | None) -> str:
    if not usage or not usage.get("calls"):
        return ""
    return (f"*OpenAI usage: {usage['calls']} calls · {usage['prompt_tokens']:,} in / "
            f"{usage['completion_tokens']:,} out tokens · ~${usage['cost_usd']:.4f}*")


def _comparison_markdown(cmp: dict, model_a: str, model_b: str) -> str:
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


def _session_markdown(session: dict) -> str:
    """Export a session (transcript + scores + report) as markdown."""
    lines = [f"# MI Coach session `{session['id']}`",
             f"*{session.get('created_at', '')} — {session.get('kind', 'practice')} — "
             f"therapist model: `{session['model']}`* — {DISCLAIMER}", "", "## Transcript"]
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
        lines += ["", _comparison_markdown(cmp, cmp["model_a"], cmp["model_b"])]
    usage = _usage_line(session.get("usage"))
    if usage:
        lines += ["", usage]
    return "\n".join(lines) + "\n"


@app.get("/sessions/{session_id}/export", response_class=PlainTextResponse)
def export_session(session_id: str) -> str:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return _session_markdown(session)


# --------------------------------------------------------------------------- ui


BEST_ITER = {"pto": 10, "grpo": 8}  # thesis-best iteration per method

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


def _scores_plot(turn_scores: list[dict]) -> Figure | None:
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


def _scores_markdown(turn_scores: list[dict], report: dict | None = None,
                     usage: dict | None = None) -> str:
    if not SCORING_ENABLED:
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
    usage_line = _usage_line(usage)
    if usage_line:
        lines += ["", usage_line]
    return "\n".join(lines)


def _stream_reply(session, user_msg, history):
    """Append the human turn + stream the counterpart's reply.
    Yields updated history; the session transcript is finalized at the end."""
    role = session["role"]
    session["messages"].append(
        {"role": "assistant" if role == "therapist" else "user", "content": user_msg})
    history = history + [{"role": "user", "content": user_msg},
                         {"role": "assistant", "content": ""}]
    yield history
    if role == "therapist":
        gen = stream_patient(session["messages"], session["patient_persona"],
                             session["params"], session["usage"])
    else:
        gen = stream_therapist(session["messages"], session["model"], session["params"])
    reply = ""
    for reply in gen:
        history[-1]["content"] = reply
        yield history
    session["messages"].append(
        {"role": "user" if role == "therapist" else "assistant", "content": reply})


def _judge_last_turn(session):
    if not SCORING_ENABLED or not session["turn_questionnaires"]:
        return
    out = judge_turn(session["messages"], session["turn_scores"],
                     session["turn_questionnaires"], session["params"],
                     session["turn_rationale"], session["usage"])
    session["turn_scores"] = out["turn_scores"]
    session["usage"] = out["openai_usage"]


def _end_session(session):
    """Run (or fetch) the end-of-session report."""
    if session is None or not SCORING_ENABLED or not session["messages"]:
        return None
    return _ensure_report(session)


def _demo_stream(session, demo_turns):
    """Full simulated session on `session`, streamed. Yields (chat_history, event)
    with event "chunk" | "scored" | "reporting" | "done". Mirrors demo_graph
    (patient -> therapist -> judge -> loop -> report) but token-by-token,
    following the stream-then-judge pattern."""
    messages = session["messages"]
    history = [{"role": "assistant", "content": GREETING}]
    yield list(history), "chunk"
    for _ in range(int(demo_turns)):
        history.append({"role": "user", "content": ""})
        reply = ""
        for reply in stream_patient(messages, session["patient_persona"],
                                    session["params"], session["usage"]):
            history[-1] = {"role": "user", "content": reply}  # fresh dict: old yields stay immutable
            yield list(history), "chunk"
        messages.append({"role": "user", "content": reply})
        if "SESSION ENDED" in reply:
            break
        history.append({"role": "assistant", "content": ""})
        reply = ""
        for reply in stream_therapist(messages, session["model"], session["params"]):
            history[-1] = {"role": "assistant", "content": reply}
            yield list(history), "chunk"
        messages.append({"role": "assistant", "content": reply})
        if session["turn_questionnaires"]:
            _judge_last_turn(session)
            yield list(history), "scored"
        if "SESSION ENDED" in reply:
            break
    yield list(history), "reporting"
    _end_session(session)
    yield list(history), "done"


def _merge_streams(gen_a, gen_b):
    """Drive two generators in worker threads; yield ("a"|"b", item) as items
    arrive, so both sides stream concurrently (incl. their blocking judge calls)."""
    q: SimpleQueue = SimpleQueue()

    def pump(tag, gen):
        try:
            for item in gen:
                q.put((tag, item, None))
            q.put((tag, None, StopIteration))
        except Exception as e:  # re-raised in the UI thread below
            q.put((tag, None, e))

    for tag, gen in (("a", gen_a), ("b", gen_b)):
        Thread(target=pump, args=(tag, gen), daemon=True).start()
    done, error = 0, None
    while done < 2:
        tag, item, exc = q.get()
        if exc is StopIteration:
            done += 1
        elif exc is not None:
            error = error or exc
            done += 1
        else:
            yield tag, item
    if error is not None:
        raise error


def _export_file(name: str, content: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="mi-coach-"), name)
    with open(path, "w") as f:
        f.write(content)
    return path


def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="MI Coach") as ui:
        gr.Markdown(f"# MI Coach — practice session\n*{DISCLAIMER}*\n\n"
                    "Play the **patient** (the fine-tuned model is the therapist) or play the "
                    "**therapist** (a simulated patient responds — and the judges score *you*). "
                    "Every therapist turn is scored live by an LLM judge using the thesis questionnaires.")
        try:
            resp = httpx.get(f"{VLLM_URL}/models", timeout=5)
            resp.raise_for_status()
            served = [m["id"] for m in resp.json()["data"]]
        except Exception:
            served = [DEFAULT_MODEL, "mi-coach-grpo-iter8"]

        # Parse served adapters into method -> sorted iterations; base model separate.
        adapters: dict[str, list[int]] = {}
        base_model = next((m for m in served if not m.startswith("mi-coach-")), "base")
        for m in served:
            if m.startswith("mi-coach-") and "-iter" in m:
                method, _, it = m.removeprefix("mi-coach-").rpartition("-iter")
                adapters.setdefault(method, []).append(int(it))
        methods = sorted(adapters) or ["pto"]

        def _iter_choices(method: str):
            best = BEST_ITER.get(method)
            return [(f"iteration {i} ★ best" if i == best else f"iteration {i}", i)
                    for i in sorted(adapters.get(method, [BEST_ITER.get(method, 1)]))]

        def _model_name(method: str, iteration: int) -> str:
            return base_model if method == "base" else f"mi-coach-{method}-iter{iteration}"

        def _q_choices():
            return [(f"{name} — {blurb}", name) for name, blurb in questionnaire_blurbs().items()]

        q_choices = _q_choices()
        initial_chat = [{"role": "assistant", "content": GREETING}]
        method_choices = [(m.upper(), m) for m in methods] + [("Base model (no adapter)", "base")]
        default_method = "pto" if "pto" in methods else methods[0]

        def _adapter_pickers(suffix: str = "", method: str | None = None):
            method = method if method in adapters else default_method
            method_dd = gr.Dropdown(method_choices, value=method,
                                    label=f"Therapist adapter{suffix}")
            iter_dd = gr.Dropdown(_iter_choices(method), value=BEST_ITER.get(method, 10),
                                  label="Iteration (★ = thesis best)")
            method_dd.change(
                lambda m: gr.update(choices=_iter_choices(m),
                                    value=BEST_ITER.get(m) or (adapters.get(m, [1])[-1]),
                                    visible=m != "base"),
                [method_dd], [iter_dd])
            return method_dd, iter_dd

        def _persona_pickers():
            with gr.Row():
                p_gender = gr.Dropdown(PERSONA_OPTIONS["gender"], value="Female", label="Gender")
                p_age = gr.Dropdown(PERSONA_OPTIONS["age"], value=61, label="Age")
                p_problem = gr.Dropdown(PERSONA_OPTIONS["problem"], value="Smoking", label="Problem")
            with gr.Row():
                p_time = gr.Dropdown(PERSONA_OPTIONS["problem_time"], value="ManyYears", label="Problem duration")
                p_tried = gr.Dropdown(PERSONA_OPTIONS["tried_to_solve"], value="ManyTimes", label="Tried before")
                p_coop = gr.Dropdown(PERSONA_OPTIONS["cooperation"], value="StartLowAndChangesToHigh", label="Cooperation")
            return [p_gender, p_age, p_problem, p_time, p_tried, p_coop]

        # -------------------------------------------------- advanced settings
        with gr.Accordion("Advanced settings (apply to all tabs)", open=False):
            with gr.Row():
                adv_t_temp = gr.Slider(0.0, 1.5, value=DEFAULT_PARAMS["therapist_temperature"],
                                       step=0.05, label="Therapist temperature")
                adv_t_max = gr.Slider(64, 512, value=DEFAULT_PARAMS["therapist_max_tokens"],
                                      step=16, label="Therapist max tokens")
                adv_p_temp = gr.Slider(0.0, 1.5, value=DEFAULT_PARAMS["patient_temperature"],
                                       step=0.05, label="Simulated-patient temperature")
            with gr.Row():
                adv_judge = gr.Dropdown(JUDGE_MODEL_CHOICES, value=DEFAULT_PARAMS["judge_model"],
                                        label="Judge model (report + per-turn)")
                adv_seed = gr.Textbox(value="", label="Seed (blank = unseeded)",
                                      placeholder="e.g. 42")
                adv_demo_turns = gr.Slider(1, 20, value=3, step=1,
                                           label="Auto-demo length (patient turns)")
            with gr.Row():
                adv_turn_rat = gr.Checkbox(False, label="Judge rationale per turn (one sentence per instrument)")
                adv_report_rat = gr.Checkbox(False, label="Judge rationale in the session report")

        # ------------------------------------------- custom questionnaires
        with gr.Accordion("Custom questionnaire (build your own judge instrument)", open=False):
            gr.Markdown("Define your own instrument: the judge rates each statement about the "
                        "**therapist** 1–5 over the running transcript, exactly like the thesis "
                        "questionnaires. Saved to `data/custom_questionnaires.json`; select it "
                        "in any judge list below (re-using a name overwrites it).")
            with gr.Row():
                cq_name = gr.Textbox(label="Name", placeholder="e.g. LISTEN-3", scale=1)
                cq_desc = gr.Textbox(label="One-line description (optional)", scale=2)
            cq_items = gr.Textbox(
                label="Statements — one per line, each rated 1-5",
                lines=4,
                placeholder="The therapist let the patient do most of the talking.\n"
                            "The therapist avoided giving unsolicited advice.")
            with gr.Row():
                cq_add = gr.Button("Add / update questionnaire", variant="primary", size="sm")
                cq_del_dd = gr.Dropdown(sorted(CUSTOM_QUESTIONNAIRES), label="Existing custom questionnaire")
                cq_del = gr.Button("Delete selected", size="sm")
            cq_status = gr.Markdown()

        ADV = [adv_t_temp, adv_t_max, adv_p_temp, adv_judge, adv_seed]

        def _ui_params(t_temp, t_max, p_temp, judge_model, seed) -> dict:
            params = {"therapist_temperature": float(t_temp), "therapist_max_tokens": int(t_max),
                      "patient_temperature": float(p_temp), "judge_model": judge_model}
            seed = str(seed).strip()
            if seed:
                try:
                    params["seed"] = int(seed)
                except ValueError:
                    pass
            return params

        def _get_or_create(session_id, model, role, turn_q, report_q, persona,
                           params, turn_rat, report_rat, kind):
            """Reuse the session unless the model/role changed; sync settings."""
            session = SESSIONS.get(session_id)
            if session is None or session.get("model") != model or session.get("role") != role:
                session = _new_session(model, list(turn_q), list(report_q), role=role,
                                       patient_persona=persona, params=params,
                                       turn_rationale=turn_rat, report_rationale=report_rat,
                                       kind=kind)
            else:
                session.update(turn_questionnaires=list(turn_q), report_questionnaires=list(report_q),
                               params=params, turn_rationale=turn_rat, report_rationale=report_rat)
            return session

        # ------------------------------------------------------------ practice
        with gr.Tab("Practice"):
            with gr.Row():
                with gr.Column(scale=3):
                    role_radio = gr.Radio(
                        [("I play the patient — the model is the therapist", "patient"),
                         ("I play the therapist — simulated patient responds, the judges score ME", "therapist")],
                        value="patient", label="Your role")
                    with gr.Row():
                        method_dd, iter_dd = _adapter_pickers()
                    chat = gr.Chatbot(value=list(initial_chat), label="Session", height=430)
                    msg = gr.Textbox(label="Your message (as the patient)",
                                     placeholder="Hi David, I'm here because...")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        end = gr.Button("End session → report")
                        demo_btn = gr.Button("Auto-demo (simulated patient)")
                        reset = gr.Button("New session")
                    with gr.Accordion("Simulated-patient persona (thesis permutations)", open=False):
                        personas = _persona_pickers()
                    export_btn = gr.Button("Export session (.md)", size="sm")
                    export_file = gr.File(label="Session export", visible=False)
                with gr.Column(scale=1):
                    turn_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_TURN_QUESTIONNAIRES,
                                               label="Live judge (every turn — 1 gpt-4o-mini call each; none = off)")
                    report_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_REPORT_QUESTIONNAIRES,
                                                 label="Report judge (at session end)")
                    score_plot = gr.Plot(label="Score timeline")
                    scores_md = gr.Markdown(_scores_markdown([]))
            state = gr.State(None)  # session id

            def on_send(user_msg, history, session_id, method, iteration, turn_q, report_q,
                        role, gender, age, problem, ptime, tried, coop,
                        t_temp, t_max, p_temp, judge_model, seed, turn_rat, report_rat):
                user_msg = user_msg.strip()
                if not user_msg:
                    yield "", history, session_id, gr.update(), gr.update()
                    return
                if role == "therapist" and not SCORING_ENABLED:
                    raise gr.Error("Therapist mode needs OPENAI_API_KEY (simulated patient).")
                params = _ui_params(t_temp, t_max, p_temp, judge_model, seed)
                persona = build_patient_persona(gender, age, problem, ptime, tried, coop)
                model = "human" if role == "therapist" else _model_name(method, iteration)
                session = _get_or_create(session_id, model, role, turn_q, report_q, persona,
                                         params, turn_rat, report_rat, kind="practice")
                if session["id"] != session_id:
                    history = list(initial_chat) if role == "patient" else []
                session_id = session["id"]
                for history in _stream_reply(session, user_msg, history):
                    yield "", history, session_id, gr.update(), gr.update()
                _judge_last_turn(session)
                yield ("", history, session_id, _scores_plot(session["turn_scores"]),
                       _scores_markdown(session["turn_scores"], usage=session["usage"]))

            def on_role(role):
                """Switch chat framing when the human changes role."""
                if role == "therapist":
                    return ([], None, gr.update(visible=False), gr.update(visible=False),
                            gr.update(label="Your message (as the THERAPIST — you open the session)",
                                      placeholder="Hello, welcome. What brings you here today?"))
                return (list(initial_chat), None, gr.update(visible=True), gr.update(visible=True),
                        gr.update(label="Your message (as the patient)",
                                  placeholder="Hi David, I'm here because..."))

            def on_end(session_id):
                session = SESSIONS.get(session_id)
                if session is None or not SCORING_ENABLED or not session["messages"]:
                    yield gr.update()
                    return
                if "report" not in session:
                    yield (_scores_markdown(session["turn_scores"], usage=session["usage"])
                           + "\n\n⏳ *Generating session report…*")
                _end_session(session)
                yield _scores_markdown(session["turn_scores"], session["report"], session["usage"])

            def on_demo(method, iteration, turn_q, report_q, gender, age, problem, ptime, tried, coop,
                        t_temp, t_max, p_temp, judge_model, seed, turn_rat, report_rat, demo_turns):
                if not SCORING_ENABLED:
                    raise gr.Error("Auto-demo needs OPENAI_API_KEY (simulated patient).")
                params = _ui_params(t_temp, t_max, p_temp, judge_model, seed)
                persona = build_patient_persona(gender, age, problem, ptime, tried, coop)
                session = _new_session(_model_name(method, iteration), list(turn_q), list(report_q),
                                       role="patient", patient_persona=persona, params=params,
                                       turn_rationale=turn_rat, report_rationale=report_rat,
                                       kind="demo")
                for history, event in _demo_stream(session, demo_turns):
                    if event == "chunk":
                        yield history, session["id"], gr.update(), gr.update()
                    elif event == "scored":
                        yield (history, session["id"], _scores_plot(session["turn_scores"]),
                               _scores_markdown(session["turn_scores"], usage=session["usage"]))
                    elif event == "reporting":
                        yield (history, session["id"], gr.update(),
                               _scores_markdown(session["turn_scores"], usage=session["usage"])
                               + "\n\n⏳ *Generating session report…*")
                    else:
                        yield (history, session["id"], _scores_plot(session["turn_scores"]),
                               _scores_markdown(session["turn_scores"], session.get("report"),
                                                session["usage"]))

            def on_export(session_id):
                session = SESSIONS.get(session_id)
                if session is None:
                    return gr.update(visible=False)
                path = _export_file(f"session-{session['id']}.md", _session_markdown(session))
                return gr.update(value=path, visible=True)

            def on_reset():
                return list(initial_chat), None, None, _scores_markdown([]), gr.update(visible=False)

            send_inputs = [msg, chat, state, method_dd, iter_dd, turn_qs, report_qs,
                           role_radio, *personas, *ADV, adv_turn_rat, adv_report_rat]
            send.click(on_send, send_inputs, [msg, chat, state, score_plot, scores_md])
            msg.submit(on_send, send_inputs, [msg, chat, state, score_plot, scores_md])
            role_radio.change(on_role, [role_radio], [chat, state, method_dd, iter_dd, msg])
            end.click(on_end, [state], [scores_md])
            demo_btn.click(on_demo, [method_dd, iter_dd, turn_qs, report_qs, *personas,
                                     *ADV, adv_turn_rat, adv_report_rat, adv_demo_turns],
                           [chat, state, score_plot, scores_md])
            export_btn.click(on_export, [state], [export_file])
            reset.click(on_reset, None, [chat, state, score_plot, scores_md, export_file])

        # ------------------------------------------------------------- compare
        with gr.Tab("Compare (A/B)"):
            gr.Markdown("Drive **two checkpoints side by side**: send the same patient message to "
                        "both, or let the simulated patient run a full auto-demo against each. "
                        "End the session to get a full report per side, then run the "
                        "**comparative review** — one judge call that reads both transcripts and "
                        "reports and says how the models differ.")
            with gr.Row():
                with gr.Column():
                    method_a, iter_a = _adapter_pickers(" — A")
                    chat_a = gr.Chatbot(value=list(initial_chat), label="A", height=360)
                    plot_a = gr.Plot(label="A score timeline")
                    scores_a = gr.Markdown(_scores_markdown([]))
                with gr.Column():
                    method_b, iter_b = _adapter_pickers(" — B", method="grpo")
                    chat_b = gr.Chatbot(value=list(initial_chat), label="B", height=360)
                    plot_b = gr.Plot(label="B score timeline")
                    scores_b = gr.Markdown(_scores_markdown([]))
            with gr.Row():
                cmp_turn_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_TURN_QUESTIONNAIRES,
                                               label="Live judge (every turn, per side; none = off)")
                cmp_report_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_REPORT_QUESTIONNAIRES,
                                                 label="Report judge (at session end, per side)")
            with gr.Accordion("Simulated-patient persona (auto-demo)", open=False):
                cmp_personas = _persona_pickers()
            cmp_msg = gr.Textbox(label="Your message (as the patient — sent to both)")
            with gr.Row():
                cmp_send = gr.Button("Send to both", variant="primary")
                cmp_demo = gr.Button("Auto-demo both (simulated patient)")
                cmp_end = gr.Button("End sessions → reports")
                cmp_review = gr.Button("⚖️ Comparative review (A vs B)")
                cmp_export = gr.Button("Export comparison (.md)")
                cmp_reset = gr.Button("New comparison")
            cmp_verdict = gr.Markdown()
            cmp_file = gr.File(label="Comparison export", visible=False)
            sid_a, sid_b = gr.State(None), gr.State(None)

            def _cmp_outputs(sess_a, sess_b, with_reports=False):
                rep_a = sess_a.get("report") if with_reports else None
                rep_b = sess_b.get("report") if with_reports else None
                return (_scores_plot(sess_a["turn_scores"]), _scores_plot(sess_b["turn_scores"]),
                        _scores_markdown(sess_a["turn_scores"], rep_a, sess_a["usage"]),
                        _scores_markdown(sess_b["turn_scores"], rep_b, sess_b["usage"]))

            def on_cmp_send(user_msg, hist_a, hist_b, a_id, b_id, m_a, i_a, m_b, i_b,
                            turn_q, report_q, t_temp, t_max, p_temp, judge_model, seed,
                            turn_rat, report_rat):
                user_msg = user_msg.strip()
                if not user_msg:
                    yield ("", hist_a, hist_b, a_id, b_id,
                           gr.update(), gr.update(), gr.update(), gr.update())
                    return
                params = _ui_params(t_temp, t_max, p_temp, judge_model, seed)
                sess_a = _get_or_create(a_id, _model_name(m_a, i_a), "patient", turn_q, report_q,
                                        None, params, turn_rat, report_rat, kind="compare")
                sess_b = _get_or_create(b_id, _model_name(m_b, i_b), "patient", turn_q, report_q,
                                        None, params, turn_rat, report_rat, kind="compare")
                if sess_a["id"] != a_id:
                    hist_a = list(initial_chat)
                if sess_b["id"] != b_id:
                    hist_b = list(initial_chat)
                a_id, b_id = sess_a["id"], sess_b["id"]
                # Stream both replies concurrently (interleaved chunks).
                gen_a = _stream_reply(sess_a, user_msg, hist_a)
                gen_b = _stream_reply(sess_b, user_msg, hist_b)
                for step_a, step_b in zip_longest(gen_a, gen_b):
                    hist_a = step_a if step_a is not None else hist_a
                    hist_b = step_b if step_b is not None else hist_b
                    yield ("", hist_a, hist_b, a_id, b_id,
                           gr.update(), gr.update(), gr.update(), gr.update())
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fa = pool.submit(_judge_last_turn, sess_a)
                    fb = pool.submit(_judge_last_turn, sess_b)
                    fa.result(), fb.result()
                yield ("", hist_a, hist_b, a_id, b_id, *_cmp_outputs(sess_a, sess_b))

            def on_cmp_demo(m_a, i_a, m_b, i_b, turn_q, report_q,
                            gender, age, problem, ptime, tried, coop,
                            t_temp, t_max, p_temp, judge_model, seed,
                            turn_rat, report_rat, demo_turns):
                if not SCORING_ENABLED:
                    raise gr.Error("Auto-demo needs OPENAI_API_KEY (simulated patient).")
                params = _ui_params(t_temp, t_max, p_temp, judge_model, seed)
                persona = build_patient_persona(gender, age, problem, ptime, tried, coop)

                def new_side(method, iteration):
                    return _new_session(_model_name(method, iteration), list(turn_q), list(report_q),
                                        role="patient", patient_persona=persona, params=params,
                                        turn_rationale=turn_rat, report_rationale=report_rat,
                                        kind="compare")

                sides = {"a": new_side(m_a, i_a), "b": new_side(m_b, i_b)}

                def side_updates(session, item):
                    """(chat, plot, md) updates for one side's stream event."""
                    history, event = item
                    if event == "chunk":
                        return history, gr.update(), gr.update()
                    if event == "reporting":
                        return (history, gr.update(),
                                _scores_markdown(session["turn_scores"], usage=session["usage"])
                                + "\n\n⏳ *Generating session report…*")
                    return (history, _scores_plot(session["turn_scores"]),
                            _scores_markdown(session["turn_scores"],
                                             session.get("report") if event == "done" else None,
                                             session["usage"]))

                # Both sides stream concurrently; each event updates only its side.
                merged = _merge_streams(_demo_stream(sides["a"], demo_turns),
                                        _demo_stream(sides["b"], demo_turns))
                for tag, item in merged:
                    chat, plot, md = side_updates(sides[tag], item)
                    if tag == "a":
                        yield (chat, gr.update(), sides["a"]["id"], sides["b"]["id"],
                               plot, gr.update(), md, gr.update())
                    else:
                        yield (gr.update(), chat, sides["a"]["id"], sides["b"]["id"],
                               gr.update(), plot, gr.update(), md)

            def on_cmp_end(a_id, b_id):
                sess_a, sess_b = SESSIONS.get(a_id), SESSIONS.get(b_id)
                if sess_a is None or sess_b is None:
                    yield gr.update(), gr.update(), gr.update(), gr.update()
                    return
                if "report" not in sess_a or "report" not in sess_b:
                    wait = "\n\n⏳ *Generating session report…*"
                    yield (gr.update(), gr.update(),
                           _scores_markdown(sess_a["turn_scores"], usage=sess_a["usage"]) + wait,
                           _scores_markdown(sess_b["turn_scores"], usage=sess_b["usage"]) + wait)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fa = pool.submit(_end_session, sess_a)
                    fb = pool.submit(_end_session, sess_b)
                    fa.result(), fb.result()
                yield _cmp_outputs(sess_a, sess_b, with_reports=True)

            def on_cmp_review(a_id, b_id):
                sess_a, sess_b = SESSIONS.get(a_id), SESSIONS.get(b_id)
                if sess_a is None or sess_b is None:
                    raise gr.Error("Send a message to both sides (or run an auto-demo) first.")
                if not SCORING_ENABLED:
                    raise gr.Error("Comparative review needs OPENAI_API_KEY.")
                yield ("⏳ *Scoring both sessions and comparing the models…*",
                       gr.update(), gr.update())
                comparison = _run_comparison(sess_a, sess_b)
                # Reports may have just been generated — refresh both score panels too.
                yield (_comparison_markdown(comparison, sess_a["model"], sess_b["model"]),
                       _scores_markdown(sess_a["turn_scores"], sess_a["report"], sess_a["usage"]),
                       _scores_markdown(sess_b["turn_scores"], sess_b["report"], sess_b["usage"]))

            def on_cmp_export(a_id, b_id):
                sess_a, sess_b = SESSIONS.get(a_id), SESSIONS.get(b_id)
                if sess_a is None or sess_b is None:
                    return gr.update(visible=False)
                cmp = sess_a.get("comparison")
                # The verdict goes at the top; strip it from the per-side markdown.
                strip = lambda s: {k: v for k, v in s.items() if k != "comparison"}
                content = (f"# MI Coach A/B comparison — `{sess_a['model']}` vs `{sess_b['model']}`\n\n"
                           + (f"{_comparison_markdown(cmp, cmp['model_a'], cmp['model_b'])}\n\n" if cmp else "")
                           + f"# Side A\n\n{_session_markdown(strip(sess_a))}\n\n"
                           f"# Side B\n\n{_session_markdown(strip(sess_b))}")
                path = _export_file(f"compare-{sess_a['id']}-vs-{sess_b['id']}.md", content)
                return gr.update(value=path, visible=True)

            def on_cmp_reset():
                return (list(initial_chat), list(initial_chat), None, None,
                        None, None, _scores_markdown([]), _scores_markdown([]),
                        "", gr.update(visible=False))

            cmp_send_inputs = [cmp_msg, chat_a, chat_b, sid_a, sid_b,
                               method_a, iter_a, method_b, iter_b, cmp_turn_qs, cmp_report_qs,
                               *ADV, adv_turn_rat, adv_report_rat]
            cmp_send_outputs = [cmp_msg, chat_a, chat_b, sid_a, sid_b,
                                plot_a, plot_b, scores_a, scores_b]
            cmp_send.click(on_cmp_send, cmp_send_inputs, cmp_send_outputs)
            cmp_msg.submit(on_cmp_send, cmp_send_inputs, cmp_send_outputs)
            cmp_demo.click(on_cmp_demo,
                           [method_a, iter_a, method_b, iter_b, cmp_turn_qs, cmp_report_qs,
                            *cmp_personas, *ADV, adv_turn_rat, adv_report_rat, adv_demo_turns],
                           [chat_a, chat_b, sid_a, sid_b, plot_a, plot_b, scores_a, scores_b])
            cmp_end.click(on_cmp_end, [sid_a, sid_b], [plot_a, plot_b, scores_a, scores_b])
            cmp_review.click(on_cmp_review, [sid_a, sid_b], [cmp_verdict, scores_a, scores_b])
            cmp_export.click(on_cmp_export, [sid_a, sid_b], [cmp_file])
            cmp_reset.click(on_cmp_reset, None,
                            [chat_a, chat_b, sid_a, sid_b, plot_a, plot_b, scores_a, scores_b,
                             cmp_verdict, cmp_file])

        # ------------------------------------------------------------- history
        with gr.Tab("History"):
            gr.Markdown("Sessions from this server run (practice, compare, and auto-demo). "
                        "Pick one to review the transcript, scores, and report, or export it.")
            hist_refresh = gr.Button("Refresh", size="sm")
            hist_table = gr.Dataframe(interactive=False, label="Sessions")
            hist_dd = gr.Dropdown([], label="Open session")
            hist_chat = gr.Chatbot(label="Transcript", height=360)
            hist_scores = gr.Markdown()
            hist_export = gr.Button("Export session (.md)", size="sm")
            hist_file = gr.File(label="Session export", visible=False)

            def _history_rows() -> pd.DataFrame:
                rows = [_session_summary(s) for s in SESSIONS.values()]
                rows.sort(key=lambda r: r["created_at"], reverse=True)
                return pd.DataFrame(rows, columns=["id", "created_at", "kind", "role", "model",
                                                   "therapist_turns", "scored_turns", "has_report",
                                                   "cost_usd"])

            def on_hist_refresh():
                rows = _history_rows()
                choices = [(f"{r.created_at} · {r.kind} · {r.model} · {r.id}", r.id)
                           for r in rows.itertuples()]
                return rows, gr.update(choices=choices, value=None)

            def on_hist_pick(session_id):
                session = SESSIONS.get(session_id)
                if session is None:
                    return [], ""
                history = [{"role": m["role"], "content": m["content"]}
                           for m in session["messages"] if m["role"] in ("user", "assistant")]
                md = _scores_markdown(session["turn_scores"], session.get("report"),
                                      session.get("usage"))
                cmp = session.get("comparison")
                if cmp:
                    md += "\n\n" + _comparison_markdown(cmp, cmp["model_a"], cmp["model_b"])
                return history, md

            def on_hist_export(session_id):
                session = SESSIONS.get(session_id)
                if session is None:
                    return gr.update(visible=False)
                path = _export_file(f"session-{session['id']}.md", _session_markdown(session))
                return gr.update(value=path, visible=True)

            hist_refresh.click(on_hist_refresh, None, [hist_table, hist_dd])
            hist_dd.change(on_hist_pick, [hist_dd], [hist_chat, hist_scores])
            hist_export.click(on_hist_export, [hist_dd], [hist_file])

        # Custom-questionnaire wiring — registered last so it can refresh every
        # judge CheckboxGroup across the tabs (choices update, selections kept).
        qs_lists = [turn_qs, report_qs, cmp_turn_qs, cmp_report_qs]

        def _qs_updates(selections, dropped: str | None = None):
            return [gr.update(choices=_q_choices(),
                              value=[v for v in sel if v != dropped])
                    for sel in selections]

        def on_cq_add(name, desc, items_text, *selections):
            try:
                add_custom_questionnaire(name, str(items_text).splitlines(), desc)
            except ValueError as e:
                return (f"⚠️ {e}", gr.update(), *[gr.update()] * len(selections))
            name = name.strip()
            n = len(CUSTOM_QUESTIONNAIRES[name]["items"])
            return (f"✅ **{name}** saved ({n} item{'s' if n != 1 else ''}) — selectable in "
                    "every judge list, scored 1-5 like the thesis instruments.",
                    gr.update(choices=sorted(CUSTOM_QUESTIONNAIRES), value=None),
                    *_qs_updates(selections))

        def on_cq_del(name, *selections):
            if not name:
                return ("*Pick a custom questionnaire to delete.*",
                        gr.update(), *[gr.update()] * len(selections))
            delete_custom_questionnaire(name)
            return (f"🗑️ Deleted **{name}**.",
                    gr.update(choices=sorted(CUSTOM_QUESTIONNAIRES), value=None),
                    *_qs_updates(selections, dropped=name))

        cq_add.click(on_cq_add, [cq_name, cq_desc, cq_items, *qs_lists],
                     [cq_status, cq_del_dd, *qs_lists])
        cq_del.click(on_cq_del, [cq_del_dd, *qs_lists],
                     [cq_status, cq_del_dd, *qs_lists])
    return ui


app = gr.mount_gradio_app(app, _build_ui(), path="/ui")
