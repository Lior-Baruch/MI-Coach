"""MI Coach app: FastAPI session API + Gradio practice UI in one service.

The user plays the *patient*; the served thesis model plays the therapist.
Phase 3: every therapist turn is scored live by an LLM judge (gpt-4o-mini +
thesis questionnaire Q1) via the LangGraph agent in agent/graph.py; sessions
can end with a full feedback report (Q2 + MITI + supervisor narrative), and an
auto-demo mode runs a simulated patient instead of the human.

Sessions are held in memory (practice tool, not a clinical record store).

Run:  .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
Env:  VLLM_URL        vLLM OpenAI endpoint   (default http://localhost:8000/v1)
      DEFAULT_MODEL   therapist model name   (default mi-coach-pto-iter10)
      OPENAI_API_KEY  enables judge/report/demo (loaded from repo .env too)
"""

import os
import tempfile
import uuid

import gradio as gr
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from agent.graph import (
    DEFAULT_REPORT_QUESTIONNAIRES,
    DEFAULT_TURN_QUESTIONNAIRES,
    GREETING,
    PERSONA_OPTIONS,
    QUESTIONNAIRES,
    build_patient_persona,
    initial_messages,
    run_demo,
    run_patient_turn,
    run_report,
    run_turn,
)

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mi-coach-pto-iter10")
SCORING_ENABLED = bool(os.environ.get("OPENAI_API_KEY"))

DISCLAIMER = (
    "MI Coach is a practice tool for Motivational Interviewing skills. "
    "It is not therapy and must not be used as a substitute for professional care."
)

SESSIONS: dict[str, dict] = {}


def _validate_questionnaires(names: list[str]) -> list[str]:
    unknown = [n for n in names if n not in QUESTIONNAIRES]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown questionnaires {unknown}; valid: {list(QUESTIONNAIRES)}")
    return names


class CreateSession(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, description="Served therapist model name (ignored when you play the therapist)")
    role: str = Field(default="patient", description="Which role YOU play: 'patient' (model is the therapist) or 'therapist' (simulated patient responds; the judges score YOU)")
    turn_questionnaires: list[str] = Field(
        default=DEFAULT_TURN_QUESTIONNAIRES, description=f"Judged every turn; any of {list(QUESTIONNAIRES)}")
    report_questionnaires: list[str] = Field(
        default=DEFAULT_REPORT_QUESTIONNAIRES, description="Judged at session end")
    # Simulated-patient persona (used when role='therapist').
    gender: str = "Female"
    age: int = 61
    problem: str = "Smoking"
    problem_time: str = "ManyYears"
    tried_to_solve: str = "ManyTimes"
    cooperation: str = "StartLowAndChangesToHigh"


class PatientMessage(BaseModel):
    content: str


class DemoRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    max_patient_turns: int = Field(default=4, ge=1, le=8)
    turn_questionnaires: list[str] = Field(default=DEFAULT_TURN_QUESTIONNAIRES)
    report_questionnaires: list[str] = Field(default=DEFAULT_REPORT_QUESTIONNAIRES)
    # Simulated-patient persona (thesis permutation dimensions).
    gender: str = Field(default="Female", description=f"one of {PERSONA_OPTIONS['gender']}")
    age: int = Field(default=61)
    problem: str = Field(default="Smoking", description=f"one of {PERSONA_OPTIONS['problem']}")
    problem_time: str = Field(default="ManyYears", description=f"one of {PERSONA_OPTIONS['problem_time']}")
    tried_to_solve: str = Field(default="ManyTimes", description=f"one of {PERSONA_OPTIONS['tried_to_solve']}")
    cooperation: str = Field(default="StartLowAndChangesToHigh", description=f"one of {PERSONA_OPTIONS['cooperation']}")


def _new_session(model: str, turn_qs: list[str] | None = None, report_qs: list[str] | None = None,
                 role: str = "patient", patient_persona: str | None = None) -> dict:
    session = {
        "id": uuid.uuid4().hex[:12],
        "model": model,
        "role": role,  # which role the HUMAN plays
        # Patient mode: model opens with the standard greeting. Therapist mode:
        # the human therapist writes their own opener; transcript starts empty.
        "messages": initial_messages() if role == "patient" else [],
        "patient_persona": patient_persona or build_patient_persona(),
        "turn_scores": [],
        "turn_questionnaires": turn_qs or DEFAULT_TURN_QUESTIONNAIRES,
        "report_questionnaires": report_qs or DEFAULT_REPORT_QUESTIONNAIRES,
    }
    SESSIONS[session["id"]] = session
    return session


def _advance(session: dict, user_message: str) -> dict:
    """Append the human's turn, run the counterpart (+judge when enabled)."""
    if session["role"] == "therapist":
        # Human is the therapist; the simulated patient replies, judges score the human.
        session["messages"].append({"role": "assistant", "content": user_message})
        if not SCORING_ENABLED:
            raise HTTPException(status_code=503, detail="therapist mode needs OPENAI_API_KEY (simulated patient)")
        state = run_patient_turn(session["messages"], session["patient_persona"],
                                 session["turn_scores"], session["turn_questionnaires"])
    else:
        # Human is the patient; the local model therapist replies.
        session["messages"].append({"role": "user", "content": user_message})
        if not SCORING_ENABLED:
            from agent.graph import therapist_node
            session["messages"] = therapist_node(
                {"messages": session["messages"], "model": session["model"]})["messages"]
            return {}
        state = run_turn(session["messages"], session["model"], session["turn_scores"],
                         session["turn_questionnaires"])
    session["messages"] = state["messages"]
    session["turn_scores"] = state["turn_scores"]
    return session["turn_scores"][-1] if session["turn_scores"] else {}


async def _list_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{VLLM_URL}/models")
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


# -------------------------------------------------------------------------- api

app = FastAPI(
    title="MI Coach",
    description=f"Practice API for Motivational Interviewing skills. {DISCLAIMER}",
    version="0.3.0",
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
    )
    return {"session_id": session["id"], "model": session["model"], "role": session["role"],
            "greeting": GREETING if body.role == "patient" else None,
            "turn_questionnaires": session["turn_questionnaires"],
            "report_questionnaires": session["report_questionnaires"]}


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
    }


@app.post("/sessions/{session_id}/report")
def session_report(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if not SCORING_ENABLED:
        raise HTTPException(status_code=503, detail="no OPENAI_API_KEY — judge disabled")
    if "report" not in session:
        session["report"] = run_report(session["messages"], session["turn_scores"],
                                       session["report_questionnaires"])
    return session["report"]


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {k: session[k] for k in ("id", "model", "messages", "turn_scores") if k in session}


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
    )
    return {
        "model": body.model,
        "messages": state["messages"][1:],  # drop system prompt
        "turn_scores": state["turn_scores"],
        "report": state["report"],
        "patient_persona": persona,
    }


def _session_markdown(session: dict) -> str:
    """Export a session (transcript + scores + report) as markdown."""
    lines = [f"# MI Coach session `{session['id']}`",
             f"*Therapist model: `{session['model']}`* — {DISCLAIMER}", "", "## Transcript"]
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
    report = session.get("report")
    if report:
        lines += ["", "## Session report"]
        for name, r in report["results"].items():
            lines.append(f"### {name} — mean {r['mean']}")
            lines.append(f"- scores: {r['scores']}")
            if "globals" in r:
                lines.append(f"- globals: {r['globals']}")
                lines.append(f"- behavior counts: {r['behaviors']}")
        a = report["assessment"]
        lines += ["", f"## Overall assessment — {a['overall_rating']}/5", a["summary"],
                  "", "**Strengths:**", *[f"- {s}" for s in a["strengths"]],
                  "", "**Growth areas:**", *[f"- {g}" for g in a["growth_areas"]],
                  "", f"**Tip:** {a['tip']}"]
    return "\n".join(lines) + "\n"


@app.get("/sessions/{session_id}/export", response_class=PlainTextResponse)
def export_session(session_id: str) -> str:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return _session_markdown(session)


# --------------------------------------------------------------------------- ui


BEST_ITER = {"pto": 10, "grpo": 8}  # thesis-best iteration per method
_EMPTY_SCORES = pd.DataFrame(columns=["turn", "mean", "instrument"])


def _scores_df(turn_scores: list[dict]) -> pd.DataFrame:
    rows = [
        {"turn": t["therapist_turns"] - 1, "mean": mean, "instrument": name}
        for t in turn_scores
        for name, mean in t["means"].items()
    ]
    return pd.DataFrame(rows) if rows else _EMPTY_SCORES


def _scores_markdown(turn_scores: list[dict], report: dict | None = None) -> str:
    if not SCORING_ENABLED:
        return "*Live scoring off (no `OPENAI_API_KEY`).*"
    lines = ["### Live scores (mean, 1-5)"]
    if turn_scores:
        for t in turn_scores:
            means = " · ".join(f"{name} **{mean}**" for name, mean in t["means"].items())
            lines.append(f"- Turn {t['therapist_turns'] - 1}: {means}")
    else:
        lines.append("*No scored turns yet.*")
    if report:
        lines.append("\n### Session report")
        for name, r in report["results"].items():
            lines.append(f"- **{name}** mean: **{r['mean']}**")
            if "globals" in r:
                lines.append(f"  - globals: {r['globals']}")
                lines.append(f"  - behavior counts: {r['behaviors']}")
        a = report["assessment"]
        stars = "★" * a["overall_rating"] + "☆" * (5 - a["overall_rating"])
        lines += [
            f"\n### Overall assessment: {stars} ({a['overall_rating']}/5)",
            a["summary"],
            "\n**Strengths:**", *[f"- {s}" for s in a["strengths"]],
            "\n**Growth areas:**", *[f"- {g}" for g in a["growth_areas"]],
            f"\n**Tip:** {a['tip']}",
        ]
    return "\n".join(lines)


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

        q_choices = [(f"{name} — {blurb}", name) for name, (_, blurb) in QUESTIONNAIRES.items()]
        initial_chat = [{"role": "assistant", "content": GREETING}]
        method_choices = [(m.upper(), m) for m in methods] + [("Base model (no adapter)", "base")]
        default_method = "pto" if "pto" in methods else methods[0]

        def _adapter_pickers(suffix: str = ""):
            method_dd = gr.Dropdown(method_choices, value=default_method,
                                    label=f"Therapist adapter{suffix}")
            iter_dd = gr.Dropdown(_iter_choices(default_method), value=BEST_ITER.get(default_method, 10),
                                  label="Iteration (★ = thesis best)")
            method_dd.change(
                lambda m: gr.update(choices=_iter_choices(m),
                                    value=BEST_ITER.get(m) or (adapters.get(m, [1])[-1]),
                                    visible=m != "base"),
                [method_dd], [iter_dd])
            return method_dd, iter_dd

        def _do_turn(user_msg, history, session_id, model, turn_q, report_q,
                     role="patient", persona=None):
            """Shared advance logic for practice + compare tabs."""
            existing = SESSIONS.get(session_id, {})
            if session_id is None or existing.get("model") != model or existing.get("role") != role:
                session_id = _new_session(model, turn_q or None, report_q or None,
                                          role=role, patient_persona=persona)["id"]
                history = list(initial_chat) if role == "patient" else []
            session = SESSIONS[session_id]
            session["turn_questionnaires"] = turn_q or DEFAULT_TURN_QUESTIONNAIRES
            session["report_questionnaires"] = report_q or DEFAULT_REPORT_QUESTIONNAIRES
            _advance(session, user_msg)
            history = history + [{"role": "user", "content": user_msg},
                                 {"role": "assistant", "content": session["messages"][-1]["content"]}]
            return history, session_id, session

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
                    with gr.Accordion("Auto-demo patient persona (thesis permutations)", open=False):
                        with gr.Row():
                            p_gender = gr.Dropdown(PERSONA_OPTIONS["gender"], value="Female", label="Gender")
                            p_age = gr.Dropdown(PERSONA_OPTIONS["age"], value=61, label="Age")
                            p_problem = gr.Dropdown(PERSONA_OPTIONS["problem"], value="Smoking", label="Problem")
                        with gr.Row():
                            p_time = gr.Dropdown(PERSONA_OPTIONS["problem_time"], value="ManyYears", label="Problem duration")
                            p_tried = gr.Dropdown(PERSONA_OPTIONS["tried_to_solve"], value="ManyTimes", label="Tried before")
                            p_coop = gr.Dropdown(PERSONA_OPTIONS["cooperation"], value="StartLowAndChangesToHigh", label="Cooperation")
                    export_btn = gr.Button("Export session (.md)", size="sm")
                    export_file = gr.File(label="Session export", visible=False)
                with gr.Column(scale=1):
                    turn_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_TURN_QUESTIONNAIRES,
                                               label="Live judge (every turn — 1 gpt-4o-mini call each)")
                    report_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_REPORT_QUESTIONNAIRES,
                                                 label="Report judge (at session end)")
                    score_plot = gr.LinePlot(_EMPTY_SCORES, x="turn", y="mean", color="instrument",
                                             y_lim=[1, 5], label="Score timeline", height=220)
                    scores_md = gr.Markdown(_scores_markdown([]))
            state = gr.State(None)  # session id

            def on_send(user_msg, history, session_id, method, iteration, turn_q, report_q,
                        role, gender, age, problem, ptime, tried, coop):
                if not user_msg.strip():
                    return "", history, session_id, gr.update(), gr.update()
                persona = build_patient_persona(gender, age, problem, ptime, tried, coop)
                model = "human" if role == "therapist" else _model_name(method, iteration)
                history, session_id, session = _do_turn(
                    user_msg, history, session_id, model, turn_q, report_q, role, persona)
                return ("", history, session_id,
                        _scores_df(session["turn_scores"]), _scores_markdown(session["turn_scores"]))

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
                if session is None or not SCORING_ENABLED or not session["turn_scores"]:
                    return gr.update()
                if "report" not in session:
                    session["report"] = run_report(session["messages"], session["turn_scores"],
                                                   session["report_questionnaires"])
                return _scores_markdown(session["turn_scores"], session["report"])

            def on_demo(method, iteration, turn_q, report_q, gender, age, problem, ptime, tried, coop):
                persona = build_patient_persona(gender, age, problem, ptime, tried, coop)
                state_out = run_demo(_model_name(method, iteration), max_patient_turns=3,
                                     patient_system_prompt=persona,
                                     turn_questionnaires=turn_q or None, report_questionnaires=report_q or None)
                history = [{"role": m["role"], "content": m["content"]} for m in state_out["messages"][1:]]
                return (history, None, _scores_df(state_out["turn_scores"]),
                        _scores_markdown(state_out["turn_scores"], state_out["report"]))

            def on_export(session_id):
                session = SESSIONS.get(session_id)
                if session is None:
                    return gr.update(visible=False)
                path = os.path.join(tempfile.mkdtemp(prefix="mi-coach-"), f"session-{session['id']}.md")
                with open(path, "w") as f:
                    f.write(_session_markdown(session))
                return gr.update(value=path, visible=True)

            def on_reset():
                return list(initial_chat), None, _EMPTY_SCORES, _scores_markdown([]), gr.update(visible=False)

            send_inputs = [msg, chat, state, method_dd, iter_dd, turn_qs, report_qs,
                           role_radio, p_gender, p_age, p_problem, p_time, p_tried, p_coop]
            send.click(on_send, send_inputs, [msg, chat, state, score_plot, scores_md])
            msg.submit(on_send, send_inputs, [msg, chat, state, score_plot, scores_md])
            role_radio.change(on_role, [role_radio], [chat, state, method_dd, iter_dd, msg])
            end.click(on_end, [state], [scores_md])
            demo_btn.click(on_demo, [method_dd, iter_dd, turn_qs, report_qs,
                                     p_gender, p_age, p_problem, p_time, p_tried, p_coop],
                           [chat, state, score_plot, scores_md])
            export_btn.click(on_export, [state], [export_file])
            reset.click(on_reset, None, [chat, state, score_plot, scores_md, export_file])

        # ------------------------------------------------------------- compare
        with gr.Tab("Compare (A/B)"):
            gr.Markdown("Send the **same patient message** to two checkpoints and compare "
                        "replies and judge scores side by side.")
            with gr.Row():
                with gr.Column():
                    method_a, iter_a = _adapter_pickers(" — A")
                    chat_a = gr.Chatbot(value=list(initial_chat), label="A", height=360)
                    scores_a = gr.Markdown(_scores_markdown([]))
                with gr.Column():
                    method_b, iter_b = _adapter_pickers(" — B")
                    chat_b = gr.Chatbot(value=list(initial_chat), label="B", height=360)
                    scores_b = gr.Markdown(_scores_markdown([]))
            cmp_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_TURN_QUESTIONNAIRES,
                                      label="Judge (every turn, per side)")
            cmp_msg = gr.Textbox(label="Your message (as the patient — sent to both)")
            with gr.Row():
                cmp_send = gr.Button("Send to both", variant="primary")
                cmp_reset = gr.Button("New comparison")
            sid_a, sid_b = gr.State(None), gr.State(None)

            def on_cmp_send(user_msg, hist_a, hist_b, a_id, b_id, m_a, i_a, m_b, i_b, qs):
                if not user_msg.strip():
                    return "", hist_a, hist_b, a_id, b_id, gr.update(), gr.update()
                hist_a, a_id, sess_a = _do_turn(user_msg, hist_a, a_id, _model_name(m_a, i_a), qs, None)
                hist_b, b_id, sess_b = _do_turn(user_msg, hist_b, b_id, _model_name(m_b, i_b), qs, None)
                return ("", hist_a, hist_b, a_id, b_id,
                        _scores_markdown(sess_a["turn_scores"]), _scores_markdown(sess_b["turn_scores"]))

            def on_cmp_reset():
                return list(initial_chat), list(initial_chat), None, None, _scores_markdown([]), _scores_markdown([])

            cmp_send.click(on_cmp_send,
                           [cmp_msg, chat_a, chat_b, sid_a, sid_b, method_a, iter_a, method_b, iter_b, cmp_qs],
                           [cmp_msg, chat_a, chat_b, sid_a, sid_b, scores_a, scores_b])
            cmp_msg.submit(on_cmp_send,
                           [cmp_msg, chat_a, chat_b, sid_a, sid_b, method_a, iter_a, method_b, iter_b, cmp_qs],
                           [cmp_msg, chat_a, chat_b, sid_a, sid_b, scores_a, scores_b])
            cmp_reset.click(on_cmp_reset, None, [chat_a, chat_b, sid_a, sid_b, scores_a, scores_b])
    return ui


app = gr.mount_gradio_app(app, _build_ui(), path="/ui")
