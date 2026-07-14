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
import uuid

import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.graph import (
    DEFAULT_REPORT_QUESTIONNAIRES,
    DEFAULT_TURN_QUESTIONNAIRES,
    GREETING,
    QUESTIONNAIRES,
    default_patient_persona,
    initial_messages,
    run_demo,
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
    model: str = Field(default=DEFAULT_MODEL, description="Served therapist model name")
    turn_questionnaires: list[str] = Field(
        default=DEFAULT_TURN_QUESTIONNAIRES, description=f"Judged every turn; any of {list(QUESTIONNAIRES)}")
    report_questionnaires: list[str] = Field(
        default=DEFAULT_REPORT_QUESTIONNAIRES, description="Judged at session end")


class PatientMessage(BaseModel):
    content: str


class DemoRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    max_patient_turns: int = Field(default=4, ge=1, le=8)
    turn_questionnaires: list[str] = Field(default=DEFAULT_TURN_QUESTIONNAIRES)
    report_questionnaires: list[str] = Field(default=DEFAULT_REPORT_QUESTIONNAIRES)


def _new_session(model: str, turn_qs: list[str] | None = None, report_qs: list[str] | None = None) -> dict:
    session = {
        "id": uuid.uuid4().hex[:12],
        "model": model,
        "messages": initial_messages(),
        "turn_scores": [],
        "turn_questionnaires": turn_qs or DEFAULT_TURN_QUESTIONNAIRES,
        "report_questionnaires": report_qs or DEFAULT_REPORT_QUESTIONNAIRES,
    }
    SESSIONS[session["id"]] = session
    return session


def _advance(session: dict, patient_message: str) -> dict:
    """Append the patient turn, run therapist (+judge when enabled), return last score."""
    session["messages"].append({"role": "user", "content": patient_message})
    if SCORING_ENABLED:
        state = run_turn(session["messages"], session["model"], session["turn_scores"],
                         session["turn_questionnaires"])
        session["messages"] = state["messages"]
        session["turn_scores"] = state["turn_scores"]
        return session["turn_scores"][-1]
    # No judge available: therapist only.
    from agent.graph import therapist_node
    session["messages"] = therapist_node({"messages": session["messages"], "model": session["model"]})["messages"]
    return {}


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
    session = _new_session(
        body.model,
        _validate_questionnaires(body.turn_questionnaires),
        _validate_questionnaires(body.report_questionnaires),
    )
    return {"session_id": session["id"], "model": session["model"], "greeting": GREETING,
            "turn_questionnaires": session["turn_questionnaires"],
            "report_questionnaires": session["report_questionnaires"]}


@app.post("/sessions/{session_id}/message")
def send_message(session_id: str, body: PatientMessage) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    turn_score = _advance(session, body.content)
    return {
        "therapist": session["messages"][-1]["content"],
        "turns": len(session["messages"]) - 1,
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
    state = run_demo(
        body.model,
        max_patient_turns=body.max_patient_turns,
        turn_questionnaires=_validate_questionnaires(body.turn_questionnaires),
        report_questionnaires=_validate_questionnaires(body.report_questionnaires),
    )
    return {
        "model": body.model,
        "messages": state["messages"][1:],  # drop system prompt
        "turn_scores": state["turn_scores"],
        "report": state["report"],
        "patient_persona": default_patient_persona(),
    }


# --------------------------------------------------------------------------- ui


BEST_ITER = {"pto": 10, "grpo": 8}  # thesis-best iteration per method


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
        lines.append(f"\n**Supervisor feedback:**\n\n{report['narrative']}")
    return "\n".join(lines)


def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="MI Coach") as ui:
        gr.Markdown(f"# MI Coach — practice session\n*{DISCLAIMER}*\n\n"
                    "**You play the patient**; the fine-tuned model plays the therapist. "
                    "Each therapist turn is scored live by an LLM judge using the thesis questionnaires.")
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
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    method_dd = gr.Dropdown(
                        [(m.upper(), m) for m in methods] + [("Base model (no adapter)", "base")],
                        value="pto" if "pto" in methods else methods[0], label="Therapist adapter")
                    iter_dd = gr.Dropdown(_iter_choices("pto"), value=BEST_ITER.get("pto", 10),
                                          label="Iteration (★ = thesis best)")
                chat = gr.Chatbot(value=list(initial_chat), label="Session", height=430)
                msg = gr.Textbox(label="Your message (as the patient)", placeholder="Hi David, I'm here because...")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    end = gr.Button("End session → report")
                    demo_btn = gr.Button("Auto-demo (simulated patient)")
                    reset = gr.Button("New session")
            with gr.Column(scale=1):
                turn_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_TURN_QUESTIONNAIRES,
                                           label="Live judge (every turn — 1 gpt-4o-mini call each)")
                report_qs = gr.CheckboxGroup(q_choices, value=DEFAULT_REPORT_QUESTIONNAIRES,
                                             label="Report judge (at session end)")
                scores_md = gr.Markdown(_scores_markdown([]))
        state = gr.State(None)  # session id

        def on_method(method):
            best = BEST_ITER.get(method)
            return gr.update(choices=_iter_choices(method), value=best or adapters.get(method, [1])[-1],
                             visible=method != "base")

        def on_send(user_msg, history, session_id, method, iteration, turn_q, report_q):
            if not user_msg.strip():
                return "", history, session_id, gr.update()
            model = _model_name(method, iteration)
            if session_id is None or SESSIONS.get(session_id, {}).get("model") != model:
                session_id = _new_session(model, turn_q or None, report_q or None)["id"]
                history = list(initial_chat)
            session = SESSIONS[session_id]
            session["turn_questionnaires"] = turn_q or DEFAULT_TURN_QUESTIONNAIRES
            session["report_questionnaires"] = report_q or DEFAULT_REPORT_QUESTIONNAIRES
            _advance(session, user_msg)
            history = history + [{"role": "user", "content": user_msg},
                                 {"role": "assistant", "content": session["messages"][-1]["content"]}]
            return "", history, session_id, _scores_markdown(session["turn_scores"])

        def on_end(session_id):
            session = SESSIONS.get(session_id)
            if session is None or not SCORING_ENABLED or len(session["messages"]) < 4:
                return gr.update()
            if "report" not in session:
                session["report"] = run_report(session["messages"], session["turn_scores"],
                                               session["report_questionnaires"])
            return _scores_markdown(session["turn_scores"], session["report"])

        def on_demo(method, iteration, turn_q, report_q):
            state_out = run_demo(_model_name(method, iteration), max_patient_turns=3,
                                 turn_questionnaires=turn_q or None, report_questionnaires=report_q or None)
            history = [{"role": m["role"], "content": m["content"]} for m in state_out["messages"][1:]]
            return history, None, _scores_markdown(state_out["turn_scores"], state_out["report"])

        def on_reset():
            return list(initial_chat), None, _scores_markdown([])

        method_dd.change(on_method, [method_dd], [iter_dd])
        send.click(on_send, [msg, chat, state, method_dd, iter_dd, turn_qs, report_qs], [msg, chat, state, scores_md])
        msg.submit(on_send, [msg, chat, state, method_dd, iter_dd, turn_qs, report_qs], [msg, chat, state, scores_md])
        end.click(on_end, [state], [scores_md])
        demo_btn.click(on_demo, [method_dd, iter_dd, turn_qs, report_qs], [chat, state, scores_md])
        reset.click(on_reset, None, [chat, state, scores_md])
    return ui


app = gr.mount_gradio_app(app, _build_ui(), path="/ui")
