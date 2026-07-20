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

import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from agent.config import DEFAULT_PARAMS, JUDGE_MODEL_CHOICES, VLLM_URL
from agent.graph import run_demo
from agent.judging import (
    CUSTOM_QUESTIONNAIRES,
    DEFAULT_REPORT_QUESTIONNAIRES,
    DEFAULT_TURN_QUESTIONNAIRES,
    QUESTIONNAIRES,
    add_custom_questionnaire,
    delete_custom_questionnaire,
    known_questionnaires,
)
from agent.thesis import GREETING, PERSONA_OPTIONS, build_patient_persona
from app import sessions
from app.rendering import session_markdown
from app.sessions import (
    DEFAULT_MODEL,
    DISCLAIMER,
    SESSIONS,
    advance,
    ensure_report,
    new_session,
    run_comparison,
    session_summary,
    store_demo_session,
)
from app.ui import build_ui


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


async def _list_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{VLLM_URL}/models")
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


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
    return {"status": "ok", "vllm_models": models, "live_scoring": sessions.SCORING_ENABLED}


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSession) -> dict:
    if body.role not in ("patient", "therapist"):
        raise HTTPException(status_code=422, detail="role must be 'patient' or 'therapist'")
    try:
        persona = build_patient_persona(body.gender, body.age, body.problem,
                                        body.problem_time, body.tried_to_solve, body.cooperation)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"invalid persona option: {e}; valid: {PERSONA_OPTIONS}")
    session = new_session(
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
    return [session_summary(s) for s in SESSIONS.values()]


@app.post("/sessions/{session_id}/message")
def send_message(session_id: str, body: PatientMessage) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    turn_score = advance(session, body.content)
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
    if not sessions.SCORING_ENABLED:
        raise HTTPException(status_code=503, detail="no OPENAI_API_KEY — judge disabled")
    return ensure_report(session)


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
    if not sessions.SCORING_ENABLED:
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
    session = store_demo_session(body.model, state, persona)
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
    if not sessions.SCORING_ENABLED:
        raise HTTPException(status_code=503, detail="no OPENAI_API_KEY — judge disabled")
    sess_a, sess_b = SESSIONS.get(body.session_a), SESSIONS.get(body.session_b)
    if sess_a is None or sess_b is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if not sess_a["messages"] or not sess_b["messages"]:
        raise HTTPException(status_code=422, detail="both sessions need at least one turn")
    return run_comparison(sess_a, sess_b)


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


@app.get("/sessions/{session_id}/export", response_class=PlainTextResponse)
def export_session(session_id: str) -> str:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return session_markdown(session)


# --------------------------------------------------------------------------- ui

app = gr.mount_gradio_app(app, build_ui(), path="/ui")
