"""Session model, in-memory store, and the session lifecycle.

A session is the app's unit of state. Sessions are deliberately in-memory
only — MI Coach is a practice tool, not a clinical record store (that is also
why there is no database anywhere).

Shared-state rules other modules rely on:

- ``SESSIONS`` crosses modules by object identity — mutate it in place, never
  rebind it. Read ``SCORING_ENABLED`` as ``sessions.SCORING_ENABLED`` (module
  attribute) so tests can monkeypatch it in one place.
- ``report`` and ``comparison`` are ABSENT from a session until set; their
  presence is the "has report / has verdict" contract used by the API
  (``GET /sessions/{id}`` returns report only once it exists) and history.
- usage aliasing is intentional and observable: after ``ensure_report`` the
  session's ``usage`` IS the report's ``usage`` dict, so later costs folded
  into the session (e.g. a comparative review, billed to side A) show up in
  both places.
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TypedDict

from fastapi import HTTPException

from agent.config import empty_usage
from agent.graph import (
    compare_sessions,
    patient_node,
    run_patient_turn,
    run_report,
    run_turn,
    therapist_node,
)
from agent.judging import DEFAULT_REPORT_QUESTIONNAIRES, DEFAULT_TURN_QUESTIONNAIRES
from agent.thesis import build_patient_persona, initial_messages

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mi-coach-pto-iter10")
SCORING_ENABLED = bool(os.environ.get("OPENAI_API_KEY"))

DISCLAIMER = (
    "MI Coach is a practice tool for Motivational Interviewing skills. "
    "It is not therapy and must not be used as a substitute for professional care."
)


class Session(TypedDict, total=False):
    """The session dict shape (a plain dict at runtime; typed for readers).

    ``report`` and ``comparison`` are ABSENT until produced — presence is the
    contract, so they are declared here but never pre-initialized.
    """

    id: str
    created_at: str
    kind: str                         # practice | compare | demo
    model: str                        # served therapist model (adapter) name
    role: str                         # which role the HUMAN plays: patient | therapist
    messages: list[dict]              # therapist-perspective: patient=user, therapist=assistant
    patient_persona: str              # thesis persona system prompt (therapist-role + demo)
    turn_scores: list[dict]           # one entry per judged therapist turn
    turn_questionnaires: list[str]    # judged every turn ([] = per-turn judging off)
    report_questionnaires: list[str]  # judged at session end
    params: dict                      # DEFAULT_PARAMS overrides (partial)
    turn_rationale: bool
    report_rationale: bool
    usage: dict                       # cumulative OpenAI usage/cost (empty_usage() shape)
    report: dict                      # set once by ensure_report(); cached
    comparison: dict                  # set on both sides by run_comparison(); cached


SESSIONS: dict[str, Session] = {}


def new_session(model: str, turn_qs: list[str] | None = None, report_qs: list[str] | None = None,
                role: str = "patient", patient_persona: str | None = None,
                params: dict | None = None, turn_rationale: bool = False,
                report_rationale: bool = False, kind: str = "practice") -> Session:
    session: Session = {
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


def store_demo_session(model: str, state: dict, persona: str, kind: str = "demo") -> Session:
    """Record a finished auto-demo run so it shows up in history/export."""
    session = new_session(model, state.get("turn_questionnaires"), state.get("report_questionnaires"),
                          role="patient", patient_persona=persona,
                          params=state.get("params") or {},
                          turn_rationale=bool(state.get("turn_rationale")),
                          report_rationale=bool(state.get("report_rationale")), kind=kind)
    session["messages"] = state["messages"]
    session["turn_scores"] = state["turn_scores"]
    session["report"] = state["report"]
    session["usage"] = state.get("openai_usage") or empty_usage()
    return session


def advance(session: Session, user_message: str) -> dict:
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


def ensure_report(session: Session) -> dict:
    """Generate (once) and cache the end-of-session report.

    Afterwards session["usage"] IS report["usage"] (same dict) — deliberate,
    so post-report costs (e.g. a comparative review) appear in both.
    """
    if "report" not in session:
        session["report"] = run_report(session["messages"], session["turn_scores"],
                                       session["report_questionnaires"], session.get("params"),
                                       session.get("report_rationale", False),
                                       dict(session.get("usage") or empty_usage()))
        session["usage"] = session["report"]["usage"]
    return session["report"]


def run_comparison(sess_a: Session, sess_b: Session) -> dict:
    """Comparative final review of two sessions: make sure both reports exist,
    then one judge call over both transcripts+reports. Cached on both sessions;
    the comparison call's cost is counted on side A."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa, fb = pool.submit(ensure_report, sess_a), pool.submit(ensure_report, sess_b)
        fa.result(), fb.result()
    cmp = compare_sessions(sess_a["model"], sess_a["messages"], sess_a["report"],
                           sess_b["model"], sess_b["messages"], sess_b["report"],
                           sess_a.get("params"), sess_a["usage"])
    comparison = {"model_a": sess_a["model"], "model_b": sess_b["model"], **cmp}
    sess_a["comparison"] = sess_b["comparison"] = comparison
    return comparison


def session_summary(session: Session) -> dict:
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
