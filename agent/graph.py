"""LangGraph agent layer for MI Coach (Phase 3).

Two compiled graphs over a shared session state:

- ``turn_graph`` (interactive practice): therapist (local vLLM) -> judge_turn
  (gpt-4o-mini scoring the running transcript with thesis questionnaire Q1).
  Invoked once per patient message; the human plays the patient.
- ``demo_graph`` (auto-demo): patient_sim (gpt-4o-mini persona from the thesis
  prompt builder) -> therapist -> judge_turn -> loop until the session ends or
  ``max_patient_turns`` is reached -> report.

The final report (also callable on interactive sessions via ``run_report``)
scores the full transcript with Q2 (17 items) + MITI globals/behaviors and adds
a short narrative summary. Judge calls use OpenAI structured outputs with the
thesis JSON schemas, so parsing is deterministic.

Budget: every OpenAI call is gpt-4o-mini; a 5-turn demo session costs ~a cent.
"""

import importlib.util
import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]

# Load repo-root .env (OPENAI_API_KEY etc.) without overriding real env vars.
if (REPO / ".env").is_file():
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "assets" / "thesis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


questionnaires = _load("questionnaires")
prompts_builder = _load("system_prompts_builder")

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
THERAPIST_SYSTEM_PROMPT = (REPO / "assets" / "therapist_system_prompt.txt").read_text().strip()
GREETING = (
    "Hello, welcome to your first motivational session with me. My name is David and "
    "I`m a professional motivational counselor. Can you start by telling me a little "
    "bit about yourself and why are you here?"
)
STOP = ["<|im_end|>", "<|im_start|>"]  # ChatML markers are plain text for the adapters

_vllm = OpenAI(base_url=VLLM_URL, api_key="unused")
_openai = OpenAI()  # OPENAI_API_KEY from env

# Selectable judge instruments (thesis questionnaires.py). Key -> (ID, blurb).
QUESTIONNAIRES = {
    "Q1": (questionnaires.QuestionnaireID.Q1, "Satisfaction, 5 items (thesis primary)"),
    "Q2": (questionnaires.QuestionnaireID.Q2, "Therapist behaviors, 17 items (thesis primary)"),
    "WAI-SR": (questionnaires.QuestionnaireID.WAI_SR, "Working Alliance Inventory (short)"),
    "CSQ-8": (questionnaires.QuestionnaireID.CSQ8, "Client Satisfaction Questionnaire"),
    "MI-SAT": (questionnaires.QuestionnaireID.MI_SAT, "MI intervention satisfaction"),
    "MITI": (questionnaires.QuestionnaireID.MITI, "MI Treatment Integrity: globals + behavior counts"),
    "PCT": (questionnaires.QuestionnaireID.PCT, "Patient change talk / readiness"),
    "MICI": (questionnaires.QuestionnaireID.MICI, "MI-inconsistent behaviors (lower is better)"),
}
DEFAULT_TURN_QUESTIONNAIRES = ["Q1"]
DEFAULT_REPORT_QUESTIONNAIRES = ["Q2", "MITI"]
# Nested instruments take a change_goal hint in their prompts.
_NESTED = {"MITI", "PCT", "MICI"}


def default_patient_persona() -> str:
    """Thesis patient persona used for auto-demo: Emma, 61, long-time smoker,
    tried to quit before, warms up as the session progresses."""
    p = prompts_builder.PatientPersonality
    return p.build_system_prompt(
        gender=p.Gender.Female,
        problem=p.Problem.Smoking,
        problem_time=p.ProblemTime.ManyYears,
        tried_to_solve=p.TriedToSolve.ManyTimes,
        cooperation_level=p.CooperationLevel.StartLowAndChangesToHigh,
        age_value=61,
    )["system_prompt"]


class SessionState(TypedDict, total=False):
    messages: list[dict]          # therapist-perspective: patient=user, therapist=assistant
    model: str                    # served therapist model (adapter) name
    turn_questionnaires: list[str]    # QUESTIONNAIRES keys judged every turn
    report_questionnaires: list[str]  # QUESTIONNAIRES keys judged at session end
    turn_scores: list[dict]       # one entry per therapist turn
    report: dict | None
    patient_system_prompt: str    # auto-demo only
    max_patient_turns: int        # auto-demo only
    session_ended: bool


def transcript(messages: list[dict]) -> str:
    """[PATIENT]/[THERAPIST] transcript, the format the thesis judges were built for."""
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"[PATIENT]: {m['content'].strip()}")
        elif m["role"] == "assistant":
            lines.append(f"[THERAPIST]: {m['content'].strip()}")
    return "\n\n".join(lines)


def _judge(questionnaire_id, conversation: str, **kwargs) -> dict:
    """One structured-output judge call; returns thesis parse_json_response dict."""
    spec = questionnaires.get_prompt_eval_questionnaire(questionnaire_id, conversation, **kwargs)
    resp = _openai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": spec["prompt"]}],
        response_format={"type": "json_schema", "json_schema": {
            "name": f"questionnaire_{getattr(questionnaire_id, 'value', questionnaire_id)}",
            "strict": True,
            "schema": spec["schema"],
        }},
        temperature=0,
    )
    return questionnaires.parse_json_response(
        resp.choices[0].message.content, questionnaire_id, spec["labels"]
    )


# ------------------------------------------------------------------------ nodes

def therapist_node(state: SessionState) -> dict:
    resp = _vllm.chat.completions.create(
        model=state["model"],
        messages=state["messages"],
        max_tokens=300,
        temperature=0.7,
        stop=STOP,
    )
    reply = resp.choices[0].message.content.strip()
    ended = state.get("session_ended", False) or "SESSION ENDED" in reply
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}],
            "session_ended": ended}


def _judge_named(name: str, conversation: str) -> dict:
    """Judge one named instrument; returns a UI/report-friendly dict."""
    qid = QUESTIONNAIRES[name][0]
    kwargs = {"change_goal": "the patient's behavioral change goal"} if name in _NESTED else {}
    result = _judge(qid, conversation, **kwargs)
    out = {"mean": round(result["mean_score"], 2), "scores": result["scores_dict"]}
    if "globals" in result:
        out["globals"] = result["globals"]
        out["behaviors"] = result["behaviors"]
    return out


def judge_turn_node(state: SessionState) -> dict:
    conv = transcript(state["messages"])
    selected = state.get("turn_questionnaires") or DEFAULT_TURN_QUESTIONNAIRES
    results = {name: _judge_named(name, conv) for name in selected}
    entry = {
        "therapist_turns": sum(1 for m in state["messages"] if m["role"] == "assistant"),
        "means": {name: r["mean"] for name, r in results.items()},
        "results": results,
    }
    return {"turn_scores": state.get("turn_scores", []) + [entry]}


def patient_node(state: SessionState) -> dict:
    """Simulated patient: sees the conversation role-flipped (patient=assistant)."""
    flipped = [{"role": "system", "content": state["patient_system_prompt"]}]
    for m in state["messages"]:
        if m["role"] == "assistant":
            flipped.append({"role": "user", "content": m["content"]})
        elif m["role"] == "user":
            flipped.append({"role": "assistant", "content": m["content"]})
    resp = _openai.chat.completions.create(
        model=JUDGE_MODEL, messages=flipped, max_tokens=300, temperature=0.8,
    )
    reply = resp.choices[0].message.content.strip()
    ended = state.get("session_ended", False) or "SESSION ENDED" in reply
    return {"messages": state["messages"] + [{"role": "user", "content": reply}],
            "session_ended": ended}


def report_node(state: SessionState) -> dict:
    conv = transcript(state["messages"])
    selected = state.get("report_questionnaires") or DEFAULT_REPORT_QUESTIONNAIRES
    results = {name: _judge_named(name, conv) for name in selected}
    per_turn = [t["means"] for t in state.get("turn_scores", [])]
    score_lines = "\n".join(
        f"{name} ({QUESTIONNAIRES[name][1]}): " + str(r.get("globals", r["scores"]))
        + (f" | behavior counts: {r['behaviors']}" if "behaviors" in r else "")
        for name, r in results.items()
    )
    narrative = _openai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content":
            "You are an MI (Motivational Interviewing) supervisor. Based on this practice-session "
            "transcript and its questionnaire scores, write a concise feedback report (<= 150 words) "
            "on the THERAPIST's MI performance: 2-3 strengths, 2-3 growth areas, one concrete tip. "
            f"\n\nTranscript:\n{conv}\n\nScores:\n{score_lines}"}],
        max_tokens=400,
        temperature=0.3,
    ).choices[0].message.content.strip()
    return {"report": {
        "results": results,
        "means": {name: r["mean"] for name, r in results.items()},
        "per_turn_means": per_turn,
        "narrative": narrative,
    }}


# ----------------------------------------------------------------------- graphs

def _continue_demo(state: SessionState) -> str:
    patient_turns = sum(1 for m in state["messages"] if m["role"] == "user")
    if state.get("session_ended") or patient_turns >= state.get("max_patient_turns", 4):
        return "report"
    return "patient"


def build_turn_graph():
    g = StateGraph(SessionState)
    g.add_node("therapist", therapist_node)
    g.add_node("judge_turn", judge_turn_node)
    g.add_edge(START, "therapist")
    g.add_edge("therapist", "judge_turn")
    g.add_edge("judge_turn", END)
    return g.compile()


def build_demo_graph():
    g = StateGraph(SessionState)
    g.add_node("patient", patient_node)
    g.add_node("therapist", therapist_node)
    g.add_node("judge_turn", judge_turn_node)
    g.add_node("report", report_node)
    g.add_edge(START, "patient")
    g.add_edge("patient", "therapist")
    g.add_edge("therapist", "judge_turn")
    g.add_conditional_edges("judge_turn", _continue_demo, {"patient": "patient", "report": "report"})
    g.add_edge("report", END)
    return g.compile()


turn_graph = build_turn_graph()
demo_graph = build_demo_graph()


def initial_messages() -> list[dict]:
    return [
        {"role": "system", "content": THERAPIST_SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
    ]


def run_turn(
    messages: list[dict],
    model: str,
    turn_scores: list[dict],
    turn_questionnaires: list[str] | None = None,
) -> SessionState:
    """One interactive practice turn (patient message already appended)."""
    return turn_graph.invoke({
        "messages": messages,
        "model": model,
        "turn_scores": turn_scores,
        "turn_questionnaires": turn_questionnaires or DEFAULT_TURN_QUESTIONNAIRES,
    })


def run_report(
    messages: list[dict],
    turn_scores: list[dict],
    report_questionnaires: list[str] | None = None,
) -> dict:
    return report_node({
        "messages": messages,
        "turn_scores": turn_scores,
        "report_questionnaires": report_questionnaires or DEFAULT_REPORT_QUESTIONNAIRES,
    })["report"]


def run_demo(
    model: str,
    max_patient_turns: int = 4,
    patient_system_prompt: str | None = None,
    turn_questionnaires: list[str] | None = None,
    report_questionnaires: list[str] | None = None,
) -> SessionState:
    """Full simulated session: patient-sim <-> therapist with per-turn scoring + report."""
    return demo_graph.invoke(
        {
            "messages": initial_messages(),
            "model": model,
            "turn_scores": [],
            "turn_questionnaires": turn_questionnaires or DEFAULT_TURN_QUESTIONNAIRES,
            "report_questionnaires": report_questionnaires or DEFAULT_REPORT_QUESTIONNAIRES,
            "patient_system_prompt": patient_system_prompt or default_patient_persona(),
            "max_patient_turns": max_patient_turns,
            "session_ended": False,
        },
        {"recursion_limit": 8 * max_patient_turns + 10},
    )
