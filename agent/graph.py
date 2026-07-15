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
import json
import os
import re
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
PATIENT_MODEL = os.environ.get("PATIENT_MODEL", "gpt-4o-mini")

# Models offered for the judge in Advanced settings, with $/1M-token (input,
# output) prices for the session cost display. gpt-4o-mini stays the default.
JUDGE_MODEL_CHOICES = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]
_PRICES_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
}

# Generation knobs exposed via the API and the UI "Advanced settings" accordion.
DEFAULT_PARAMS = {
    "therapist_temperature": 0.7,
    "therapist_max_tokens": 300,
    "patient_temperature": 0.8,
    "judge_model": JUDGE_MODEL,
    "seed": None,  # int seeds vLLM + OpenAI calls (best-effort); None = unseeded
}


def resolve_params(params: dict | None) -> dict:
    """DEFAULT_PARAMS overlaid with any non-None user overrides."""
    return {**DEFAULT_PARAMS, **{k: v for k, v in (params or {}).items() if v is not None}}


def _seed_kwargs(p: dict) -> dict:
    return {"seed": int(p["seed"])} if p.get("seed") is not None else {}


def empty_usage() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def _add_usage(acc: dict, usage, model: str) -> dict:
    """Accumulate one OpenAI response's token usage and $ cost into acc (in place)."""
    if usage is not None:
        inp, out = _PRICES_PER_MTOK.get(model, _PRICES_PER_MTOK["gpt-4o-mini"])
        acc["calls"] += 1
        acc["prompt_tokens"] += usage.prompt_tokens
        acc["completion_tokens"] += usage.completion_tokens
        acc["cost_usd"] = round(
            acc["cost_usd"] + (usage.prompt_tokens * inp + usage.completion_tokens * out) / 1e6, 6)
    return acc
THERAPIST_SYSTEM_PROMPT = (REPO / "assets" / "therapist_system_prompt.txt").read_text().strip()
GREETING = (
    "Hello, welcome to your first motivational session with me. My name is David and "
    "I`m a professional motivational counselor. Can you start by telling me a little "
    "bit about yourself and why are you here?"
)
STOP = ["<|im_end|>", "<|im_start|>"]  # ChatML markers are plain text for the adapters


def _clean_reply(text: str) -> str:
    """Cut malformed ChatML markers (e.g. "<|im_end>") that slip past vLLM's
    exact-match stop strings."""
    return re.split(r"<\|im_", text)[0].strip()

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


# Thesis patient-persona dimensions (system_prompts_builder.PatientPersonality).
PERSONA_OPTIONS = {
    "gender": ["Female", "Male"],
    "age": [61, 27],
    "problem": ["Smoking", "Obesity"],
    "problem_time": ["ManyYears", "FewMonths"],
    "tried_to_solve": ["ManyTimes", "Never"],
    "cooperation": ["StartLowAndChangesToHigh", "High", "Low"],
}


def build_patient_persona(
    gender: str = "Female",
    age: int = 61,
    problem: str = "Smoking",
    problem_time: str = "ManyYears",
    tried_to_solve: str = "ManyTimes",
    cooperation: str = "StartLowAndChangesToHigh",
) -> str:
    """Build a thesis patient system prompt from named permutation choices."""
    p = prompts_builder.PatientPersonality
    return p.build_system_prompt(
        gender=p.Gender[gender],
        problem=p.Problem[problem],
        problem_time=p.ProblemTime[problem_time],
        tried_to_solve=p.TriedToSolve[tried_to_solve],
        cooperation_level=p.CooperationLevel[cooperation],
        age_value=int(age),
    )["system_prompt"]


def default_patient_persona() -> str:
    """Auto-demo default: Emma, 61, long-time smoker, tried to quit before,
    warms up as the session progresses."""
    return build_patient_persona()


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
    params: dict                  # DEFAULT_PARAMS overrides (temperatures, judge model, seed...)
    turn_rationale: bool          # ask the judge for a one-sentence rationale per turn
    report_rationale: bool        # ...and per report instrument
    openai_usage: dict            # cumulative OpenAI calls/tokens/cost (empty_usage() shape)


def transcript(messages: list[dict]) -> str:
    """[PATIENT]/[THERAPIST] transcript, the format the thesis judges were built for."""
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"[PATIENT]: {m['content'].strip()}")
        elif m["role"] == "assistant":
            lines.append(f"[THERAPIST]: {m['content'].strip()}")
    return "\n\n".join(lines)


def _judge(questionnaire_id, conversation: str, params: dict, rationale: bool = False,
           usage: dict | None = None, **kwargs) -> dict:
    """One structured-output judge call; returns thesis parse_json_response dict."""
    spec = questionnaires.get_prompt_eval_questionnaire(questionnaire_id, conversation, **kwargs)
    prompt, schema = spec["prompt"], spec["schema"]
    if rationale:
        schema = json.loads(json.dumps(schema))  # don't mutate the thesis schema
        schema["properties"]["rationale"] = {
            "type": "string",
            "description": "One concise sentence justifying the overall assessment"}
        schema["required"] = list(schema["required"]) + ["rationale"]
        prompt += ('\n\nAlso include a top-level string field "rationale": one concise sentence '
                   "justifying your overall assessment of the therapist on this instrument.")
    model = params["judge_model"]
    resp = _openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {
            "name": f"questionnaire_{getattr(questionnaire_id, 'value', questionnaire_id)}",
            "strict": True,
            "schema": schema,
        }},
        temperature=0,
        **_seed_kwargs(params),
    )
    if usage is not None:
        _add_usage(usage, resp.usage, model)
    data = json.loads(resp.choices[0].message.content)
    result = questionnaires.parse_json_response(data, questionnaire_id, spec["labels"])
    if rationale and data.get("rationale"):
        result["rationale"] = data["rationale"]
    return result


# ------------------------------------------------------------------------ nodes

def therapist_node(state: SessionState) -> dict:
    p = resolve_params(state.get("params"))
    resp = _vllm.chat.completions.create(
        model=state["model"],
        messages=state["messages"],
        max_tokens=int(p["therapist_max_tokens"]),
        temperature=float(p["therapist_temperature"]),
        stop=STOP,
        **_seed_kwargs(p),
    )
    reply = _clean_reply(resp.choices[0].message.content)
    ended = state.get("session_ended", False) or "SESSION ENDED" in reply
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}],
            "session_ended": ended}


def stream_therapist(messages: list[dict], model: str, params: dict | None = None):
    """Stream a therapist reply from vLLM; yields the growing text (last value is final)."""
    p = resolve_params(params)
    stream = _vllm.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=int(p["therapist_max_tokens"]),
        temperature=float(p["therapist_temperature"]),
        stop=STOP,
        stream=True,
        **_seed_kwargs(p),
    )
    text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            text += delta
            yield _clean_reply(text)


def stream_patient(messages: list[dict], patient_system_prompt: str,
                   params: dict | None = None, usage: dict | None = None):
    """Stream a simulated-patient reply (OpenAI); yields the growing text."""
    p = resolve_params(params)
    stream = _openai.chat.completions.create(
        model=PATIENT_MODEL,
        messages=_patient_messages(messages, patient_system_prompt),
        max_tokens=300,
        temperature=float(p["patient_temperature"]),
        stream=True,
        stream_options={"include_usage": True},
        **_seed_kwargs(p),
    )
    text = ""
    for chunk in stream:
        if usage is not None and chunk.usage is not None:
            _add_usage(usage, chunk.usage, PATIENT_MODEL)
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            text += delta
            yield text.strip()


def _judge_named(name: str, conversation: str, params: dict | None = None,
                 rationale: bool = False, usage: dict | None = None) -> dict:
    """Judge one named instrument; returns a UI/report-friendly dict."""
    qid = QUESTIONNAIRES[name][0]
    kwargs = {"change_goal": "the patient's behavioral change goal"} if name in _NESTED else {}
    result = _judge(qid, conversation, resolve_params(params), rationale=rationale,
                    usage=usage, **kwargs)
    out = {"mean": round(result["mean_score"], 2), "scores": result["scores_dict"]}
    if "globals" in result:
        out["globals"] = result["globals"]
        out["behaviors"] = result["behaviors"]
    if result.get("rationale"):
        out["rationale"] = result["rationale"]
    return out


def judge_turn_node(state: SessionState) -> dict:
    conv = transcript(state["messages"])
    selected = state.get("turn_questionnaires") or DEFAULT_TURN_QUESTIONNAIRES
    usage = dict(state.get("openai_usage") or empty_usage())
    rationale = bool(state.get("turn_rationale"))
    results = {name: _judge_named(name, conv, state.get("params"), rationale, usage)
               for name in selected}
    entry = {
        "therapist_turns": sum(1 for m in state["messages"] if m["role"] == "assistant"),
        "means": {name: r["mean"] for name, r in results.items()},
        "results": results,
    }
    return {"turn_scores": state.get("turn_scores", []) + [entry], "openai_usage": usage}


def _patient_messages(messages: list[dict], patient_system_prompt: str) -> list[dict]:
    """The simulated patient sees the conversation role-flipped (patient=assistant)."""
    flipped = [{"role": "system", "content": patient_system_prompt}]
    for m in messages:
        if m["role"] == "assistant":
            flipped.append({"role": "user", "content": m["content"]})
        elif m["role"] == "user":
            flipped.append({"role": "assistant", "content": m["content"]})
    return flipped


def patient_node(state: SessionState) -> dict:
    """Simulated patient turn (auto-demo / human-therapist mode)."""
    p = resolve_params(state.get("params"))
    usage = dict(state.get("openai_usage") or empty_usage())
    resp = _openai.chat.completions.create(
        model=PATIENT_MODEL,
        messages=_patient_messages(state["messages"], state["patient_system_prompt"]),
        max_tokens=300,
        temperature=float(p["patient_temperature"]),
        **_seed_kwargs(p),
    )
    _add_usage(usage, resp.usage, PATIENT_MODEL)
    reply = resp.choices[0].message.content.strip()
    ended = state.get("session_ended", False) or "SESSION ENDED" in reply
    return {"messages": state["messages"] + [{"role": "user", "content": reply}],
            "session_ended": ended, "openai_usage": usage}


_ASSESSMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overall_rating", "summary", "strengths", "growth_areas", "tip"],
    "properties": {
        "overall_rating": {"type": "integer", "minimum": 1, "maximum": 5,
                           "description": "Overall MI performance of the therapist, 1-5"},
        "summary": {"type": "string", "description": "3-4 sentence overall review of the therapist"},
        "strengths": {"type": "array", "items": {"type": "string"}, "description": "2-3 strengths"},
        "growth_areas": {"type": "array", "items": {"type": "string"}, "description": "2-3 growth areas"},
        "tip": {"type": "string", "description": "One concrete, actionable tip for the next session"},
    },
}


def report_node(state: SessionState) -> dict:
    conv = transcript(state["messages"])
    p = resolve_params(state.get("params"))
    usage = dict(state.get("openai_usage") or empty_usage())
    rationale = bool(state.get("report_rationale"))
    selected = state.get("report_questionnaires") or DEFAULT_REPORT_QUESTIONNAIRES
    results = {name: _judge_named(name, conv, p, rationale, usage) for name in selected}
    per_turn = [t["means"] for t in state.get("turn_scores", [])]
    score_lines = "\n".join(
        f"{name} ({QUESTIONNAIRES[name][1]}): mean {r['mean']} | " + str(r.get("globals", r["scores"]))
        + (f" | behavior counts: {r['behaviors']}" if "behaviors" in r else "")
        + (f" | judge note: {r['rationale']}" if "rationale" in r else "")
        for name, r in results.items()
    )
    # Overall assessment: a reviewer pass over the transcript AND the judges' outputs.
    resp = _openai.chat.completions.create(
        model=p["judge_model"],
        messages=[{"role": "user", "content":
            "You are a senior MI (Motivational Interviewing) supervisor reviewing a practice "
            "session. You are given the transcript and the questionnaire scores produced by "
            "independent judges. Weigh both — where the transcript and the scores disagree, "
            "say so. Review the THERAPIST only.\n\n"
            f"Transcript:\n{conv}\n\nJudge scores:\n{score_lines}"
            + (f"\n\nPer-turn score trajectory: {per_turn}" if per_turn else "")}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "mi_overall_assessment", "strict": True, "schema": _ASSESSMENT_SCHEMA}},
        max_tokens=600,
        temperature=0.3,
        **_seed_kwargs(p),
    )
    _add_usage(usage, resp.usage, p["judge_model"])
    assessment = json.loads(resp.choices[0].message.content)
    return {"report": {
        "results": results,
        "means": {name: r["mean"] for name, r in results.items()},
        "per_turn_means": per_turn,
        "assessment": assessment,
        "usage": usage,  # cumulative session OpenAI usage at report time
    }, "openai_usage": usage}


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


def build_patient_turn_graph():
    """Human-plays-therapist mode: the simulated patient answers the human's
    therapist turn, then the judges score the (human) therapist so far."""
    g = StateGraph(SessionState)
    g.add_node("patient", patient_node)
    g.add_node("judge_turn", judge_turn_node)
    g.add_edge(START, "patient")
    g.add_edge("patient", "judge_turn")
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
patient_turn_graph = build_patient_turn_graph()
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
    params: dict | None = None,
    turn_rationale: bool = False,
    usage: dict | None = None,
) -> SessionState:
    """One interactive practice turn (patient message already appended)."""
    return turn_graph.invoke({
        "messages": messages,
        "model": model,
        "turn_scores": turn_scores,
        "turn_questionnaires": turn_questionnaires or DEFAULT_TURN_QUESTIONNAIRES,
        "params": params or {},
        "turn_rationale": turn_rationale,
        "openai_usage": usage or empty_usage(),
    })


def run_patient_turn(
    messages: list[dict],
    patient_system_prompt: str,
    turn_scores: list[dict],
    turn_questionnaires: list[str] | None = None,
    params: dict | None = None,
    turn_rationale: bool = False,
    usage: dict | None = None,
) -> SessionState:
    """One human-therapist turn (therapist message already appended): the
    simulated patient replies, then the judges score the human's therapy."""
    return patient_turn_graph.invoke({
        "messages": messages,
        "patient_system_prompt": patient_system_prompt,
        "turn_scores": turn_scores,
        "turn_questionnaires": turn_questionnaires or DEFAULT_TURN_QUESTIONNAIRES,
        "params": params or {},
        "turn_rationale": turn_rationale,
        "openai_usage": usage or empty_usage(),
    })


def judge_turn(
    messages: list[dict],
    turn_scores: list[dict],
    turn_questionnaires: list[str] | None = None,
    params: dict | None = None,
    turn_rationale: bool = False,
    usage: dict | None = None,
) -> dict:
    """Judge-only step for the streaming UI path (reply already appended).
    Returns {"turn_scores": [...], "openai_usage": {...}}."""
    return judge_turn_node({
        "messages": messages,
        "turn_scores": turn_scores,
        "turn_questionnaires": turn_questionnaires or DEFAULT_TURN_QUESTIONNAIRES,
        "params": params or {},
        "turn_rationale": turn_rationale,
        "openai_usage": usage or empty_usage(),
    })


def run_report(
    messages: list[dict],
    turn_scores: list[dict],
    report_questionnaires: list[str] | None = None,
    params: dict | None = None,
    report_rationale: bool = False,
    usage: dict | None = None,
) -> dict:
    """Full-session report; report["usage"] carries the updated cumulative usage."""
    return report_node({
        "messages": messages,
        "turn_scores": turn_scores,
        "report_questionnaires": report_questionnaires or DEFAULT_REPORT_QUESTIONNAIRES,
        "params": params or {},
        "report_rationale": report_rationale,
        "openai_usage": usage or empty_usage(),
    })["report"]


def run_demo(
    model: str,
    max_patient_turns: int = 4,
    patient_system_prompt: str | None = None,
    turn_questionnaires: list[str] | None = None,
    report_questionnaires: list[str] | None = None,
    params: dict | None = None,
    turn_rationale: bool = False,
    report_rationale: bool = False,
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
            "params": params or {},
            "turn_rationale": turn_rationale,
            "report_rationale": report_rationale,
            "openai_usage": empty_usage(),
        },
        {"recursion_limit": 8 * max_patient_turns + 10},
    )
