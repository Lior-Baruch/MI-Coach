"""The judge: questionnaire registry, custom instruments, structured scoring.

One judge call = one instrument scored over the full transcript so far, via
OpenAI **structured outputs** (``response_format: json_schema``) — the reply
is guaranteed to parse; there is no free-text score scraping. Thesis
instruments get their prompt + strict schema from the copied thesis code
(``agent.thesis.questionnaires``); custom instruments get a generic prompt
built from user-defined statements.

``QUESTIONNAIRES`` is the single source of truth for which thesis instruments
exist; UI checkbox lists and API validation both derive from it (plus
``CUSTOM_QUESTIONNAIRES``). ``CUSTOM_QUESTIONNAIRES`` crosses modules by
object identity — mutate it in place, never rebind it.

The optional judge *rationale* is added by copying the schema at call time
and appending one instruction line, so the thesis files stay untouched.
"""

import json
import re

from agent import config
from agent.config import REPO, add_usage, resolve_params, seed_kwargs
from agent.thesis import questionnaires

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

# User-defined instruments: a name + statements about the therapist, each rated
# 1-5 by the judge over the running transcript (same structured-output call as
# the thesis instruments). Persisted so they survive restarts.
CUSTOM_QUESTIONNAIRES_FILE = REPO / "data" / "custom_questionnaires.json"
CUSTOM_QUESTIONNAIRES: dict[str, dict] = {}  # name -> {"description": str, "items": [str]}
if CUSTOM_QUESTIONNAIRES_FILE.is_file():
    CUSTOM_QUESTIONNAIRES.update(json.loads(CUSTOM_QUESTIONNAIRES_FILE.read_text()))


def questionnaire_blurbs() -> dict[str, str]:
    """name -> one-line blurb for every selectable instrument (thesis + custom)."""
    out = {name: blurb for name, (_, blurb) in QUESTIONNAIRES.items()}
    for name, q in CUSTOM_QUESTIONNAIRES.items():
        desc = f": {q['description']}" if q.get("description") else ""
        out[name] = f"custom, {len(q['items'])} items{desc}"
    return out


def known_questionnaires() -> list[str]:
    return list(QUESTIONNAIRES) + list(CUSTOM_QUESTIONNAIRES)


def filter_known(names: list[str]) -> list[str]:
    """Drop questionnaire names that no longer exist (e.g. a deleted custom one)."""
    return [n for n in names if n in QUESTIONNAIRES or n in CUSTOM_QUESTIONNAIRES]


def _save_custom_questionnaires() -> None:
    CUSTOM_QUESTIONNAIRES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_QUESTIONNAIRES_FILE.write_text(json.dumps(CUSTOM_QUESTIONNAIRES, indent=2) + "\n")


def add_custom_questionnaire(name: str, items: list[str], description: str = "") -> None:
    """Register (or overwrite) a custom instrument and persist it."""
    name = name.strip()
    items = [i.strip() for i in items if i and i.strip()]
    if not name:
        raise ValueError("questionnaire name must not be empty")
    if len(name) > 60:
        raise ValueError("questionnaire name must be at most 60 characters")
    if name in QUESTIONNAIRES:
        raise ValueError(f"{name!r} is a built-in thesis questionnaire; pick another name")
    if not 1 <= len(items) <= 20:
        raise ValueError("a custom questionnaire needs 1-20 non-empty statements")
    CUSTOM_QUESTIONNAIRES[name] = {"description": description.strip(), "items": items}
    _save_custom_questionnaires()


def delete_custom_questionnaire(name: str) -> None:
    if name not in CUSTOM_QUESTIONNAIRES:
        raise KeyError(name)
    del CUSTOM_QUESTIONNAIRES[name]
    _save_custom_questionnaires()


def _structured_judge_call(name: str, prompt: str, schema: dict, params: dict,
                           rationale: bool = False, usage: dict | None = None) -> dict:
    """One structured-output judge call; returns the parsed JSON dict."""
    if rationale:
        schema = json.loads(json.dumps(schema))  # don't mutate the caller's schema
        schema["properties"]["rationale"] = {
            "type": "string",
            "description": "One concise sentence justifying the overall assessment"}
        schema["required"] = list(schema["required"]) + ["rationale"]
        prompt += ('\n\nAlso include a top-level string field "rationale": one concise sentence '
                   "justifying your overall assessment of the therapist on this instrument.")
    model = params["judge_model"]
    resp = config.openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {
            "name": name, "strict": True, "schema": schema}},
        temperature=0,
        **seed_kwargs(params),
    )
    if usage is not None:
        add_usage(usage, resp.usage, model)
    return json.loads(resp.choices[0].message.content)


def _judge_thesis(questionnaire_id, conversation: str, params: dict, rationale: bool = False,
                  usage: dict | None = None, **kwargs) -> dict:
    """Judge one thesis instrument; returns the thesis parse_json_response dict."""
    spec = questionnaires.get_prompt_eval_questionnaire(questionnaire_id, conversation, **kwargs)
    data = _structured_judge_call(
        f"questionnaire_{getattr(questionnaire_id, 'value', questionnaire_id)}",
        spec["prompt"], spec["schema"], params, rationale, usage)
    result = questionnaires.parse_json_response(data, questionnaire_id, spec["labels"])
    if rationale and data.get("rationale"):
        result["rationale"] = data["rationale"]
    return result


def _judge_custom(name: str, conversation: str, params: dict, rationale: bool = False,
                  usage: dict | None = None) -> dict:
    """Judge a custom instrument: prompt assembled from the user's statements,
    same transcript framing and 1-5 scale as the thesis questionnaires."""
    q = CUSTOM_QUESTIONNAIRES[name]
    items = q["items"]
    numbered = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
    prompt = (
        "You are a professional motivational interview therapist; the conversation below is "
        "between a patient [PATIENT] and a therapist [THERAPIST]. Evaluate the THERAPIST by "
        "rating each statement with a single integer on a scale of 1-5, with 1 meaning the "
        "statement is not true at all of the therapist's behavior and 5 meaning it is fully "
        "true. Use critical thinking and your professional experience; be honest and fair.\n\n"
        f"Instrument: {name}"
        + (f" — {q['description']}" if q.get("description") else "") + "\n\n"
        f"Statements:\n{numbered}\n\n"
        '**IMPORTANT**: Output your response as a JSON object: {"scores": [<score1>, <score2>, ...]} '
        "with one integer (1-5) per statement, in order.\n\n"
        "Transcript:\n--------------------\n"
        f"{conversation}\n--------------------"
    )
    schema = {
        "type": "object",
        "properties": {"scores": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1, "maximum": 5},
            "minItems": len(items), "maxItems": len(items),
        }},
        "required": ["scores"],
        "additionalProperties": False,
    }
    safe_name = "custom_" + re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:40]
    data = _structured_judge_call(safe_name, prompt, schema, params, rationale, usage)
    scores = data["scores"]
    out = {"mean": round(sum(scores) / len(scores), 2), "scores": dict(zip(items, scores))}
    if rationale and data.get("rationale"):
        out["rationale"] = data["rationale"]
    return out


def judge_named(name: str, conversation: str, params: dict | None = None,
                rationale: bool = False, usage: dict | None = None) -> dict:
    """Judge one named instrument (thesis or custom); returns a UI/report-friendly dict."""
    if name in CUSTOM_QUESTIONNAIRES:
        return _judge_custom(name, conversation, resolve_params(params), rationale, usage)
    qid = QUESTIONNAIRES[name][0]
    kwargs = {"change_goal": "the patient's behavioral change goal"} if name in _NESTED else {}
    result = _judge_thesis(qid, conversation, resolve_params(params), rationale=rationale,
                           usage=usage, **kwargs)
    out = {"mean": round(result["mean_score"], 2), "scores": result["scores_dict"]}
    if "globals" in result:
        out["globals"] = result["globals"]
        out["behaviors"] = result["behaviors"]
    if result.get("rationale"):
        out["rationale"] = result["rationale"]
    return out
