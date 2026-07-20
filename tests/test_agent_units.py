"""Unit pins for the agent layer: params, usage math, transcripts, judging,
custom questionnaires, the demo loop, and the streaming generators."""

import itertools
import json

import pytest

from agent import config as C
from agent import graph as G
from agent import thesis as T
from tests.fakes import COST_PER_MINI_CALL
from tests.patch_points import CUSTOM_FILE, resolve
from types import SimpleNamespace

MITI_GLOBAL_LABELS = {
    "MITI1_CultivatingChangeTalk",
    "MITI2_SofteningSustainTalk",
    "MITI3_Partnership",
    "MITI4_Empathy",
}
MITI_BEHAVIOR_LABELS = {
    "MITI_B1_GI", "MITI_B2_Persuade", "MITI_B3_Q", "MITI_B4_SR",
    "MITI_B5_CR", "MITI_B6_AF", "MITI_B7_Seek",
}


def test_usage_accounting():
    acc = C.empty_usage()
    assert acc == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    C.add_usage(acc, usage, "gpt-4o-mini")
    assert acc == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20,
                   "cost_usd": COST_PER_MINI_CALL}
    # unknown model falls back to gpt-4o-mini prices; None usage is a no-op
    C.add_usage(acc, usage, "some-unknown-model")
    assert acc["calls"] == 2
    assert acc["cost_usd"] == pytest.approx(2 * COST_PER_MINI_CALL)
    C.add_usage(acc, None, "gpt-4o-mini")
    assert acc["calls"] == 2
    # cost is rounded to 6 decimals on every accumulation (tiny costs vanish)
    tiny = C.empty_usage()
    C.add_usage(tiny, SimpleNamespace(prompt_tokens=1, completion_tokens=0), "gpt-4o-mini")
    assert tiny["cost_usd"] == 0.0


def test_resolve_params_and_seed_kwargs():
    assert C.resolve_params(None) == C.DEFAULT_PARAMS
    params = C.resolve_params({"therapist_temperature": 0.1, "seed": 7, "judge_model": None})
    assert params["therapist_temperature"] == 0.1
    assert params["judge_model"] == C.DEFAULT_PARAMS["judge_model"]  # None never overrides
    assert C.seed_kwargs(params) == {"seed": 7}
    assert C.seed_kwargs(C.resolve_params(None)) == {}


def test_transcript_format():
    messages = [
        {"role": "system", "content": "sys prompt is skipped"},
        {"role": "assistant", "content": " Hello there. "},
        {"role": "user", "content": "Hi.\n"},
    ]
    assert T.transcript(messages) == "[THERAPIST]: Hello there.\n\n[PATIENT]: Hi."


def test_clean_reply_cuts_malformed_chatml():
    assert T.clean_reply("Sure. <|im_end|> junk") == "Sure."
    assert T.clean_reply("Sure. <|im_end") == "Sure."  # malformed marker
    assert T.clean_reply("  plain reply \n") == "plain reply"


def test_patient_messages_role_flip():
    messages = [
        {"role": "system", "content": "therapist system prompt (dropped)"},
        {"role": "assistant", "content": "T1"},
        {"role": "user", "content": "P1"},
    ]
    assert G._patient_messages(messages, "persona") == [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "T1"},        # therapist lines become user
        {"role": "assistant", "content": "P1"},   # patient lines become assistant
    ]


def test_build_patient_persona_default_and_all_options():
    default = T.default_patient_persona()
    assert "Emma" in default
    assert "61 years old female" in default
    assert "smoking" in default.lower()
    prompts = set()
    for combo in itertools.product(*T.PERSONA_OPTIONS.values()):
        prompt = T.build_patient_persona(*combo)
        assert 'write "SESSION ENDED"' in prompt
        prompts.add(prompt)
    assert len(prompts) == 96  # every thesis permutation yields a distinct prompt


def test_judge_named_thesis_instruments(fake_clients):
    conv = "[PATIENT]: Hi.\n\n[THERAPIST]: Hello."
    usage = C.empty_usage()

    q1 = G._judge_named("Q1", conv, None, rationale=False, usage=usage)
    assert q1 == {"mean": 1.0, "scores": {f"Q1_{i}": 1 for i in range(1, 6)}}
    assert usage["calls"] == 1 and usage["cost_usd"] == COST_PER_MINI_CALL
    call = fake_clients.openai.calls[-1]
    assert call["temperature"] == 0  # judge determinism
    assert call["model"] == "gpt-4o-mini"
    assert call["response_format"]["json_schema"]["strict"] is True

    miti = G._judge_named("MITI", conv, None, rationale=False, usage=usage)
    assert miti["mean"] == 1.0
    assert set(miti["globals"]) == MITI_GLOBAL_LABELS
    assert set(miti["behaviors"]) == MITI_BEHAVIOR_LABELS
    assert set(miti["scores"]) == MITI_GLOBAL_LABELS | MITI_BEHAVIOR_LABELS

    rationale = G._judge_named("Q1", conv, None, rationale=True, usage=usage)
    assert rationale["rationale"] == "stub"
    # params override reaches the judge call
    G._judge_named("Q1", conv, {"judge_model": "gpt-4o", "seed": 5}, usage=usage)
    call = fake_clients.openai.calls[-1]
    assert call["model"] == "gpt-4o" and call["seed"] == 5


def test_custom_questionnaire_crud_and_judging(custom_file):
    for bad_args in [("", ["s"]), ("x" * 61, ["s"]), ("Q1", ["s"]), ("Empty", []),
                     ("TooMany", ["s"] * 21)]:
        with pytest.raises(ValueError):
            G.add_custom_questionnaire(*bad_args)

    G.add_custom_questionnaire(" LISTEN-2 ", [" statement one ", "", "statement two"], " desc ")
    saved = G.CUSTOM_QUESTIONNAIRES["LISTEN-2"]
    assert saved == {"description": "desc", "items": ["statement one", "statement two"]}
    raw = custom_file.read_text()
    assert raw.endswith("\n") and json.loads(raw) == {"LISTEN-2": saved}
    assert "LISTEN-2" in G.known_questionnaires()
    assert G.questionnaire_blurbs()["LISTEN-2"] == "custom, 2 items: desc"

    out = G._judge_named("LISTEN-2", "[PATIENT]: Hi.", None, usage=C.empty_usage())
    assert out == {"mean": 1.0, "scores": {"statement one": 1, "statement two": 1}}
    assert G._known(["Q1", "LISTEN-2", "Ghost"]) == ["Q1", "LISTEN-2"]

    G.delete_custom_questionnaire("LISTEN-2")
    assert json.loads(custom_file.read_text()) == {}
    with pytest.raises(KeyError):
        G.delete_custom_questionnaire("LISTEN-2")


def test_run_demo_loop_and_report(fake_clients):
    state = G.run_demo("mi-coach-pto-iter10", max_patient_turns=2)
    roles = [m["role"] for m in state["messages"]]
    assert roles == ["system", "assistant", "user", "assistant", "user", "assistant"]
    assert len(state["turn_scores"]) == 2
    assert state["turn_scores"][0] == {
        "therapist_turns": 2,  # greeting + first reply
        "means": {"Q1": 1.0},
        "results": {"Q1": {"mean": 1.0, "scores": {f"Q1_{i}": 1 for i in range(1, 6)}}},
    }
    report = state["report"]
    assert set(report["means"]) == {"Q2", "MITI"}
    assert report["per_turn_means"] == [{"Q1": 1.0}, {"Q1": 1.0}]
    assessment = report["assessment"]
    assert set(assessment) == {"overall_rating", "summary", "strengths", "growth_areas", "tip"}
    assert assessment["overall_rating"] == 1
    # OpenAI calls: 2 patient + 2 turn judges + 2 report judges + 1 assessment = 7;
    # the 2 therapist calls hit vLLM and are never billed.
    assert state["openai_usage"]["calls"] == 7
    assert state["openai_usage"]["cost_usd"] == pytest.approx(7 * COST_PER_MINI_CALL)
    assert report["usage"] == state["openai_usage"]
    assert fake_clients.vllm.n_calls == 2


def test_run_demo_stops_on_session_ended(fake_clients):
    fake_clients.openai.queue("I feel much better now. SESSION ENDED 2")
    state = G.run_demo("mi-coach-pto-iter10", max_patient_turns=5)
    assert state["session_ended"] is True
    assert sum(1 for m in state["messages"] if m["role"] == "user") == 1
    # 1 patient + 1 turn judge + 2 report judges + 1 assessment
    assert state["openai_usage"]["calls"] == 5


def test_stream_generators(fake_clients):
    fake_clients.vllm.queue("Hello patient! <|im_end|>trailing junk")
    chunks = list(G.stream_therapist([{"role": "user", "content": "hi"}], "m", None))
    assert chunks[0] == "Hello pa"          # 8-char fake chunks
    assert chunks[-1] == "Hello patient!"   # marker cleaned even mid-stream

    usage = C.empty_usage()
    fake_clients.openai.queue("I am okay.")
    chunks = list(G.stream_patient([{"role": "assistant", "content": "hi"}], "persona",
                                   None, usage))
    assert chunks[-1] == "I am okay."
    assert usage == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20,
                     "cost_usd": COST_PER_MINI_CALL}
