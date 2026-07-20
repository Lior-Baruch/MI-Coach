"""Bridge to the copied thesis assets (``assets/thesis/``, ``assets/*.txt``).

Loads the thesis modules by file path (no package install needed) and owns
every constant that exists because of how the thesis adapters were trained:
the expert-therapist system prompt, David's greeting, the plain-text ChatML
stop markers, the ``[PATIENT]``/``[THERAPIST]`` transcript format the thesis
judges were built for, and the patient-persona permutation builder.

The thesis files themselves are read-only copies — extensions wrap them here
(and in ``agent.judging``) rather than editing them, so they stay diffable
against the thesis originals.
"""

import importlib.util
import re

from agent.config import REPO


def _load_thesis_module(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "assets" / "thesis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


questionnaires = _load_thesis_module("questionnaires")
prompts_builder = _load_thesis_module("system_prompts_builder")

THERAPIST_SYSTEM_PROMPT = (REPO / "assets" / "therapist_system_prompt.txt").read_text().strip()
GREETING = (
    "Hello, welcome to your first motivational session with me. My name is David and "
    "I`m a professional motivational counselor. Can you start by telling me a little "
    "bit about yourself and why are you here?"
)
STOP = ["<|im_end|>", "<|im_start|>"]  # ChatML markers are plain text for the adapters


def clean_reply(text: str) -> str:
    """Cut malformed ChatML markers (e.g. "<|im_end>") that slip past vLLM's
    exact-match stop strings."""
    return re.split(r"<\|im_", text)[0].strip()


def transcript(messages: list[dict]) -> str:
    """[PATIENT]/[THERAPIST] transcript, the format the thesis judges were built for."""
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"[PATIENT]: {m['content'].strip()}")
        elif m["role"] == "assistant":
            lines.append(f"[THERAPIST]: {m['content'].strip()}")
    return "\n\n".join(lines)


def initial_messages() -> list[dict]:
    """The opening every patient-role session was trained on: the expert
    system prompt plus David's greeting."""
    return [
        {"role": "system", "content": THERAPIST_SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
    ]


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
