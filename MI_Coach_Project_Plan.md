# MI Coach — flagship side project plan
*Created 2026-07-13. Budget: a few hours/week — every phase must be shippable on its own.*

**Status 2026-07-15: all four phases complete — tagged v1.0.** Shipped beyond plan:
role toggle (judges score the human therapist), A/B compare with per-side reports,
multi-persona eval sweep, judge rationales, streaming replies, session history/export,
per-session cost tracking, advanced generation controls.

## Pitch (future CV bullet / README opener)
Training-and-practice tool for Motivational Interviewing: a fine-tuned Llama-3.2-1B therapist
(from the thesis) served with vLLM, orchestrated by a LangGraph agent that runs practice
sessions and scores them live with an LLM-as-a-Judge panel using validated MI questionnaires.

Framing note: it's a *practice tool for MI skills*, not therapy for end users — say so in the
README (disclaimer) and on the CV.

## Why this project
Closes the exact CV gaps found in the 2026 posting scan: vLLM / inference optimization,
agent frameworks (LangGraph), Docker/serving, end-to-end shipping. Reuses thesis assets
(adapters, patient prompts V3, questionnaires, judge prompts) so new work is mostly the
engineering layer — the missing signal.

## Constraints
- Thesis first; ~3-5 h/week here.
- vLLM is Linux-first → run under WSL2 on the RTX 5070 Ti (12 GB), or Colab for benchmarks.
- OpenAI budget is binding (~$300 spent on thesis): judge = gpt-4o-mini, few calls per
  session (pennies); patient sim optional / local.

## Phases (each ends in a pushed, README-ed state)

### Phase 1 — Serve + benchmark (~2 weeks) → CV: "vLLM, inference optimization"
- New public repo `MI-Coach`. Copy one LoRA adapter (best PTO LA0 iter) + base model ref.
- Serve Llama-3.2-1B + adapter via vLLM (OpenAI-compatible endpoint) under WSL2.
- Benchmark: vLLM vs HF Transformers vs quantized — tokens/s, p50/p95 latency, VRAM.
- README with the results table. **This alone updates the CV.**

### Phase 2 — API + UI + Docker (~1-2 weeks) → CV: "Docker, FastAPI"
- FastAPI wrapper with session state; simple Gradio chat UI (user plays the patient).
- Dockerfile (+ compose for server+UI). Short demo GIF in README.

### Phase 3 — LangGraph agent layer (~2-3 weeks) → CV: "LangGraph, agents"
- Graph: therapist node (local vLLM) → optional patient-sim node (gpt-4o-mini or local) →
  judge node (gpt-4o-mini + thesis questionnaires) → session feedback report node.
- Two modes: interactive practice (human patient) and auto-demo (simulated patient).
- Live per-turn scoring panel in the UI.

### Phase 4 — Polish (~1 week) — DONE 2026-07-15
- Eval table (base vs PTO adapter, auto-demo mode, n sessions), demo video/GIF,
  architecture diagram in README, pin deps, tag v1.0.
- Added: per-iteration eval of all 20 checkpoints (independently recovered the thesis
  best picks PTO-10/GRPO-8) + 4-persona robustness sweep ($0.05 total judge cost).

## CV integration checkpoints
- After Phase 1: add vLLM + benchmark bullet to Projects (replace or push down diacritization).
- After Phase 3: promote MI Coach to top project; add LangGraph to Skills.
- Skills adds when earned: vLLM, FastAPI, Docker, LangGraph, Gradio.

## How Claude helps
Point a session at the `MI-Coach` repo folder — scaffold, write the benchmark harness,
Dockerfile, LangGraph graph, README. Thesis repo stays untouched; MI-Coach imports nothing
from it except copied prompt/questionnaire files (keep licenses/attribution consistent).
