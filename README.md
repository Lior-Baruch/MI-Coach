# MI Coach

A training-and-practice tool for **Motivational Interviewing (MI)** skills: a fine-tuned
Llama-3.2-1B therapist model (from my M.Sc. thesis on PTO vs GRPO post-training, ICLR 2025
workshop paper) served as a real, benchmarked service.

> **Disclaimer:** MI Coach is a *practice tool for MI skills* aimed at trainees and
> researchers. It is **not therapy** and must not be used as a substitute for professional
> mental-health care.

## Status — v1.0 (all phases complete)

- **Phase 1:** Llama-3.2-1B + thesis LoRA adapters served via
  [vLLM](https://github.com/vllm-project/vllm) (OpenAI-compatible endpoint), benchmarked
  against plain HF Transformers (results below).
- **Phase 2:** FastAPI session API + Gradio practice UI (you play the patient),
  Dockerfile + compose.
- **Phase 3:** [LangGraph](https://github.com/langchain-ai/langgraph) agent —
  every therapist turn is scored live by an LLM-as-a-Judge (gpt-4o-mini + the thesis
  questionnaires), sessions end with an MI feedback report, and an auto-demo mode runs
  a simulated patient.
- **Phase 4:** training-iteration + multi-persona evaluations in the deployed system,
  streaming replies, A/B compare with per-side reports, judge rationales, session
  history/export, per-session cost tracking, advanced generation controls.

![MI Coach practice session demo](docs/demo.gif)

## Agent layer

```mermaid
graph LR
    P[patient turn<br/>human or gpt-4o-mini sim] --> T[therapist node<br/>local vLLM + LoRA]
    T --> J[judge node<br/>gpt-4o-mini + thesis Q1]
    J -->|auto-demo loop| P
    J -->|session end| R[report node<br/>Q2 + MITI + supervisor narrative]
```

Judging uses the thesis evaluation stack (`assets/thesis/questionnaires.py`) with
**selectable instruments** — Q1, Q2, WAI-SR, CSQ-8, MI-SAT, MITI, PCT, MICI — chosen
independently for the live per-turn judge (default Q1) and the end-of-session report
(default Q2 + MITI), all via OpenAI structured outputs against the thesis JSON
schemas — no free-text parsing. The judge can optionally attach a **one-sentence
rationale per instrument** (separate toggles for per-turn and report judging). The
simulated patient uses the thesis persona prompts
(`assets/thesis/system_prompts_builder.py`). All judge/patient calls are gpt-4o-mini
by default (selectable); one call per selected instrument per judged point, and every
session tracks its actual token usage and cost — a default scored demo session costs
about a cent.

## Training-iteration evaluation

Every checkpoint of both methods was evaluated in the deployed system with
`eval/run_eval.py`: 3 auto-demo sessions per checkpoint (fixed thesis patient persona),
scored per turn with Q1 and at session end with Q2 + MITI (gpt-4o-mini judge):

![Judge scores vs training iteration](docs/eval_scores.png)

Both post-training methods clearly improve over the base model, and the checkpoints
this evaluation ranks best — **PTO iteration 10** and **GRPO iteration 8** — are exactly
the checkpoints the thesis evaluation selected:

| Checkpoint | Q1 (per-turn) | Q2 (17 items) | MITI globals |
|---|---|---|---|
| base Llama-3.2-1B | 2.58 ± 0.23 | 3.49 ± 0.78 | 3.08 ± 0.38 |
| PTO iter 10 (best) | 3.49 ± 0.30 | **4.43 ± 0.24** | **4.25 ± 0.43** |
| GRPO iter 8 (best) | **3.78 ± 0.14** | 4.29 ± 0.00 | **4.33 ± 0.29** |

Full per-iteration table in `eval/results/`. Honest caveats: the judge is an LLM,
sessions are short (3 patient turns) and sampled, and n=3 per checkpoint — error bars
are wide. This reproduces the *shape* of the thesis evaluation in the deployed system;
it is not a re-run of the thesis experiments.

### Robustness across patient personas

The best checkpoints were also swept over four thesis patient personas
(`eval/run_eval.py --personas all`): different problem (smoking/obesity), age, history,
and cooperation level — including a young resistant smoker who never tried quitting:

![Judge scores by patient persona](docs/eval_personas.png)

Both adapters beat the base model on **every persona and metric**; the hardest persona
for everyone is the resistant patient, the easiest the eager one — the expected
ordering. Full table in `eval/results/eval-personas-latest.md`; the entire 24-session
sweep cost **$0.05** in gpt-4o-mini calls (per-session cost is tracked end-to-end).

## Setup

Requirements: Linux or WSL2, NVIDIA GPU (developed on an RTX 5070 Ti, 12 GB), Python 3.12,
[uv](https://docs.astral.sh/uv/), and a Hugging Face token with access to the gated
`meta-llama/Llama-3.2-1B-Instruct`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://flashinfer.ai/whl/cu130   # flashinfer-jit-cache (prebuilt kernels, no nvcc needed)
export HF_TOKEN=...   # keys via env vars only — never committed
```

vLLM additionally needs FFmpeg shared libraries (via `torchcodec`) and a C compiler
(Triton/Inductor JIT). Either `sudo apt install ffmpeg build-essential`, or — no sudo
needed — run the bundled fallback once (FFmpeg libs from the PyAV wheel, `zig cc` from
the ziglang wheel; `serve.sh` picks both up automatically):

```bash
bash scripts/setup_local_toolchain.sh
```

Place the thesis LoRA adapters under `assets/adapters/` — `pto-iter10/` (default) and
optionally `grpo-iter8/`. Adapter weights are not committed; see `assets/README.md`.

## Serve

```bash
bash scripts/serve.sh
```

Starts vLLM on `http://localhost:8000/v1`. Each adapter under `assets/adapters/` is
exposed as its own model (`mi-coach-<method>-iter<N>`); run
`bash scripts/link_adapters.sh` once to link **every training iteration of both
methods** (PTO 1-10, GRPO 1-10 — thesis-best: **PTO iter 10**, **GRPO iter 8**), so you
can compare any checkpoint per request via the `model` field. The base model stays
available under its Hugging Face id. LoRA swapping is cheap: vLLM keeps 4 adapters on
GPU and LRU-caches the rest in RAM.

The adapters were trained on base `meta-llama/Llama-3.2-1B` with the therapist system
prompt in `assets/therapist_system_prompt.txt` and a ChatML template whose markers are
plain text — pass them as `stop` strings:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d @- <<'EOF'
{
  "model": "mi-coach-pto-iter10",
  "messages": [
    {"role": "system", "content": "You are a motivational interviewing counselor named David. You partner with the patient to understand his problems. You are empathetic towards him and help the patient explore their ambivalence regarding behavioral change. You are non-judgmental while encouraging the patient to change. In your answer, please avoid repetitions and unnecessary loops in the conversation. In your answer, please avoid repeating expressions of gratitude or similar sentiments multiple times if you've already expressed them during the conversation. You should only end the session when at least one of the following conditions is met. If you need to end the session, write \"SESSION ENDED\" followed by the condition number: 1. If you believe that you have provided the appropriate treatment to the patient and have nothing else to advise in the current session.2. When time is up."},
    {"role": "assistant", "content": "Hello, welcome to your first motivational session with me. My name is David and I`m a professional motivational counselor. Can you start by telling me a little bit about yourself and why are you here?"},
    {"role": "user", "content": "Hi David. I have smoked for years and I know I should stop, but it is the only thing that helps me unwind."}
  ],
  "max_tokens": 200,
  "stop": ["<|im_end|>", "<|im_start|>"]
}
EOF
```

(The system prompt is the thesis expert-therapist prompt, also in
`assets/therapist_system_prompt.txt`.)

## Practice app (FastAPI + Gradio)

With the vLLM server running:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

- **UI:** http://localhost:8080/ui —
  - **Play either role**: as the *patient* the fine-tuned model counsels you; as the
    *therapist* a simulated patient (thesis persona, configurable: problem, cooperation
    level, age, …) responds and **the judges score you** — that's the "coach" in MI Coach.
  - Pick the method (PTO/GRPO/base) and training iteration (thesis-best marked ★), and
    choose which questionnaires judge live each turn vs. in the final report (or none —
    per-turn judging can be switched off entirely). Replies **stream** token by token.
  - Live score timeline chart with optional **judge rationales**; *End session → report*
    for per-instrument scores plus an **overall assessment** (a reviewer model weighs
    the transcript *and* the judges' scores: rating /5, strengths, growth areas, one
    concrete tip). Each session shows its actual OpenAI token usage and cost.
  - **Compare (A/B) tab** with full practice parity: send the same patient message to
    two checkpoints, or auto-demo both against the same simulated persona — per-side
    score timelines, end-of-session reports, and a combined markdown export.
  - **History tab**: browse every session of the server run (practice, compare, demo)
    with transcript, scores, report, and export.
  - **Advanced settings** (apply everywhere): therapist/patient temperatures, max
    tokens, judge model, sampling seed, auto-demo length, rationale toggles.
  - Export any session as markdown (UI button or `GET /sessions/{id}/export`).
- **API:** `POST /sessions` (model, your role, persona, questionnaires, generation
  `params`, rationale flags) → `POST /sessions/{id}/message` (returns the reply,
  per-turn judge scores, cumulative usage/cost) → `POST /sessions/{id}/report`;
  `POST /demo` runs a full simulated session; `GET /sessions` lists them.
  OpenAPI docs at `/docs`; `GET /health`.
  Sessions are in-memory (practice tool, not a clinical record store).
- Set `OPENAI_API_KEY` (e.g. in `.env`) to enable judging; without it the app degrades
  to plain chat.

## Docker

The app has its own image; vLLM runs from the official `vllm/vllm-openai` image with
the adapters mounted read-only. Requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
MODEL_ID=meta-llama/Llama-3.2-1B HF_TOKEN=... docker compose up --build
# UI on http://localhost:8080/ui, raw vLLM endpoint on http://localhost:8000/v1
```

## Benchmark

```bash
# terminal 1
bash scripts/serve.sh
# terminal 2
.venv/bin/python bench/run_bench.py            # vLLM (sequential + concurrent x8)
# then stop the server and, for a clean VRAM reading:
.venv/bin/python bench/run_bench.py --skip-vllm  # HF Transformers baseline
```

### Results

Llama-3.2-1B (bf16) + thesis LoRA adapters, RTX 5070 Ti (12 GB), WSL2, 8 MI practice
prompts per config, 256 max new tokens, temperature 0.7 (2026-07-13):

| Config | Tokens/s | p50 latency (s) | p95 latency (s) | Peak VRAM (MiB) |
|---|---|---|---|---|
| HF Transformers + LoRA `pto-iter10` (sequential) | 46.0 | 3.70 | 5.12 | 2,450 |
| vLLM + LoRA `pto-iter10` (sequential) | 154.7 | 0.30 | 1.04 | 11,406¹ |
| vLLM + LoRA `pto-iter10` (concurrent ×8) | 657.8 | 0.37 | 1.30 | 11,406¹ |
| vLLM + LoRA `grpo-iter8` (sequential) | 147.6 | 0.79 | 1.69 | 11,406¹ |
| vLLM + LoRA `grpo-iter8` (concurrent ×8) | 911.1 | 0.89 | 1.79 | 11,406¹ |

**vLLM delivers ~3.4× single-stream throughput over HF Transformers, and up to ~20×
aggregate throughput with continuous batching** (911 vs 46 tokens/s).

¹ vLLM preallocates 85% of VRAM up front (weights + paged KV cache sized for ~58
concurrent 4k-token sequences); HF's number is the actual allocation for one sequence.
Latency columns are not directly comparable across engines — completions are sampled and
stop-string–terminated, so output lengths differ; tokens/s is the like-for-like metric.
Raw runs: `bench/results/` (regenerate with `bench/run_bench.py`).
