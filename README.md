# MI Coach

A training-and-practice tool for **Motivational Interviewing (MI)** skills: a fine-tuned
Llama-3.2-1B therapist model (from my M.Sc. thesis on PTO vs GRPO post-training, ICLR 2025
workshop paper) served as a real, benchmarked service.

> **Disclaimer:** MI Coach is a *practice tool for MI skills* aimed at trainees and
> researchers. It is **not therapy** and must not be used as a substitute for professional
> mental-health care.

## Status — Phase 1: Serve + benchmark

Llama-3.2-1B + thesis LoRA adapter served via [vLLM](https://github.com/vllm-project/vllm)
(OpenAI-compatible endpoint), benchmarked against plain HF Transformers.
Later phases add a FastAPI/Gradio app, Docker, and a LangGraph agent layer with an
LLM-as-a-Judge scoring panel (see `MI_Coach_Project_Plan.md`).

## Setup

Requirements: Linux or WSL2, NVIDIA GPU (developed on an RTX 5070 Ti, 12 GB), Python 3.12,
[uv](https://docs.astral.sh/uv/), and a Hugging Face token with access to the gated
`meta-llama/Llama-3.2-1B-Instruct`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
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
exposed as its own model — `mi-coach-pto-iter10` (thesis default) and
`mi-coach-grpo-iter8` — so you choose the adapter per request via the `model` field;
the base model stays available under its Hugging Face id.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mi-coach-pto-iter10",
    "messages": [{"role": "user", "content": "I know I should cut back on drinking, but it is the only thing that helps me unwind."}],
    "max_tokens": 128
  }'
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

*(pending — table produced by `bench/run_bench.py` goes here)*

| Config | Tokens/s | p50 latency (s) | p95 latency (s) | Peak VRAM (MiB) | Requests |
|---|---|---|---|---|---|
| vLLM sequential | – | – | – | – | – |
| vLLM concurrent x8 | – | – | – | – | – |
| HF Transformers | – | – | – | – | – |
