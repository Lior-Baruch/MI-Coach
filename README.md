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

vLLM needs FFmpeg shared libraries at startup (via `torchcodec`). Either
`sudo apt install ffmpeg`, or — no sudo needed — run the bundled fallback once:

```bash
bash scripts/setup_ffmpeg_libs.sh   # symlinks the FFmpeg libs shipped inside the PyAV wheel
```

Place the thesis LoRA adapter in `assets/adapter/` (adapter weights are not committed;
see `assets/README.md`).

## Serve

```bash
bash scripts/serve.sh
```

Starts vLLM on `http://localhost:8000/v1`. The LoRA adapter is exposed as model
`mi-coach`; the base model stays available under its Hugging Face id.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mi-coach",
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
