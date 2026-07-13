# CLAUDE.md — MI Coach

## What this is
Employer-facing side project of Lior Baruch (M.Sc. thesis: PTO vs GRPO post-training of
small therapist LLMs for Motivational Interviewing, ICLR 2025 workshop paper). MI Coach
deploys the thesis model as a real service: a practice tool for MI skills — **not therapy
for end users** (keep this disclaimer in the README).

Full phase plan: `MI_Coach_Project_Plan.md` (repo root). Read it before any work.

## Current phase: 2 — API + UI + Docker
Phase 1 (serve + benchmark) is done and pushed: vLLM serves both thesis adapters
(mi-coach-pto-iter10 default, mi-coach-grpo-iter8) with results table in README.
Phase 2 goal: FastAPI wrapper with session state; simple Gradio chat UI where the user
plays the patient; Dockerfile (+ compose for server+UI); short demo GIF in README.
Still zero OpenAI calls — no agents/judging until Phase 3.

Key serving facts (learned in Phase 1): adapters trained on BASE Llama-3.2-1B;
requests need assets/therapist_system_prompt.txt as system message, roles
patient=user / therapist=assistant, and stop strings ["<|im_end|>", "<|im_start|>"]
(ChatML markers are plain text, not special tokens).

## Environment
- WSL2 Ubuntu on Windows, RTX 5070 Ti (12 GB VRAM), NVIDIA driver via Windows (verify `nvidia-smi`).
- Fresh venv per this repo; do NOT reuse the thesis env. Python 3.11+ recommended for vLLM.
- 12 GB VRAM fits Llama-3.2-1B easily; leave headroom for KV cache (`--gpu-memory-utilization 0.85`).

## Thesis assets (copy in, never import from thesis paths)
Thesis repo (Windows side): `/mnt/c/Users/baruc/Desktop/Projects/Thesis_PTO_GRPO/`
- LoRA adapters: under `Exp3_PTO_GRPO/data/pto_Exp3/` (symlinked dir — ask Lior for the
  exact best-adapter path, likely PTO LA0 iteration 10).
- Questionnaires / judge rubrics: `Exp3_PTO_GRPO/code/questionnaires.py`
- Patient prompts: `Exp3_PTO_GRPO/code/system_prompts_builder.py`
Copy files into `assets/` with a header comment noting origin. Phase 1 only needs the adapter.

## Hard rules
- Public repo: NO API keys, key files, student data, or thesis conversation data. Keys via
  env vars only; `.gitignore` from day one.
- The thesis repo is read-only reference — never modify it from here.
- OpenAI budget is tight: Phase 1 needs zero OpenAI calls. Later phases use gpt-4o-mini sparingly.
- Each phase ends pushed + README-ed (CV checkpoint). Prefer small, reviewable commits.

## Definition of done (Phase 1)
`bash scripts/serve.sh` starts the vLLM server; `python bench/run_bench.py` produces the
comparison table; README shows setup, one curl example, and the benchmark results.
