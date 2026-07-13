"""Benchmark: vLLM (OpenAI-compatible server) vs plain HF Transformers.

Measures tokens/s, p50/p95 end-to-end latency, and peak VRAM for the same set of
MI-style practice prompts, then prints a markdown results table (paste into README).

Usage:
    # 1. start the server:            bash scripts/serve.sh
    # 2. in another terminal:         .venv/bin/python bench/run_bench.py
    # HF-only or vLLM-only:           ... run_bench.py --skip-hf / --skip-vllm

The vLLM side is measured twice: sequential (single client, comparable to HF) and
concurrent (8 parallel requests, where continuous batching pays off).
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B")
# HF baseline uses one adapter; default matches the thesis default (PTO iter 10).
ADAPTER = os.environ.get("ADAPTER", str(Path(__file__).resolve().parents[1] / "assets" / "adapters" / "pto-iter10"))
BASE_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
MAX_NEW_TOKENS = 256
CONCURRENCY = 8

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SYSTEM_PROMPT = (ASSETS / "therapist_system_prompt.txt").read_text().strip()
GREETING = (
    "Hello, welcome to your first motivational session with me. My name is David and "
    "I`m a professional motivational counselor. Can you start by telling me a little "
    "bit about yourself and why are you here?"
)
# The adapters' ChatML markers are plain text (not special tokens), so generation
# must stop on them explicitly.
STOP = ["<|im_end|>", "<|im_start|>"]


def build_messages(client_utterance: str) -> list[dict]:
    """Session format the thesis adapters were trained on: therapist system prompt,
    the counselor's standard greeting, then the client's (patient's) turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
        {"role": "user", "content": client_utterance},
    ]

# Client-role openers a trainee therapist would respond to — the served model plays
# the therapist, so these exercise realistic MI turns of varying length.
PROMPTS = [
    "I know I should cut back on drinking, but honestly it's the only thing that helps me unwind after work.",
    "My doctor keeps nagging me about exercise. I've tried gyms three times and quit every time, so what's the point?",
    "Everyone says I spend too much time gaming. Maybe they're right, but it's where all my friends are.",
    "I want to stop smoking for my kids, I really do. I've just failed so many times that I don't trust myself anymore.",
    "Part of me wants to go back to school and finish my degree, but I'm 41 and it feels ridiculous to start over.",
    "My partner thinks I have a problem with online shopping. I think I just like nice things. We fight about it constantly.",
    "I've been putting off the diabetes diet my dietitian gave me. The food is bland and my family won't eat it anyway.",
    "I keep saying I'll look for a less stressful job, and then another year goes by and I'm still here, exhausted.",
]


def gpu_mem_mb() -> int:
    """Current GPU memory usage in MiB via nvidia-smi (0 if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def summarize(name: str, latencies: list[float], total_tokens: int, wall: float, vram: int) -> dict:
    lat = sorted(latencies)
    return {
        "config": name,
        "tokens_per_s": round(total_tokens / wall, 1),
        "p50_s": round(statistics.median(lat), 2),
        "p95_s": round(lat[max(0, int(len(lat) * 0.95) - 1)], 2),
        "peak_vram_mb": vram,
        "n_requests": len(latencies),
    }


# --------------------------------------------------------------------------- vLLM

def bench_vllm(model_name: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key="unused")

    def one_request(prompt: str) -> tuple[float, int]:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model_name,
            messages=build_messages(prompt),
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            stop=STOP,
        )
        return time.perf_counter() - t0, resp.usage.completion_tokens

    one_request(PROMPTS[0])  # warmup

    results = []
    # Sequential — apples-to-apples with the HF loop below.
    t0, lats, toks = time.perf_counter(), [], 0
    for p in PROMPTS:
        dt, n = one_request(p)
        lats.append(dt)
        toks += n
    results.append(summarize(f"vLLM sequential ({model_name})", lats, toks, time.perf_counter() - t0, gpu_mem_mb()))

    # Concurrent — where continuous batching shines.
    t0, lats, toks = time.perf_counter(), [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        for dt, n in pool.map(one_request, PROMPTS * 2):
            lats.append(dt)
            toks += n
    results.append(summarize(f"vLLM concurrent x{CONCURRENCY} ({model_name})", lats, toks, time.perf_counter() - t0, gpu_mem_mb()))
    return results


# ----------------------------------------------------------------------------- HF

def bench_hf() -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    name = "HF Transformers (base)"
    if Path(ADAPTER, "adapter_config.json").is_file():
        from peft import PeftModel
        # Adapter dir ships the tokenizer + the ChatML template it was trained with.
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
        model = PeftModel.from_pretrained(model, ADAPTER)
        name = "HF Transformers (+LoRA)"
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()

    def one_request(prompt: str) -> tuple[float, int]:
        enc = tokenizer.apply_chat_template(
            build_messages(prompt), add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=STOP, tokenizer=tokenizer,
            )
        return time.perf_counter() - t0, out.shape[1] - enc["input_ids"].shape[1]

    one_request(PROMPTS[0])  # warmup
    torch.cuda.reset_peak_memory_stats()

    t0, lats, toks = time.perf_counter(), [], 0
    for p in PROMPTS:
        dt, n = one_request(p)
        lats.append(dt)
        toks += n
    vram = int(torch.cuda.max_memory_allocated() / 2**20)
    return [summarize(name, lats, toks, time.perf_counter() - t0, vram)]


# --------------------------------------------------------------------------- main

def to_markdown(rows: list[dict]) -> str:
    header = "| Config | Tokens/s | p50 latency (s) | p95 latency (s) | Peak VRAM (MiB) | Requests |"
    sep = "|---|---|---|---|---|---|"
    body = [
        f"| {r['config']} | {r['tokens_per_s']} | {r['p50_s']} | {r['p95_s']} | {r['peak_vram_mb']} | {r['n_requests']} |"
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-vllm", action="store_true", help="skip the vLLM server benchmark")
    parser.add_argument("--skip-hf", action="store_true", help="skip the HF Transformers benchmark")
    args = parser.parse_args()

    rows = []
    if not args.skip_vllm:
        from openai import OpenAI
        served = [m.id for m in OpenAI(base_url=BASE_URL, api_key="unused").models.list().data]
        print(f"vLLM server models: {served}")
        for model_name in served:
            rows += bench_vllm(model_name)
    if not args.skip_hf:
        if not args.skip_vllm:
            print("NOTE: run HF with the vLLM server stopped for a clean VRAM reading (--skip-vllm).")
        rows += bench_hf()

    table = to_markdown(rows)
    print("\n" + table)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"bench-{stamp}.json").write_text(json.dumps(rows, indent=2))
    (out_dir / f"bench-{stamp}.md").write_text(table + "\n")
    print(f"\nSaved to {out_dir}/bench-{stamp}.{{json,md}}")


if __name__ == "__main__":
    main()
