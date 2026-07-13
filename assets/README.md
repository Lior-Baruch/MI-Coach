# assets/

Thesis artifacts (from `Thesis_PTO_GRPO`, read-only reference). **Nothing in this
directory is committed except this README** — the raw thesis trees include training
data that must never reach the public repo.

## Layout

| Path | What it is |
|---|---|
| `adapters/pto-iter10/` | **serving copy** — best PTO adapter (LA0, iteration 10), served as `mi-coach-pto-iter10` (default) |
| `adapters/grpo-iter8/` | **serving copy** — best GRPO adapter (iteration 8), served as `mi-coach-grpo-iter8` |
| `PTO/…_PTgreedy/` | raw thesis run (all iterations) — local reference only |
| `GRPO/…_G8/` | raw thesis run (all iterations) — local reference only |

The serving copies were taken from `<run>/iteration_<n>/adapter/` and contain
`adapter_config.json`, `adapter_model.safetensors`, the tokenizer, and the ChatML
`chat_template.jinja` the adapters were trained with (`scripts/serve.sh` passes it
to vLLM automatically). The GRPO `ref/` sub-adapter (training-time reference policy)
is not needed for inference and was not copied.

Note: adapters were trained on **`meta-llama/Llama-3.2-1B`** (base, not Instruct),
with roles patient=`user` / therapist=`assistant` and the expert-therapist system
prompt in `therapist_system_prompt.txt` (copied from the thesis
`Exp3_PTO_GRPO/code/system_prompts_builder.py`, CounselorPersonality "Good", name
David). The ChatML markers (`<|im_start|>`, `<|im_end|>`) are plain text, not special
tokens — always pass them as `stop` strings when generating.
