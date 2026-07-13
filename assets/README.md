# assets/

Files copied from the thesis repo (`Thesis_PTO_GRPO`, read-only reference) — each copied
file carries a header comment noting its origin. Nothing here is imported from thesis paths
at runtime.

## adapters/ (not committed)

Thesis LoRA adapters for `meta-llama/Llama-3.2-1B-Instruct`. Weights are gitignored.
`scripts/serve.sh` serves every subdirectory here that contains an `adapter_config.json`
as model `mi-coach-<subdir>`:

| Directory | Served as | Notes |
|---|---|---|
| `adapters/pto-iter10/` | `mi-coach-pto-iter10` | **default** — best PTO adapter (LA0, iteration 10) |
| `adapters/grpo-iter8/` | `mi-coach-grpo-iter8` | optional — best GRPO adapter (iteration 8) |

To set up locally (thesis adapters live on Google Drive, symlinked as G: on Windows):

```bash
sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g
THESIS_DATA="/mnt/g/My Drive/Thesis_PTO_GRPO/Exp3_PTO_GRPO/data"
mkdir -p assets/adapters
cp -r "$THESIS_DATA/pto_Exp3/<pto-iter10-adapter-dir>"  assets/adapters/pto-iter10
cp -r "$THESIS_DATA/grpo_Exp3/<grpo-iter8-adapter-dir>" assets/adapters/grpo-iter8
```

Each copied directory should contain `adapter_config.json` + `adapter_model.safetensors`.
