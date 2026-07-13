# assets/

Files copied from the thesis repo (`Thesis_PTO_GRPO`, read-only reference) — each copied
file carries a header comment noting its origin. Nothing here is imported from thesis paths
at runtime.

## adapter/ (not committed)

The best thesis LoRA adapter (PTO, LA0, best iteration) for `meta-llama/Llama-3.2-1B-Instruct`.
Adapter weights are gitignored; to set up locally, copy the adapter directory (containing
`adapter_config.json` + `adapter_model.safetensors`) here:

```bash
# Thesis adapters live on Google Drive, symlinked as G: on Windows. In WSL:
sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g
cp -r "/mnt/g/My Drive/Thesis_PTO_GRPO/Exp3_PTO_GRPO/data/pto_Exp3/<best-adapter-dir>" assets/adapter
```
