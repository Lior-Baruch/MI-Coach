#!/usr/bin/env bash
# vLLM imports torchcodec at startup, which needs FFmpeg shared libraries.
# If you can, just: sudo apt install ffmpeg   (and skip this script).
#
# No-sudo fallback: the PyAV wheel bundles real FFmpeg .so files under mangled
# names; this script symlinks them under their standard sonames in
# .venv/ffmpeg-libs. serve.sh puts both dirs on LD_LIBRARY_PATH automatically.

set -euo pipefail
cd "$(dirname "$0")/.."

AV_LIBS=.venv/lib/python3.12/site-packages/av.libs
[[ -d "$AV_LIBS" ]] || { echo "PyAV not installed — run: uv pip install --python .venv/bin/python av" >&2; exit 1; }

mkdir -p .venv/ffmpeg-libs
cd .venv/ffmpeg-libs
for f in ../../"$AV_LIBS"/*.so.*; do
  base=$(basename "$f")
  soname=$(echo "$base" | sed -E 's/-[0-9a-f]{8}//; s/(\.so\.[0-9]+).*/\1/')
  ln -sf "../lib/python3.12/site-packages/av.libs/$base" "$soname"
done
echo "Created $(ls | wc -l) soname symlinks in .venv/ffmpeg-libs"
