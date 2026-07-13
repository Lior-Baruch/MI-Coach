#!/usr/bin/env bash
# No-sudo fallbacks for two system deps vLLM needs at runtime. If you have sudo,
# `sudo apt install ffmpeg build-essential` makes this script unnecessary.
#
# 1. FFmpeg shared libraries (dlopen'd by torchcodec, imported by vLLM at startup):
#    the PyAV wheel bundles real FFmpeg .so files under auditwheel-mangled names;
#    we symlink them under their standard sonames in .venv/ffmpeg-libs.
# 2. C compiler (Triton/Inductor JIT compiles small C launcher stubs):
#    the ziglang wheel ships `zig cc` (clang); we shim it as .venv/bin/zigcc,
#    rewriting the `-l:libcuda.so.1` syntax that zig's linker does not support.
#
# serve.sh detects both fallbacks automatically (LD_LIBRARY_PATH / CC).

set -euo pipefail
cd "$(dirname "$0")/.."

AV_LIBS=.venv/lib/python3.12/site-packages/av.libs
[[ -d "$AV_LIBS" ]] || { echo "PyAV not installed — run: uv pip install --python .venv/bin/python av" >&2; exit 1; }
.venv/bin/python -c "import ziglang" 2>/dev/null \
  || { echo "ziglang not installed — run: uv pip install --python .venv/bin/python ziglang" >&2; exit 1; }

mkdir -p .venv/ffmpeg-libs
(
  cd .venv/ffmpeg-libs
  for f in ../lib/python3.12/site-packages/av.libs/*.so.*; do
    base=$(basename "$f")
    soname=$(echo "$base" | sed -E 's/-[0-9a-f]{8}//; s/(\.so\.[0-9]+).*/\1/')
    ln -sf "../lib/python3.12/site-packages/av.libs/$base" "$soname"
  done
)
echo "ffmpeg-libs: $(ls .venv/ffmpeg-libs | wc -l) soname symlinks"

cat > .venv/bin/zigcc <<'EOF'
#!/bin/sh
# zig cc shim for Triton/Inductor. zig's linker rejects "-l:libNAME.so.N";
# rewrite it to "-lNAME" (the plain .so is on the same -L path).
for a in "$@"; do
  shift
  case "$a" in
    -l:lib*.so*) n=${a#-l:lib}; a="-l${n%%.so*}" ;;
  esac
  set -- "$@" "$a"
done
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$DIR/python" -m ziglang cc "$@"
EOF
chmod +x .venv/bin/zigcc
echo "zigcc shim: .venv/bin/zigcc ($(.venv/bin/zigcc --version 2>&1 | head -1))"
