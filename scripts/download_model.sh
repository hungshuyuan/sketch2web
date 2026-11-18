#!/bin/bash
set -e
DEST="${1:-$HOME/models/qwen38}"
mkdir -p "$DEST"
if ! command -v hf >/dev/null 2>&1; then
  python3 -m venv "$HOME/.hf-venv"
  "$HOME/.hf-venv/bin/pip" install -U huggingface_hub
  HF="$HOME/.hf-venv/bin/hf"
else
  HF=hf
fi
"$HF" download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include "*UD-IQ4_XS*" --include "mmproj-F16.gguf" --local-dir "$DEST"
ls -lh "$DEST" "$DEST/UD-IQ4_XS"
