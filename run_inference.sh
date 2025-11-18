#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <sd3|sd3.5> \"prompt text\" [additional sd3_infer.py args]" >&2
  exit 1
fi

MODEL_CHOICE="$1"
shift
PROMPT_TEXT="$1"
shift || true

MODEL_DIR=${MODEL_DIR:-models}
MODEL_FOLDER=${MODEL_FOLDER:-$MODEL_DIR}
VAE_PATH=${VAE_PATH:-$MODEL_DIR/sd3_vae.safetensors}

case "$MODEL_CHOICE" in
  sd3)
    MODEL_PATH=${SD3_MODEL_PATH:-$MODEL_DIR/sd3_medium.safetensors}
    SHIFT_VALUE=${SD3_SHIFT:-1.0}
    ;;
  sd3.5)
    MODEL_PATH=${SD35_MODEL_PATH:-$MODEL_DIR/sd3.5_large.safetensors}
    SHIFT_VALUE=${SD35_SHIFT:-3.0}
    ;;
  *)
    echo "Unknown model choice '$MODEL_CHOICE'. Use 'sd3' or 'sd3.5'." >&2
    exit 1
    ;;
endcase

python sd3_infer.py \
  --prompt "$PROMPT_TEXT" \
  --model "$MODEL_PATH" \
  --vae "$VAE_PATH" \
  --shift "$SHIFT_VALUE" \
  --model_folder "$MODEL_FOLDER" \
  "$@"
