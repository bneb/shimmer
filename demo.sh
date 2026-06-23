#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BINARY="target/release/shimmer"
MODEL="models/gemma4-12b.gguf"

if [ ! -f "$BINARY" ]; then
    echo "Building shimmer..."
    cargo build --release
fi

if [ ! -f "$MODEL" ]; then
    echo "No model found at $MODEL."
    echo "Download a GGUF model (e.g. Gemma 4 12B Coding Q4_K_M) and place it at $MODEL,"
    echo "or pass a custom path with --main-model."
    exit 1
fi

echo "=== Shimmer Demo ==="
"$BINARY" --no-tools --sample "temp=0.0,topk=0,repp=1.0" --main-model "$MODEL" \
    --prompt 'Say "Hello, Shimmer!" and nothing else.'
echo ""
echo "Demo complete."
