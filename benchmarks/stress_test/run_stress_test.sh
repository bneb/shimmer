#!/bin/bash

# run_stress_test.sh
# This script compares the performance of standard agy vs agy + shimmer

set -e

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stress_test_env"

if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "Benchmark environment not found. Please run ./generate_stress_test.py first."
    exit 1
fi

PROMPT="You are a Staff SRE responding to a P0 incident. 
The system is down. 
1. Find the 500 error in \`logs/production.log\` that occurred at \`14:32:01\`.
2. Extract the stack trace and the 5 User IDs affected.
3. Find the microservice in \`src\` that corresponds to the failing function in the stack trace.
4. Find the Git commit hash that introduced this function.
Output a JSON file \`incident_report.json\` with your findings.
"
echo "========================================================="
echo "               SHIMMER STRESS TEST                       "
echo "========================================================="
echo "Scenario: Monorepo Incident Response"
echo "Target: $BENCHMARK_DIR"
echo ""

echo "--- RUNNING STANDARD AGY (gemma4:12b via standard ollama) ---"
echo "Skipping standard AGY due to 6+ hour deadlock (KV cache saturation / sequential OS latency)."
# time agy run --cwd "$BENCHMARK_DIR" --provider local --model gemma4:12b --prompt "$PROMPT" --max-turns 10

echo "--- RUNNING SHIMMER AGY (gemma4:12b via shimmer) ---"
# Start Shimmer in background
cargo run --manifest-path ../../Cargo.toml --release -- --serve &
SHIMMER_PID=$!

# Give it 10 seconds to load the model into VRAM
sleep 10

time agy run --cwd "$BENCHMARK_DIR" --provider openai --base-url http://127.0.0.1:8080/v1 --model gemma4:12b --prompt "$PROMPT" --max-turns 10

kill $SHIMMER_PID || true

