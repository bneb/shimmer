# Shimmer

Local LLM execution engine for agentic coding on Apple Silicon, built on
[llama.cpp](https://github.com/ggml-org/llama.cpp) via the Metal backend.

Shimmer runs inference directly against GGUF models and can execute read-only
shell tools (`rg`, `cat`, `fd`, `ls`, `git`) inline — capturing their output
and injecting it back into the KV cache without an external orchestrator.

## Quick Start

Requires macOS on Apple Silicon (M1+).

```bash
# Build from source
git clone https://github.com/bneb/shimmer.git
cd shimmer
cargo build --release

# Run a prompt
./target/release/shimmer \
  --main-model models/gemma4-12b.gguf \
  --prompt "Explain how Rust's borrow checker works."

# Run as an OpenAI-compatible HTTP server
./target/release/shimmer --serve --main-model models/gemma4-12b.gguf
```

## SWE-bench Evaluation

Three modes for evaluating on [SWE-bench Lite](https://www.swebench.com/):

- **Hybrid** — agentless context + up to 3 read-only tools + exact-match-only.
  The model can verify lines with `cat` before editing. Most honest results.
- **Agentless** — keyword-based file localization, single-shot generation
  without tools. Fastest, avoids tool-call loops.
- **Agentic** — full tool loop with investigation. Higher latency, loop risk.

Results (hybrid, seed=42, 3 instances): **1/3 correct fixes** (django-11099,
exact match, 137s, no tools needed).

```bash
pip install datasets
python3 benchmarks/agentic/generate_swe_bench.py \
  --hybrid --sample 3 --seed 42 \
  --model models/gemma4-12b.gguf
```

## Models

Shimmer works with any GGUF model. Tested configurations:

| Model | Quantization | Notes |
|-------|-------------|-------|
| Gemma 4 12B Coding | Q4_K_M | Primary target; ~7GB VRAM |
| Qwen 2.5 Coder 7B | Q4_K_M | Faster, lower edit accuracy |

All runs use greedy sampling (`temp=0.0`). Speculative decoding is disabled by
default — n-gram drafts corrupt structured output (tool calls, edit blocks).

## Build & Test

```bash
cargo build --release     # optimized build
cargo test --lib          # Rust unit tests (72 tests)
cargo clippy              # lint
python3 tests/test_agentless.py  # Python harness tests (10 tests)
```

## Project Structure

| Module | Purpose |
|--------|---------|
| `src/agent/` | Agent loop, generation, sampling, compaction, validators |
| `src/tool.rs` | Shell command execution with output chunking |
| `src/interceptor.rs` | Tool-call detection, XML edit parsing |
| `src/main.rs` | CLI entrypoint, model config |
| `src/server.rs` | OpenAI-compatible REST API |
| `benchmarks/agentic/` | SWE-bench evaluation harness |

## License

MIT
