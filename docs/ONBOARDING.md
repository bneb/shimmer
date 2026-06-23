# Shimmer Onboarding Guide

This guide covers the setup, configuration, and execution of the Shimmer engine.

## 1. System Requirements

- **Hardware:** Apple M-Series Chip (M1/M2/M3/M4).
- **OS:** macOS 13.0 or later.
- **Language:** Rust 1.75+.
- **Model:** A quantized GGUF model file (e.g., `gemma-4-12b.gguf`).

## 2. Installation

The build process relies on `llama-cpp-2` and will compile the Metal graph headers.

```bash
git clone https://github.com/bneb/shimmer.git
cd shimmer
cargo build --release
cargo install --path .
```

## 3. Execution Modes

Shimmer provides different execution modes based on the required workload.

### Interactive REPL Mode
For testing prompt behaviors or running a single agent loop interactively:
```bash
shimmer --speculative --main-model ~/.models/gemma-4-12b.gguf
```

### Batched Swarm Mode (MASC)
For bootstrapping multiple agents simultaneously:
```bash
shimmer --enable-swarm --main-model ~/.models/gemma-4-12b.gguf
```

### API Server Mode
Runs Shimmer as a standard HTTP daemon serving an OpenAI-compatible REST API (for generic clients):
```bash
shimmer --serve --main-model ~/.models/gemma-4-12b.gguf
```

### UDS Daemon Mode (Recommended)
Runs Shimmer on a local Unix Domain Socket (`/tmp/shimmer.sock`) for zero-latency communication:
```bash
shimmer --daemon --main-model ~/.models/gemma-4-12b.gguf
```

## 4. Configuration Flags

| Flag | Description | Use Case |
| :--- | :--- | :--- |
| `--speculative` | Enables Tree-PLD (Matrix-Matrix Decoding). | **Currently disabled by default** — n-gram drafts from prompt text cause token corruption in chat-template responses. |
| `--enable-swarm` | Evaluates multiple agents in a single pass. | Multi-agent frameworks. |
| `--enable-time-travel`| Spawns OS subprocesses before tool generation finishes. | Tasks involving high IO latency. |
| `--enable-ane-compression`| Compresses KV blocks on the Neural Engine. | Long-running context sessions. |
| `--warmup` | Forces Metal shader pre-compilation. | Cold-start optimization. |
| `--no-blind-edit-blocker` | Disables the blind edit blocker validator. | When model already knows the answer without investigation. |
| `--no-search-verifier` | Disables search content exact-match verification. | Debugging or non-exact-match workflows. |
| `--no-path-blocker` | Disables hallucinated file path blocker. | When working with files not yet on disk. |
| `--no-syntax-checker` | Disables pre-flight Python AST syntax checker. | Non-Python codebases. |
| `--no-insanity-detector` | Disables repeated tool call dedup. | When identical repeated tool calls are intentional. |
| `--no-preprocessor` | Skips heuristic retrieval preprocessor. | When repo context is not needed. |
| `--no-tools` | **Disables all tool detection and execution.** JSON markers are plain text. Edit tags still parse. All 6 validators bypassed. | Agentless single-shot generation (`--agentless` mode). |
| `--sample` | Sampling config: `temp=0.0,topk=0,repp=1.0`. | Greedy required for reliable tool detection (agentic mode only). |

### SWE-bench Pipeline

The recommended SWE-bench invocation (Gemma 4 12B):

**Agentic mode** (model investigates with tools):
```bash
python3 benchmarks/agentic/generate_swe_bench.py \
  --model models/gemma4-12b.gguf \
  --sample 10 \
  --seed 42 \
  --sample-cfg "temp=0.0,topk=0,repp=1.0"
```

**Agentless mode** (keyword localization + single-shot repair, no tool loop):
```bash
python3 benchmarks/agentic/generate_swe_bench.py \
  --agentless \
  --model models/gemma4-12b.gguf \
  --sample 10 \
  --seed 42
```

Resume from partial run: add `--resume` to skip already-completed instances.

## 5. Integrating with Google Antigravity (`agy`)

Shimmer supports **Google Antigravity (`agy`)** workspaces. While you can use the standard OpenAI HTTP interface, we highly recommend using the UDS adapter for maximum speed.

<details open>
<summary><b>Option A: UDS Socket (Zero Latency)</b></summary>
<br>
Start the UDS background server:

```bash
shimmer --daemon --main-model ~/.models/gemma-4-12b.gguf
```
*Note: This requires configuring the `agy_uds_client.py` adapter to route requests through `/tmp/shimmer.sock`.*
</details>

<details>
<summary><b>Option B: Standard REST API</b></summary>
<br>
Start the HTTP background server:

```bash
shimmer --serve --main-model ~/.models/gemma-4-12b.gguf
```
Run `agy` against the Shimmer HTTP host:

```bash
agy run --agent SWE_Agent --provider openai --base-url http://127.0.0.1:8080/v1 --model gemma4-12b
```
</details>
