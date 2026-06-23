# Project Shimmer

Shimmer is a low-latency execution engine for local LLM agents, built natively for Apple Silicon via Apple Metal and `llama.cpp`.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://www.rust-lang.org)

## Summary

In standard AI agent architectures, an agent loop works by having the LLM generate a tool call, stopping inference, and sending the text over an HTTP API to a Python orchestrator. The Python orchestrator parses the tool, runs the subprocess, formats the output, and sends the entire conversation history back to the LLM to resume generation.

Shimmer bypasses this bottleneck by moving read-only tool execution directly to the model daemon. When Shimmer detects an agent attempting to run a command (like `rg`, `fd`, or `cat`), it pauses the GPU, natively spawns the OS subprocess, and injects the stdout results directly into the local KV cache. The model then immediately resumes generation. By eliminating the HTTP round-trip and Python serialization overhead, agents can rapidly search and read through local codebases without waiting on an external orchestrator.

### Current Status (June 2026)

Shimmer successfully runs SWE-bench instances with Gemma 4 12B Coding (Q4_K_M). The pipeline includes 6 safety validators (blind edit blocker, TDD enforcement, path blocker, search verifier, syntax checker, insanity detector) all operating at turn boundaries with append semantics.

| Model | TPS | SWE-bench |
|-------|-----|-----------|
| Gemma 4 12B Coding (Q4_K_M) | 2.6-3.5 | Produces correct edits with good prompts |
| Qwen 2.5 Coder 7B | 8.5-8.7 | Fast, lower accuracy |

For a full breakdown of Shimmer's speedup, RAM footprint, and parameter scaling capabilities, see the [Performance Benchmarks](docs/BENCHMARKS.md).

## Architecture

Traditional agent frameworks incur latency through HTTP serialization and isolated sequential execution. Shimmer embeds agentic orchestration directly within the inference loop to minimize overhead.

*   **Tree-PLD (Matrix-Matrix Prompt Lookup Decoding):** Implements speculative decoding by matching n-grams from the KV cache and branching draft trees for batch evaluation on Metal. Currently disabled for agentic workloads (n-gram drafts corrupt chat-template tool calls).
*   **MASC (Batched Multi-Agent Concurrency):** Evaluates multiple independent agent sequences in a single forward pass by sharing the underlying system prompt KV cache.
*   **Edit Validator Pipeline:** 6 validators guard against common model failures (editing without investigation, hallucinated paths, incorrect search strings, syntax errors). All operate at turn boundaries with append semantics.
*   **Speculative Tool Execution:** Spawns OS subprocesses eagerly upon detecting tool invocation syntax, masking tool execution latency behind sequence generation.
*   **ANE KV Compression:** Offloads and compresses evicted reasoning blocks into dense vectors via the Apple Neural Engine to manage context limits.
*   **RoPE-Safe KV Compaction:** Shifts continuous KV blocks and updates Rotary Positional Embeddings to prevent OOM errors during extended context usage.

## Quick Start

Shimmer is written in Rust and requires macOS running on Apple Silicon.

### 1. Build and Install

```bash
git clone https://github.com/bneb/shimmer.git
cd shimmer
cargo install --path .
```

### 2. Run the Engine

<details open>
<summary><b>Option A: UDS Daemon (Recommended for Zero-Latency Agents)</b></summary>
<br>
Start the background daemon on a Unix Domain Socket to completely eliminate TCP/IP and Nagle's algorithm overhead:

```bash
shimmer --daemon --main-model ~/.models/gemma-4-12b.gguf
```
</details>

<details>
<summary><b>Option B: REST API (For standard OpenAI compatibility)</b></summary>
<br>
If your client doesn't support UDS, you can run a standard HTTP server:

```bash
shimmer --serve --main-model ~/.models/gemma-4-12b.gguf
```
</details>

### 3. Usage with Antigravity (`agy`)

Shimmer is specifically designed to act as an ultra-fast accelerator backend for **Google Antigravity (`agy`)**, an autonomous agentic workspace platform. 

To achieve the maximum 8.3x speedup, point `agy` to the Shimmer UDS socket using the custom UDS adapter (if configured), or use the standard local HTTP loopback:

```bash
agy run --agent SWE_Agent --provider openai --base-url http://127.0.0.1:8080/v1 --model gemma4-12b
```

### 4. Verify the Benchmarks

Don't just take our word for it. You can independently verify the 8.3x end-to-end speedup on your own Mac by running the included `swe_bench_shimmer.py` benchmark:
```bash
cd benchmarks/agentic
python swe_bench_shimmer.py
```

## Documentation

*   [Onboarding & Configuration](docs/ONBOARDING.md)
*   [Architecture Overview](docs/ARCHITECTURE.md)
*   [Performance Benchmarks](docs/BENCHMARKS.md)
