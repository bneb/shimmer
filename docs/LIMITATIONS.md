# Known Limitations

Shimmer is under active development. This document records design tradeoffs
and constraints that are deliberate rather than accidental.

## Speculative decoding

Prompt Lookup Decoding is enabled by default for raw text generation. It
provides a ~55% throughput improvement by drafting n-gram continuations
from the context window and verifying them in parallel via Metal.

It is **not recommended** for agentic workloads (SWE-bench, tool-using agents).
The n-gram drafts can match prompt text instead of model output, causing
token corruption in JSON tool-call markers: missing braces, token fusion
(e.g., `"toolrg"` instead of `{"name": "rg"...}`). Disable with
`--no-speculative` or `--speculative false` when using tools.

## Temperature sampling

Temperature values above zero apply softmax-based multinomial sampling
from the scored candidate distribution. This produces non-deterministic
output. It is **incompatible** with reliable tool detection — the
interceptor matches exact ````json` markers, and temperature-induced
variance in these markers causes missed detections.

For agentic workloads, use `temp=0.0` (greedy). For creative or non-tool
generation, `temp=0.7` with `--no-tools` works well.

## Swarm mode

Despite the module name, swarm mode runs the same prompt on multiple
parallel contexts — batch throughput, not multi-agent coordination.
Each context shares a single KV cache arena with per-agent LoRA adapters
and compaction. It is useful for benchmarking throughput or serving
multiple identical requests concurrently, but it does not implement
agent-to-agent communication or task distribution.

## 12B Q4 is the practical floor

At 6.86 GiB (Q4_K_M), Gemma 4 12B fits comfortably in unified memory but
edit accuracy is limited by parameter count and quantization. The model
finds correct files ~60% of the time on SWE-bench Lite but produces
semantically correct edits on only ~20% of instances. Larger models
(30B+) would improve accuracy but exceed the memory bandwidth of consumer
Apple Silicon for real-time use.

## GPU debug output

The `llama.cpp` Metal backend occasionally emits GPU kernel compilation
logs to stdout, which can interleave with model output and corrupt
patches during SWE-bench extraction. This is an upstream issue in
`llama-cpp-sys-2`. Mitigation: the Python harness strips known Metal
debug patterns from output before patch extraction.

## REPL mode and raw chat templates

The REPL mode (`shimmer` with no flags, reading stdin) emits raw
`<start_of_turn>` / `<end_of_turn>` template tokens to the terminal.
This is a debugging convenience, not a polished user interface.

## Context window

The default context size is 16384 tokens. Tool output consumes window
space permanently (append semantics — see ADR-001). KV cache compaction
evicts old tool output pages to free space, but large repositories
with many tool calls can still exhaust the window. The hard cap is
8192 generated tokens per query.
