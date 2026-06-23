# Design Overview

Shimmer is an execution engine for tool-using LLM agents on Apple Silicon. It wraps
`llama.cpp` with Metal GPU acceleration, adding streaming tool interception, edit
validation, and KV cache compaction inside the inference loop.

## Data flow

### 1. Prompt intake

Prompts enter through one of three paths:
- **CLI** (`main.rs`): stdin or `--prompt` flag, single-shot or REPL mode
- **HTTP** (`server.rs`): OpenAI-compatible `POST /v1/chat/completions` with
  streaming SSE and sync JSON response modes. Optional Bearer-token auth via
  `--api-key`.
- **UDS daemon** (`daemon.rs`): JSON-line protocol over a Unix domain socket at
  `$SHIMMER_SOCKET`, `$XDG_RUNTIME_DIR/shimmer.sock`, or `/tmp/shimmer.sock`.

### 2. Preprocessing

Before inference, `preprocessor.rs` scans the repository working directory:
- Extracts search terms from the prompt (camelCase, snake_case, file paths)
- Greps matching files via `rg`, ranks by match count
- Injects a `<repository_context>` block with directory structure, test file
  locations, and matched source snippets
- Results are cached by prompt hash in `$SHIMMER_CACHE_DIR` or `/tmp`

### 3. Model loading

`models.rs` detects the model architecture from the GGUF path (Gemma 4,
Qwen 2.5 Coder) and configures the appropriate chat template. The model is
loaded via `llama-cpp-2` with Metal GPU offloading. Context size defaults
to 16384 tokens.

### 4. Generation loop

`agent/mod.rs` drives token-by-token generation. Each token passes through
the `ToolInterceptor` (`interceptor.rs`), which scans the output buffer for:

- **JSON tool calls**: `` ```json\n{...}\n``` `` markers trigger tool execution
- **XML edit blocks**: `<edit file="...">` tags are parsed for path, search,
  replace, and closing `</edit>` boundaries

The loop runs until `MAX_TOKENS` (8192) or an end-of-generation token.

### 5. Tool execution

When a tool call is detected, `agent/tools.rs` dispatches it via `tool.rs`:
- **Read-only tools** (rg, grep, fd, find, cat, ls, git status/diff): executed
  synchronously with 30s timeout
- **Blocked tools** (sed -i): rejected with an error message
- **run_test**: executes `python -m pytest` or a custom test command via bash
- **Speculative execution** (`interceptor.rs`): idempotent tools can be
  pre-spawned before the JSON block closes, reducing latency

Tool output is appended to the KV cache as a synthetic user turn. No rollback
is performed — see [ADR-001](adr/001-append-not-rollback.md).

### 6. Edit validation

Six validators (`agent/validators.rs`) fire at turn boundaries (after `</edit>`
or end-of-generation), in order:

1. **Blind edit blocker**: rejects edits made without prior tool investigation
2. **TDD enforcement**: requires `run_test` between successive edits
3. **Path blocker**: rejects edits targeting non-existent files
4. **Search verifier**: rejects `<search>` blocks that match zero or
   multiple times in the target file. Dedup tracking prevents retry loops.
5. **Syntax checker**: AST-compiles patched Python files via `python3`, rejects
   syntax errors
6. **Insanity detector**: catches repeated tool calls with identical arguments,
   escalating from nudge to hard abort at 11 repetitions

All validators use append semantics — see [ADR-003](adr/003-validators-at-turn-boundaries.md).

### 7. Patch extraction

The Python harness (`benchmarks/agentic/generate_swe_bench.py`) extracts
patches from model output:
- Strips everything before the final `Begin now.` prompt marker
- Finds `<edit>` blocks via two-step regex (block boundary, then content)
- Applies fuzzy search/replace with line-by-line matching
- Deduplicates identical (file, search, replace) tuples
- Produces a `git diff` patch for SWE-bench evaluation

## Key design decisions

| Decision | Rationale | ADR |
|----------|-----------|-----|
| Append, not rollback | Compatible with speculative decoding; avoids position-tracking drift | [001](adr/001-append-not-rollback.md) |
| Greedy sampling (temp=0.0) | Required for reliable tool-call detection (exact `` ```json `` matching) | [002](adr/002-greedy-sampling-required.md) |
| Validators at turn boundaries | Prevents infinite loops from mid-thought interruption (Gemma 4 `<channel\|thought>`) | [003](adr/003-validators-at-turn-boundaries.md) |

## Module map

| Module | Purpose |
|--------|---------|
| `agent/mod.rs` | Agent struct, generation loop, public API |
| `agent/state.rs` | `AgentConfig`, `EngineState`, constants |
| `agent/sampling.rs` | Token sampling, temperature, top-k |
| `agent/compaction.rs` | KV cache compaction, entropy summarization |
| `agent/tools.rs` | Tool execution with append semantics |
| `agent/swarm.rs` | Multi-agent swarm, continuation logic |
| `agent/validators.rs` | Six safety validators at turn boundaries |
| `tool.rs` | Shell command execution, output chunking |
| `compaction.rs` | KV cache page management |
| `interceptor.rs` | Tool-call detection, XML edit parsing |
| `main.rs` | CLI entrypoint, `--no-tools`, model config |
| `server.rs` | OpenAI-compatible REST API (`--serve`) |
| `preprocessor.rs` | Repo search, context injection |
| `speculative.rs` | Draft token generation + verification |
| `models.rs` | Model detection + capability flags |
| `daemon.rs` | Unix socket daemon mode |
