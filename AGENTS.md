# Shimmer — Agent Instructions

Local LLM execution engine for agentic coding (SWE-bench) on Apple Silicon via Metal + llama.cpp.

## Quality Gates

Violating any of these fails review.

| Gate | Limit | Fix |
|------|-------|-----|
| File size | < 400 LOC | Split into sub-modules |
| Function body | < 32 lines | Extract helpers |
| Nesting depth | ≤ 3 levels | Early return, guard clauses |
| Test coverage | Every new `fn` has a test | `cargo test` must pass |
| Clippy | Zero new warnings | `cargo clippy` on build |
| `unwrap`/`expect` | Banned in new code | `Result` propagation or error enum |
| Visibility | `pub` only for cross-module API | `pub(crate)` preferred |

## Build & Test

```bash
cargo build --release          # build
cargo test                     # 62 tests (Rust)
cargo clippy                   # lint
python3 tests/test_agentless.py  # 10 tests (Python agentless)
python3 tests/smoke_test.py      # end-to-end smoke test
```

## Module Map

| Module | LOC | Purpose | Touch when |
|--------|-----|---------|------------|
| `agent/mod.rs` | 1150 | Agent struct, generation loop, public API | Adding agent behavior |
| `agent/state.rs` | 108 | `AgentConfig`, `EngineState`, constants | Adding config flags |
| `agent/sampling.rs` | 88 | Token sampling, temperature, top-k | Changing sampling |
| `agent/compaction.rs` | 143 | KV cache compaction, entropy summarization | Context window issues |
| `agent/tools.rs` | 79 | Tool execution with append semantics | Adding tools |
| `agent/swarm.rs` | 383 | Multi-agent swarm, continuation logic | Multi-agent features |
| `tool.rs` | 294 | Shell command execution, output chunking | Tool output formatting |
| `compaction.rs` | 336 | KV cache page management | Memory issues |
| `interceptor.rs` | 389 | Tool-call detection, XML edit parsing, `--no-tools` guard | Edit/tool parsing |
| `main.rs` | 349 | CLI entrypoint, `--no-tools` flag, model config | Adding CLI flags |
| `server.rs` | 439 | OpenAI-compatible REST API (`--serve`) | API changes |
| `preprocessor.rs` | 307 | Repo search, context injection, structure scan | Preprocessing |
| `speculative.rs` | 448 | Draft token generation + verification | Speculation (disabled) |
| `models.rs` | 202 | Model detection + capability flags | New model support |
| `daemon.rs` | 158 | Unix socket daemon mode | Daemon changes |
| `generate_swe_bench.py` | 1050 | SWE-bench harness (agentic + agentless) | Eval pipeline changes |
| `test_agentless.py` | 170 | Agentless unit tests | Agentless feature work |

## Models (M4, June 2026)

| Model | Detected As | Agentic TPS | Agentless TPS | Notes |
|-------|------------|-------------|---------------|-------|
| Gemma 4 12B Coding (Q4_K_M) | `Gemma4_12B` | 2.6–3.5 | ~5.1 | Primary; 80% edit rate on SWE-bench Lite (30 instances) |

## SWE-bench Lite (June 2026)

Hybrid mode (context + up to 3 read-only tools, exact-match-only), seed=42:

| Instance | Applied | Wall | Tools | Result |
|----------|---------|------|-------|--------|
| django-11099 | 2/2 | 137s | 0 | ✓ regex fix (exact) |
| django-13551 | 1/1 | 350s | 1 | ✗ wrong file |
| matplotlib-25498 | 0/0 | 900s | 2 | timed out |

**1/3 plausible, 0/3 test-verified** (test infrastructure needs pytest in clone envs).

Earlier agentless runs achieved 80% edit application but 0% test pass rate —
the fuzzy matching inflated numbers without improving correctness. Hybrid mode
with exact-match-only is more honest.

All runs: `temp=0.0,topk=0,repp=1.0` (greedy). Speculation disabled — n-gram drafts corrupt JSON tool calls.

## Tool Detection & Edit Parsing

Tools are detected via ````json` markers in the interceptor. `ToolInterceptor::feed_token()` scans the buffer for `TOOL_MARKER_START`/`TOOL_MARKER_END`. When found, the tool is executed and output is injected as a user turn (append semantics — no KV cache rollback).

**`--no-tools` flag:** Sets `AgentConfig.disable_tool_interceptor = true`. JSON tool detection is gated behind `detect_json_tools = false`. Edit tag parsing (`<edit>`, `<search>`, `<replace>`, `</edit>`) continues to run. All 6 validators are bypassed. Used by agentless mode.

## SWE-bench Pipeline

Two modes in `benchmarks/agentic/generate_swe_bench.py`:

**Agentic** (`build_prompt`): Model gets repo structure + tools + "INVESTIGATE BEFORE EDITING" prompt. Uses `rg`/`cat`/`fd` to investigate, then produces `<edit>` blocks. Tool calls capped at 8.

**Agentless** (`build_agentless_prompt` + `--agentless`): Python extracts keywords → `rg` grep ranks files → files inlined into prompt → `shimmer --no-tools` single-shot generation. 0 tool calls, no loop risk.

Both extract patches via `extract_patch()` — XML `<edit>` blocks with fuzzy search/replace, path resolution, deduplication. `Begin now.` anchor filters prompt examples from extraction.

## Known Footguns

1. **temp>0 breaks tool detection.** The interceptor matches exact ````json` markers. Must use `temp=0.0`.
2. **Speculation corrupts chat templates.** n-gram drafts inject prompt-text tokens during generation. Disabled.
3. **Repetition loops are the #1 failure mode.** 5/30 instances (17%) looped until 900s timeout. The insanity detector catches repeated tool calls but not repeated text without tools.
4. **Metal GPU debug output can leak.** `llama.cpp` Metal backend sometimes emits kernel compilation logs to stdout, interleaving with model output.
5. **`global_settings.py` vs `storage.py`.** The model often picks implementation files over config files. The agentless prompt includes "prefer settings/conf files" guidance.
6. **SIGTERM doesn't kill GPU kernels.** Use the `SIGTERM → SIGKILL` escalation in `run_shimmer()`.
7. **Prompt examples bleed into extraction.** The `Begin now.` anchor in `extract_patch()` strips everything before it. Placeholder names use `__demo_placeholder__` to prevent matches in real files.
8. **SWE-bench clones need >15GB.** django + matplotlib + sympy clones accumulate.

## Key Architecture Decisions

- **Append, not rollback.** Tool output is appended to KV cache. Rollback breaks speculation compatibility, causes position-tracking drift (`n_cur` vs `history.len()`), and interrupts thought chains.
- **Validators fire at turn boundaries** (after `</edit>` or EOG), never mid-generation. Blind edit blocker and TDD enforcement use `edit_tag_closed` — the model finishes its thought before intervention.
- **`history.len()` is ground truth for position.** Never trust `n_cur` for position tracking after compaction.
