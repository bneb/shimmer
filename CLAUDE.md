# Shimmer — Quality Standards & Project Context

## Quality Gates (mandatory for all changes)

| Gate | Limit | Enforcement |
|------|-------|-------------|
| **File size** | < 400 LOC | Split files when exceeded |
| **Function size** | < 32 LOC (body only) | Extract helpers for anything longer |
| **Nesting depth** | ≤ 3 levels | Early returns, guard clauses, helper fns |
| **Test coverage** | Every new function has a test | `cargo test` must pass before commit |
| **Clippy** | Zero new warnings | `cargo clippy` on every build |
| **unwrap/expect** | Banned in new code | Use `Result` propagation or proper error enums |
| **Module visibility** | `pub` only for cross-module API | `pub(crate)` preferred; bare `fn` for internal |

## Architecture

Shimmer is a local LLM execution engine for agentic coding (SWE-bench) on Apple Silicon via Metal + llama.cpp.

### Key Modules

| Module | Purpose | LOC | Status |
|--------|---------|-----|--------|
| `agent/mod.rs` | Agent struct, daemon, generation loop, public API | 1147 | ⚠️ (>400, needs split) |
| `agent/state.rs` | EngineState, AgentConfig, constants, benchmark types | 107 | ✅ |
| `agent/sampling.rs` | Token sampling, temperature, rep penalty, top-k | 88 | ✅ |
| `agent/compaction.rs` | KV cache compaction, entropy summarization | 143 | ✅ |
| `agent/tools.rs` | Tool execution with append semantics, capacity mgmt | 79 | ✅ |
| `agent/swarm.rs` | Multi-agent swarm, continuation logic | 383 | OK |
| `tool.rs` | Shell command execution, output formatting, smart chunking | 294 | OK |
| `compaction.rs` | KV cache page management, recent-output preservation | 336 | OK |
| `interceptor.rs` | Streaming tool-call detection, XML edit tag parsing, edit boundary detection | 341 | OK |
| `main.rs` | CLI entrypoint, chat templates, model config, sample parsing | 343 | OK |
| `server.rs` | OpenAI-compatible REST API | 439 | OK |
| `preprocessor.rs` | Repo search, file-based context injection, structure scan | 307 | OK |
| `speculative.rs` | Draft token generation + verification | 448 | OK |
| `models.rs` | Model detection + capability flags | 202 | OK |
| `daemon.rs` | Unix socket daemon mode | 158 | OK |

### Model Configurations (benchmarked on M4, June 2026)

| Model | Detected As | TPS | Best For |
|-------|------------|-----|----------|
| Gemma 4 12B Coding (Q4_K_M) | `Gemma4_12B` | 2.6-3.5 | SWE-bench. Finds relevant files. Produces correct edits when the prompt structures the investigation. |
| Qwen 2.5 Coder 7B | `Qwen25Coder7B` | 8.5-8.7 | SWE-bench. Faster but lower edit accuracy on complex problems. |

All models: speculative decoding disabled (see Known Limitations #5), Metal GPU offloading (99 layers), greedy sampling (temp=0.0) required for reliable tool detection.

### Validator Architecture (6 safety layers, all append-semantics)

Validators operate at **turn boundaries** (after `</edit>` or EOG), never mid-generation. All use `history.len()` as ground-truth position — no `n_cur` drift.

| # | Validator | Trigger | Action | KV Cache |
|---|-----------|---------|--------|-----------|
| 1 | **Blind edit blocker** | ANY `</edit>` closure with `tool_calls == 0` | Inject user-turn nudge: "use tools first" | Append |
| 2 | **TDD enforcement** | Complete edit (has `<replace>`) with `tests_since_last_edit == 0` | Inject user-turn nudge: "run tests first" | Append |
| 3 | **Path blocker** | `<edit file="...">` with non-existent path | Inject user-turn nudge with actual path | Append |
| 4 | **Search verifier** | `<search>...</search>` that doesn't match exactly once | Inject nudge + dedup check + "use `cat` to read the file" | Append |
| 5 | **Syntax checker** | Complete Python edit block | AST-compile the patched file, reject on error | Append |
| 6 | **Insanity detector** | Repeated tool call with identical args | Dedup check via `tool_history` HashSet; nudge on repeat, hard reject at 11 | Append |

**Key architectural fix (2026-06-21):** The blind edit blocker and TDD enforcement previously fired on *partial* `<edit` detection by scanning `interceptor.buffer.contains("<edit ")`. This interrupted the model mid-thought, causing infinite loops with Gemma 4's `<channel|thought>` reasoning. They now fire on `edit_tag_closed` (set when `</edit>` is seen) — the model finishes its thought before we intervene.

**Why append, not rollback:** See `docs/ARCHITECTURE.md#validator-design`. Short version: rollback breaks speculation compatibility, causes position-tracking drift (`n_cur` vs `history.len()`), and interrupts thought chains in capable models.

### SWE-bench Pipeline (verified working as of 2026-06-21)

```
generate_swe_bench.py
  → git clone repo → checkout base commit
  → preprocessor: scan repo structure + inject exact source file paths
  → construct prompt with: repo layout + source files + tools + convergence rules
  → shimmer: load model → inference
  → model uses tools (output → .shimmer_tool_N.txt)
  → hard limit at 8 tool calls → force model to produce edit
  → model produces XML edit block (<edit file="..."><search>...</search><replace>...</replace></edit>)
  → extract_patch: two-step extraction (find block, extract within) + dedup + fuzzy matching → apply edits → git diff
  → append to predictions.jsonl with wall time and model name
```

**Key pipeline improvements (June 2026):**

| Date | Layer | Implementation |
|------|-------|---------------|
| Jun 19 | **Tool execution** | Append semantics (no KV cache rollback). 30s timeout per tool. Smart chunking for large outputs. |
| Jun 19 | **Convergence** | Three-phase: encourage search (<6 tools) → prompt edit (6+) → hard block tools (8). Cap: 10 continuation rounds. |
| Jun 19 | **Edit extraction** | XML `<edit file="...">` + JSON `{"edit": {"file":...}}` + bare `{"file":...}`. Fuzzy line-by-line matching for search strings. |
| Jun 19 | **Repo context** | Directory structure + test file locations + exact source file paths (walk 2 levels, 50 files) injected into prompt. |
| Jun 21 | **Validator architecture** | All 6 validators use append semantics + `history.len()` ground truth. Blind edit blocker & TDD enforcement fire on `</edit>` boundary, not partial `<edit`. |
| Jun 21 | **Search verifier** | Added `edit_history` dedup to break retry loops. Feedback now suggests `cat <file>` to read actual content. |
| Jun 21 | **Extraction fixes** | Empty-search guard (prevents `\n` → `""` → file explosion). Non-spanning regex (two-step: find block, extract within). Dedup identical (file, search, replace) tuples. |
| Jun 21 | **Prompt engineering** | Added "INVESTIGATE BEFORE EDITING: FIRST use rg to search" section. Model now finds `global_settings.py` (correct) instead of `db/utils.py` (wrong). |
| Jun 21 | **Models** | Removed Qwen3Coder30B and Qwen36_27B (unused). Speculation stays disabled (n-gram drafts corrupt JSON). |

### Prompt Template

The prompt in `benchmarks/agentic/generate_swe_bench.py::build_prompt` includes an "INVESTIGATE BEFORE EDITING" section that structures the workflow as: (1) search with `rg`, (2) read with `cat`, (3) produce the edit. Without this section, the model often skips investigation and edits a plausible but incorrect location. On `django__django-10914`, this mattered: the model edited `db/utils.py` without it, and `conf/global_settings.py` (the correct target) with it. See the source for the current template.

### Tool Output Architecture

Tool output is written to `.shimmer_tool_N.txt` files and injected as a user turn using the model's chat template. The system prompt instructs the model to wrap JSON tool calls in ```json blocks; the interceptor detects these blocks and dispatches tool execution.

## Known Limitations

1. **Model accuracy limited by parameter count and quantization.** On `django__django-10914`, Gemma 4 12B (Q4_K_M) produced a correct edit but required an explicit "search first" prompt instruction; without it, the model selected a plausible but incorrect file. Qwen 2.5 Coder 7B finds relevant files but produces semantically incorrect edits (no-ops, wrong logic) on non-trivial problems.
2. **temp>0 breaks tool detection.** The interceptor matches exact ```json markers. At temperature > 0, token sampling introduces variance in these markers, causing missed detections. All SWE-bench runs use temp=0.0.
3. **Speculative decoding disabled.** n-gram drafts sourced from the prompt cause token corruption in chat-template responses. Observed failures: missing `{` and `"` characters, token fusion (`"toolrg"` instead of `{"name": "rg"...}`). Root cause: drafts match prompt continuations, not model-generation patterns. No corruption observed in raw text generation.
4. **Disk space.** SWE-bench clones (django, matplotlib) require >15GB free for batch evaluation.
5. **TDD enforcement and syntax checking require complete edit blocks.** These validators apply only when the model produces `<search>`, `<replace>`, and `</edit>` tags. Incomplete edit blocks (missing `<replace>`) are caught by the blind edit blocker instead.
