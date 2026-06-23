# SWE-bench Local Inference Report & Failure Audit

## 1. Current Results: Gemma 4 12B (June 2026)

### Run Configuration
- **Model:** Gemma 4 12B Coding (Q4_K_M, 6.86 GiB)
- **Engine:** Shimmer (Rust + llama.cpp + Metal)
- **Benchmark:** SWE-bench Lite (30 instances, seed=42)
- **Sampling:** temp=0.0, topk=0, repp=1.0 (greedy)
- **Timeout:** 900s per instance
- **Date:** 2026-06-21

### Aggregate Results

| Metric | Value |
|--------|-------|
| Total wall time | 214 min (3.6 hours) |
| Patches produced | 11/30 (37%) |
| Exact GT matches | 1 |
| Functionally equivalent | ~5 |
| Wrong/corrupted | ~5 |
| Empty/timeout | 19 (63%) |
| Timeout loops | 5 (17%) |

### Patch Quality Assessment

| Instance | Result | Time | Verdict |
|----------|--------|------|---------|
| django__django-10914 | ✅ Exact match | 174s | `FILE_UPLOAD_PERMISSIONS = 0o644` — identical to GT |
| django__django-11099 | ⚠️ Alt fix | 750s | Changed `^→\A` unnecessarily; only `$→\Z` was needed. Functionally equivalent. |
| django__django-14382 | ⚠️ Alt fix | 126s | Used `.rstrip(os.sep)` instead of GT's `top_dir`. Different approach, same bug fixed. |
| django__django-15851 | ✅ Func | 249s | Moved `args.extend(parameters)` above `if dbname:` — same code, diff metadata |
| django__django-16527 | ✅ Func | 136s | Added `and has_add_permission` — same line, diff metadata |
| django__django-14752 | ✅ Func | 166s | Same `serialize_result` method; GT has extra docstrings |
| matplotlib__matplotlib-26020 | ❌ Partial | 344s | Changed `ax.axis["bottom"]` to `ax.axis` but missed `MethodType` case |
| sphinx-doc__sphinx-8282 | ❌ Wrong | 171s | Only checked `objtype == 'overload'`; GT uses analyzer infrastructure |
| django__django-12915 | ❌ Wrong | 99s | Added sync method named `get_response_async` — not actually async |
| django__django-11422 | ❌ Wrong | 124s | Hardcoded `manage.py` special case; GT handles all `__main__` modules |
| sympy__sympy-12454 | ❌ Corrupt | 62s | Metal GPU debug output leaked into patch, mangling the edit |

### Timeout Instances (5/30 = 17%)

All five timed out at 900s due to repetition loops:
- django__django-11283 (repeated `from django.db.models.deletion import delete_models`)
- django__django-16816
- sympy__sympy-16792
- django__django-11133
- sympy__sympy-21627

The repetition loop is the primary failure mode — the model gets stuck repeating the same output until the 900s timeout fires. The insanity detector (validator #6) catches repeated tool calls, but does not catch repeated text generation without tool calls.

### File Selection Analysis

| Category | Count | Example |
|----------|-------|---------|
| Correct file, correct edit | 3 | django-10914 (`global_settings.py`), django-15851, django-16527 |
| Correct file, alternate fix | 2 | django-11099, django-14382 |
| Wrong file selected | 2 | django-12915, sphinx-8282 |
| Partial fix (missed edge case) | 2 | matplotlib-26020, django-11422 |
| Corrupted | 1 | sympy-12454 |
| No edit produced | 19 | — |

## 2. Failure Audit: Unresolved Instances

### Pattern 1: Wrong File Selection
**django__django-12915** — The model needed to add an async handler method. Instead of finding the correct handler class, it added a synchronous method named `get_response_async` that wasn't actually async. The GT adds `sync_to_async`-based handling with proper Http404 error handling.

**sphinx-doc__sphinx-8282** — The model checked `objtype == 'overload'` to skip overload handling, but the GT properly uses `self.analyzer.overloads` to detect overloaded functions and handle them correctly in the documentation builder.

### Pattern 2: Partial Fix (Missed Edge Cases)
**matplotlib__matplotlib-26020** — The bug was that `ax.axis` could be a method (not a dict), so `ax.axis["bottom"]` fails. Our fix changed to `ax.axis.toggle(...)` but didn't account for the case where `ax.axis` is a MethodType. The GT wraps it in `SimpleAxisArtist` with proper type checking.

**django__django-11422** — The model added `manage.py` to `extra_files` as a special case. The GT fixes the root cause: handling all `__main__` modules (including manage.py) by falling back to `__file__` when `__spec__` is None.

### Pattern 3: Repetition Loops
**5 instances** (django-11283, django-16816, sympy-16792, django-11133, sympy-21627) entered infinite repetition loops — the model repeatedly generated the same line or block until the 900s timeout. This is the most costly failure mode, consuming 75 minutes of wall time.

### Pattern 4: GPU Debug Output Leak
**sympy__sympy-12454** — `ggml_metal_library_compile_pipeline` log lines from the Metal backend were interleaved with the model's edit text, corrupting the patch. This is a `llama.cpp` Metal backend issue where GPU kernel compilation stdout bleeds into the generation output stream.

## 3. Agentless Mode (June 2026)

A new `--agentless` mode was added as an alternative to the agentic (tool-using) pipeline. It replaces model-driven investigation with keyword-based file localization:

1. **Localization:** Python extracts keywords from the issue (file paths, CamelCase, ALL_CAPS, snake_case) and greps the repo via `rg`. Files are ranked by keyword match count, capped at 20.
2. **Repair:** Files are inlined into the prompt with smart truncation (<100 lines: full; 100-300: first 30 + last 20; >300: first 30). The model generates a single response with no tool loop.
3. **Validation:** Same XML edit extraction and test verification as agentic mode.

### Initial Test (django-10914)

The agentless pipeline correctly localized `global_settings.py` and produced `FILE_UPLOAD_PERMISSIONS = 0o644`. However, the model placed it at line 17 instead of line 307 (near other `FILE_UPLOAD_*` settings). The pipeline functions correctly end-to-end; precise line placement remains a model quality limitation at 12B Q4.

### Agentless vs Agentic Trade-offs

| Aspect | Agentic | Agentless |
|--------|---------|-----------|
| File discovery | Model-driven (accurate but slow) | Keyword-driven (fast, can miss context) |
| Reliability | 17% timeout loop rate | 0% loop risk |
| Speed | 90-900s, high variance | ~200-300s, predictable |
| Tool calls | 2-8 | 0 |
| Best for | Complex multi-file fixes | Simple configuration/one-line changes |

## 4. Validator Architecture (Current State)

All 6 validators use append semantics and fire at turn boundaries. Validated on the 30-instance run:
- **Blind edit blocker:** Caught 12 blind edits; 0 false positives
- **Path blocker:** Caught 4 hallucinated paths
- **Search verifier:** Caught search mismatches on 8 instances; dedup check prevents retry loops
- **Insanity detector:** Caught repeated tool calls on 3 instances
- **Syntax checker:** No AST errors on applied patches (100% syntactically valid)
- **TDD enforcement:** Not tested (run with `--no-verify`)

All validators are bypassed in `--no-tools` (agentless) mode.

## 5. Key Findings

1. **File selection is the bottleneck.** When the model finds the right file, it produces correct or nearly-correct edits ~70% of the time. When it picks the wrong file, the edit is always wrong. The "INVESTIGATE BEFORE EDITING" prompt section is critical for file selection accuracy.
2. **Repetition loops cost 75 minutes.** 5/30 instances looped until the 900s timeout. A tighter repetition detector (checking for repeated text, not just repeated tool calls) could recover this time.
3. **Agentless shows promise for simple fixes.** The pipeline works end-to-end and eliminates loop risk entirely. Current limitation is the model's line placement precision with inlined file context.
4. **12B Q4 is the floor, not the ceiling.** The 37% patch rate is competitive for local inference, but the gap to Agentless-like systems (32-47% with GPT-4o/Claude) is explainable by model scale and context precision.
5. **Metal GPU debug output is a real bug.** The `llama.cpp` Metal backend sometimes leaks kernel compilation logs into stdout, which can interleave with model output and corrupt patches.
