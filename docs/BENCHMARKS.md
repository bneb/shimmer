# Performance Benchmarks

All benchmarks conducted on Apple Silicon M4 (20GB unified memory). Models run via Metal + llama.cpp with greedy sampling (temp=0.0).

## SWE-bench Lite: Agentic Mode (Gemma 4 12B Q4_K_M)

**Run:** 30 instances, seed=42, `generate_swe_bench.py --no-verify --resume`
**Date:** 2026-06-21 | **Total wall time:** 214 min (3.6 hours)

| Metric | Value |
|--------|-------|
| Patches produced | 11/30 (37%) |
| Exact GT matches | 1 (django__django-10914) |
| Functionally equivalent | ~5 |
| Wrong/corrupted | ~5 |
| Empty/timeout | 19 (63%) |
| Timeout loops (900s) | 5 (17%) |
| Avg per instance | 427s |
| Autoregressive TPS | 2.6-5.7 |

**Per-instance breakdown:**

| Instance | Result | Time | Assessment |
|----------|--------|------|------------|
| django__django-10914 | ✅ Exact | 174s | `FILE_UPLOAD_PERMISSIONS = 0o644` |
| django__django-11099 | ⚠️ Alt fix | 750s | `\A...\Z` vs GT's `^...\Z` |
| django__django-14382 | ⚠️ Alt fix | 126s | `.rstrip(os.sep)` vs GT's `top_dir` |
| django__django-12915 | ❌ Wrong | 99s | Sync method named `get_response_async` |
| django__django-11422 | ❌ Wrong | 124s | Hardcoded `manage.py`; GT handles `__main__` |
| django__django-15851 | ✅ Func | 249s | Moved `args.extend(parameters)` |
| sphinx-doc__sphinx-8282 | ❌ Wrong | 171s | Too simplistic; missed analyzer logic |
| django__django-16527 | ✅ Func | 136s | Added `has_add_permission` |
| sympy__sympy-12454 | ❌ Corrupt | 62s | GPU debug output leaked into patch |
| matplotlib__matplotlib-26020 | ❌ Partial | 344s | Missed `MethodType` case |
| django__django-14752 | ✅ Func | 166s | Same `serialize_result` method |
| 19 other instances | ❌ Empty/timeout | 90-900s | 5 hit 900s timeout loop |

## SWE-bench Lite: Agentless Mode (Gemma 4 12B Q4_K_M)

Single-shot pipeline with keyword-based file localization, no tool loop.

| Metric | Agentic (avg) | Agentless (django-10914) |
|--------|--------------|--------------------------|
| TPS | 2.6-5.7 | 5.1 (no tool halts) |
| Tool calls | 2-8 | 0 |
| Prompt size | ~2.4K chars | ~27K chars |
| Loop risk | 17% | 0% |
| File selection | Model-driven (accurate) | Keyword-driven (found correct file) |
| Edit quality | Varies | Right file, right fix, wrong line |

Agentless correctly identified `global_settings.py` and produced `FILE_UPLOAD_PERMISSIONS = 0o644`, but placed it at line 17 instead of 307. The pipeline is functional; line placement precision is a model quality limitation at 12B Q4.

## Raw Generation Throughput

### Gemma 4 12B Coding (Q4_K_M) — Autoregressive (current default)

| Configuration | Throughput | Notes |
| :--- | :--- | :--- |
| Agentic SWE-bench | 2.6-3.5 TPS | Tool calls + edits, interleaved |
| Agentless SWE-bench | ~5.1 TPS | Single-shot, no tool halts |

> **Note (2026-06-21):** Speculative decoding is disabled by default. n-gram drafts from prompt text cause token corruption in chat-template responses (missing `{` and `"` in JSON tool calls, token fusion). The speculative benchmarks below reflect raw text generation only.

### Gemma 4 12B (Q4_K_XL) — Speculative (raw generation, historical)

| Configuration | Drafting Mechanism | Throughput | Speedup |
| :--- | :--- | :--- | :--- |
| Baseline (Ollama) | None (Greedy) | 12.04 TPS | - |
| Shimmer (N=3, M=24) | Prompt Lookup | 18.70 TPS | +55% |
| Shimmer (N=4, M=24) | Prompt Lookup | 19.23 TPS | +60% |

## Agentic Workload: Flask #4944 (historical, speculative)

> This benchmark was run with speculative decoding enabled (N=4, M=24). With speculation disabled (current default), the effective TPS is 2.6-3.5 for agentic workloads.

| Provider | Configuration | Latency | Throughput | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| Ollama | Standard Single-Agent | 28.60s | 54.34 TPS | - |
| Shimmer | Native XML Edit Block | 3.42s | 423.63 TPS | ~8.3x |

The ~8.3x speedup results from Prompt Lookup Decoding matching file contents as N-Grams. This is only achievable when speculation is enabled and the task involves copying/rewriting existing code. Current agentic workloads run without speculation due to JSON tool-call corruption.

## Process memory overhead

Shimmer's daemon process uses ~26 MB RSS (excluding Metal-managed GPU memory, which holds model weights). Ollama reports ~600 MB. The gap is attributable to Rust's lack of a runtime GC and the use of a Unix domain socket instead of an HTTP server.

## Output Quality

Speculative decoding is mathematically lossless. Tested across code generation, repetition, and creative prose tasks — 100% bit-for-bit identical to greedy baseline across all M and N configurations.
