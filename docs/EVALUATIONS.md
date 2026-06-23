# Shimmer Quality Evaluations

To ensure that Shimmer's optimizations do not degrade the output quality of the underlying model, we perform rigorous evaluations against a greedy baseline.

## 1. Speculative Decoding Verification (Loss-less Check)

Prompt Lookup Decoding (PLD) and other speculative decoding methods must be mathematically loss-less when `temperature=0.0`. This means that regardless of the N-Gram overlap size ($N$) or the maximum draft window ($M$), the generated tokens must be bit-for-bit identical to standard auto-regressive generation.

**Methodology:**
We run a set of deterministic prompts against the standard `llama.cpp` inference engine, and then against Shimmer using various $(N, M)$ configurations.

**Results:**
| Prompt Category | Baseline | Shimmer (N=3, M=24) | Shimmer (N=4, M=24) | Shimmer (N=3, M=27) | Match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Code Structure** | `Box::new(TreeNode::Node { ...` | `Box::new(TreeNode::Node { ...` | `Box::new(TreeNode::Node { ...` | `Box::new(TreeNode::Node { ...` | ✅ |
| **Repetitive Text** | `Tick, tick, tick, the seconds fly...` | `Tick, tick, tick, the seconds fly...` | `Tick, tick, tick, the seconds fly...` | `Tick, tick, tick, the seconds fly...` | ✅ |
| **Creative Prose** | `Electric violet hues bled...` | `Electric violet hues bled...` | `Electric violet hues bled...` | `Electric violet hues bled...` | ✅ |

*Result: Across all tested $(N, M)$ configurations on raw text prompts, speculative output was bit-for-bit identical to the greedy baseline. Note: this applies to raw text generation without tool calls. For agentic workloads with chat templates, speculation is disabled (see [Architecture](ARCHITECTURE.md)).*

## 2. Tool Execution Accuracy

Shimmer executes subprocess tool calls natively during inference. We evaluate whether the model correctly adheres to the syntax when fine-tuned, and whether Shimmer accurately intercepts and halts generation.

**Result:**
The `ToolInterceptor` correctly detects JSON tool calls in ````json` blocks, executes the system command (e.g., `rg`, `cat`), and appends the output as a user turn using append semantics (no KV cache rollback). The model successfully resumes generation.

## 3. Edit Validator Accuracy

Shimmer's 6 validators guard against common model failure modes: editing without investigation, hallucinated file paths, incorrect search strings, and syntax errors. All validators operate at turn boundaries using append semantics.

**Validated on 30-instance SWE-bench run (Gemma 4 12B):**
| Validator | Result |
|-----------|--------|
| Blind edit blocker | Caught 12 blind edits; 0 false positives |
| Path blocker | Caught 4 hallucinated paths |
| Search verifier | Caught search mismatches on 8 instances; dedup prevents retry loops |
| Syntax checker | 100% syntactically valid patches across all 11 produced |
| Insanity detector | Caught repeated tool calls on 3 instances |
| TDD enforcement | Not tested (run with `--no-verify`) |

All validators are bypassed in `--no-tools` (agentless) mode.

## 4. Prompt Structure

On `django__django-10914`, the position of the edit example relative to "Begin now." determined file selection:

| Prompt structure | File selected | Outcome |
|-----------------|--------------|---------|
| Edit example immediately before "Begin now." | `django/db/utils.py` | Wrong module |
| "INVESTIGATE BEFORE EDITING" before example | `django/conf/global_settings.py` | Correct |

With the investigation-first prompt, the model searched with `rg`, found `FILE_UPLOAD_PERMISSIONS` in `global_settings.py`, read the file, and changed the default from `None` to `0o644`.

## 5. Agentless Mode Evaluation

The agentless pipeline (`--agentless`) was tested on `django__django-10914` end-to-end:

| Component | Result |
|-----------|--------|
| Keyword extraction | Correctly extracted `FILE_UPLOAD_PERMISSIONS` from issue |
| File localization | `global_settings.py` ranked #2 (of 953 matched) |
| Prompt assembly | 27K chars (~7K tokens), no OOM, well within budget |
| `--no-tools` flag | 0 tool calls detected; model generated straight through |
| Extraction filter | `Begin now.` anchor correctly excluded prompt examples |
| Model edit | Right file (`global_settings.py`), right name (`FILE_UPLOAD_PERMISSIONS`), right value (`0o644`), wrong line (17 vs 307) |

**Key findings:**
- The pipeline infrastructure works correctly end-to-end — every Python and Rust component functions as designed
- Line placement precision is the main quality gap — the model places settings near the top of the file rather than near related settings
- The `__demo_placeholder__` example guard + `Begin now.` extraction anchor prevent example bleed
- Smart file truncation keeps prompts under 50K chars even with 20+ files inlined
- Agentless TPS (~5.1) is higher than agentic (~2.6-3.5) because there are no tool halts
