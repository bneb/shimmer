# ADR-002: Greedy sampling required for tool detection

## Status
Accepted (2026-06-19)

## Context
The `ToolInterceptor` detects tool calls by scanning for exact ````json`
markers in the token stream. At temperature > 0, token sampling introduces
variance in these markers, causing missed detections.

## Decision
All agentic workloads use greedy sampling (`temp=0.0`, `topk=0`, `repp=1.0`).
The `--sample` flag exists for non-tool use cases but tool detection is
unreliable with non-greedy sampling.

## Consequences
- 100% reliable tool-call detection. No missed detections from token variance.
- Deterministic outputs simplify debugging and benchmarking.
- No output diversity — the model always produces the same response for the
  same prompt. Acceptable for SWE-bench where correctness matters over
  creativity.
- Higher repetition-loop risk since the model cannot sample out of degenerate
  states. The insanity detector partially mitigates this.
