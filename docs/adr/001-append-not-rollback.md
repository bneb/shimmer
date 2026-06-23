# ADR-001: Append semantics for tool output injection

## Status
Accepted (2026-06-19)

## Context
When the model generates a tool call, the engine must inject tool output into
the conversation. Two approaches exist: KV cache rollback (undo the tool call
tokens, inject output, re-generate) or append (keep tool call tokens in
history, inject output as a new user turn).

## Decision
Use append semantics. Tool output is injected as a synthetic user turn after
the model's tool call tokens. No KV cache rollback is performed.

## Consequences
- Compatible with speculative decoding — no save/restore of KV cache state
  needed, avoiding token-level rollback logic.
- No position-tracking drift between `n_cur` and `history.len()` — history
  length is the single source of truth.
- Tool call tokens permanently consume context window space. Mitigated by
  KV cache compaction which evicts old tool output pages.
- The model sees its own tool calls repeated in history, which can reinforce
  loops. Mitigated by the insanity detector (validator #6).
