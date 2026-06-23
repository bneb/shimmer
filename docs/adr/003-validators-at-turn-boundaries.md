# ADR-003: Validators fire at turn boundaries, not mid-generation

## Status
Accepted (2026-06-21)

## Context
Early versions of the blind edit blocker and TDD enforcement scanned for
partial `<edit` tags in the interceptor buffer. With Gemma 4's
`<channel|thought>` reasoning, this interrupted the model mid-thought,
causing infinite loops where the model would start an edit, get blocked,
restart, and get blocked again.

## Decision
All 6 validators fire at turn boundaries only — after `</edit>` closure or
end-of-generation. The blind edit blocker and TDD enforcement use
`edit_tag_closed` (set when `</edit>` is seen) rather than scanning for
partial open tags.

## Consequences
- Eliminated infinite thought-interruption loops. The model completes its
  reasoning before validators intervene.
- Validators see complete edit blocks with full search/replace content,
  enabling accurate verification.
- The model can waste tokens on a long thought chain before being blocked.
  Acceptable trade-off given the loop elimination.
- Rapid detection of blind edits is slightly slower (wait for `</edit>`
  rather than `<edit`). Measured impact is negligible — most blind edits
  complete their block within a few tokens.
