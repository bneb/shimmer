# Shimmer — Roadmap

## Current (June 2026)

Agentless pipeline on Gemma 4 12B Coding (Q4_K_M) on M4, 20GB unified memory.

```
72 Rust tests, 0 clippy warnings, 10 Python tests
CTX_SIZE=16384, prompt budget 48K chars
Speculation OFF (corrupts structured output)
```

## Pipeline

```
Issue → keyword extraction → rg grep → file ranking → top-5 files
→ agentless prompt with line-annotated snippets
→ model generates XML <edit> blocks → GPU noise strip
→ extract_patch → apply_fuzzy_edit (exact → normalized → difflib)
→ trailing-line preservation → autopep8 → ast.parse
→ RALPH syntax retry (300s) → test-feedback RALPH (300s)
```

## Results (June 2026)

Hybrid mode (context + up to 3 tools, exact-match-only), seed=42, 3 instances:

| Instance | Applied | Wall | Tools | Assessment |
|----------|---------|------|-------|------------|
| django-11099 | 2/2 | 137s | 0 | Correct fix (exact match, no tools needed) |
| django-13551 | 1/1 | 350s | 1 | Wrong file edited |
| matplotlib-25498 | 0/0 | 900s | 2 | Timed out investigating |

**1/3 correct fix, 2/3 applied edits.** Exact-match-only filtering rejected
3 fuzzy matches that earlier pipeline versions would have silently accepted.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| XML edit blocks | Format the model was trained on |
| Greedy sampling (temp=0) | Required for reliable tool-call detection |
| Speculation OFF | N-gram drafts corrupt structured output |
| Append semantics | Compatible with speculation; avoids position-tracking drift |
| Validators at turn boundaries | Prevents infinite loops from mid-thought interruption |
| Normalized matching | Handles 12B Q4 transcription errors |
| Difflib anchor matching | Finds edit location when model can't reproduce exact lines |
| Trailing-line preservation | Keeps scope intact when difflib extends past search |

## Known Issues

- **Test pass rates unverified** — need Python 3.10/3.11 environment for Django instances
- **Docstring deletion** — model sometimes deletes docstrings in difflib matches
- **GPU hangs** — Metal GPU memory not fully freed by SIGKILL; needs system reboot
- **12B Q4 ceiling** — flat-function edits reliable, nested-structure edits unreliable
