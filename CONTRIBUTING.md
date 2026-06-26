# Contributing

## Development Setup

```sh
git clone https://github.com/bneb/shimmer.git
cd shimmer
cargo build
cargo test
```

Requirements: Rust toolchain (latest stable), macOS with Metal support (Apple
Silicon recommended).

## Code Style

- Format with `rustfmt` (run `cargo fmt`).
- Lint with `clippy` — zero warnings required (`cargo clippy`).
- Follow the quality gates documented in `AGENTS.md`: file size under 400 LOC,
  function bodies under 32 lines, nesting at most 3 levels.
- No `unwrap()` or `expect()` in new code. Use `Result` propagation.
- Mark items `pub` only when they form a cross-module API; prefer
  `pub(crate)`.

## PR Process

1. Open an issue describing the problem or feature before starting work.
2. Discuss the approach with maintainers in the issue thread.
3. Submit a focused PR that addresses one concern. Keep diffs small.
4. Every new function must include a test. Run `cargo test` before pushing.
5. Ensure `cargo clippy -- -D warnings` passes with no new warnings.
6. Reference the issue number in the PR description.
