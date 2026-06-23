#!/bin/bash
echo "Running tests with AddressSanitizer and LeakSanitizer..."
# Requires Rust nightly
export RUSTFLAGS="-Zsanitizer=address"
export ASAN_OPTIONS="detect_leaks=1"
cargo +nightly test --target aarch64-apple-darwin
