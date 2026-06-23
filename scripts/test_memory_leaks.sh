#!/usr/bin/env bash
set -e

echo "======================================"
echo " Running AddressSanitizer & LeakSanitizer "
echo "======================================"

# Ensure nightly is installed
rustup toolchain install nightly

# Run tests with sanitizers
export RUSTFLAGS="-Zsanitizer=address -Zsanitizer=leak"
export ASAN_OPTIONS="detect_leaks=1"
cargo +nightly test -Zbuild-std --target aarch64-apple-darwin -- --nocapture

echo "Memory leak test completed successfully!"
