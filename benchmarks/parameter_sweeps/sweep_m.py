"""
Parameter Sweep Script for Shimmer's Speculative Decoding.

This module performs a fine-grained grid search over n-gram sizes and maximum 
draft token limits for the Shimmer engine. It modifies the underlying Rust source 
code (`speculative.rs`), rebuilds the project, and records throughput (TPS) to 
determine the optimal hyperparameter configuration.

Example usage:
    python sweep_m.py
"""

import os
import re
import subprocess
import time

NGRAM_SIZES = [3, 4]
MAX_DRAFT_TOKENS = list(range(20, 29))

SPECULATIVE_RS_PATH = "src/speculative.rs"

def modify_constants(ngram_size, draft_tokens):
    """
    Modifies hyperparameter constants directly in the Shimmer Rust source code.
    
    Args:
        ngram_size (int): The n-gram length to use for speculative matching.
        draft_tokens (int): The maximum number of draft tokens to propose.
    """
    with open(SPECULATIVE_RS_PATH, "r") as f:
        content = f.read()
    
    content = re.sub(r"let ngram_size = \d+;", f"let ngram_size = {ngram_size};", content)
    content = re.sub(r"pub const MAX_DRAFT_TOKENS: usize = \d+;", f"pub const MAX_DRAFT_TOKENS: usize = {draft_tokens};", content)
    
    with open(SPECULATIVE_RS_PATH, "w") as f:
        f.write(content)

def run_benchmark():
    """
    Compiles the Shimmer project and runs a speculative benchmark.
    
    Returns:
        float: The extracted throughput (TPS) from the Shimmer benchmark run,
               or 0.0 if the run failed or the output couldn't be parsed.
    """
    # Build first
    subprocess.run(["cargo", "build", "--release"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Run benchmark
    result = subprocess.run(["cargo", "run", "--release", "--", "--speculative"], capture_output=True, text=True)
    
    # Parse TPS
    match = re.search(r"Performance:\s+([\d.]+)\s+TPS", result.stdout)
    if match:
        return float(match.group(1))
    return 0.0

def main():
    """
    Main execution entrypoint for the parameter sweep.
    
    Iterates through all combinations of NGRAM_SIZES and MAX_DRAFT_TOKENS,
    evaluates their performance, prints the results in a tabular format, 
    and finally restores the source code to the best-performing configuration.
    """
    print("Starting Fine-Grained Parameter Sweep...")
    print(f"{'N-Gram':<10} | {'Draft Tokens':<15} | {'TPS':<10}")
    print("-" * 40)
    
    best_tps = 0
    best_params = None
    
    for n in NGRAM_SIZES:
        for m in MAX_DRAFT_TOKENS:
            modify_constants(n, m)
            tps = run_benchmark()
            print(f"{n:<10} | {m:<15} | {tps:<10.2f}", flush=True)
            
            if tps > best_tps:
                best_tps = tps
                best_params = (n, m)
                
    print("-" * 40)
    print(f"Best: N-Gram={best_params[0]}, Draft Tokens={best_params[1]} -> {best_tps} TPS")
    
    # Restore to best
    modify_constants(best_params[0], best_params[1])
    print("Restored speculative.rs to best parameters.")

if __name__ == "__main__":
    main()
