"""
Parameter sweep script for Shimmer PLD (Parallel Language Decoding).
Evaluates performance (TPS) across various n-gram and draft sizes.

Example usage:
    python sweep.py --query "Explain the history of the CPU."
"""

import subprocess
import re
import argparse
from typing import Tuple, List

NGRAM_SIZES = [1, 2, 3, 4, 5, 6]
DRAFT_SIZES = [4, 8, 16, 24, 32]

def run_benchmark(ngram: int, draft: int, query: str = "Explain the history of the CPU.") -> float:
    """
    Runs the Shimmer benchmark with specified n-gram and draft sizes.

    Args:
        ngram: The n-gram size parameter.
        draft: The draft size parameter.
        query: The prompt to evaluate.

    Returns:
        float: The tokens per second (TPS) achieved during the run, or 0.0 if failed.
    """
    cmd = [
        "cargo", "run", "--release", "--", 
        "--speculative", 
        "--ngram-size", str(ngram), 
        "--draft-size", str(draft),
        "--prompt", query
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        match = re.search(r"Tokens per second:\s*([\d.]+)", result.stdout)
        if not match:
            # Fallback to the old output format if needed
            match = re.search(r"Performance:\s+([\d.]+)\s+TPS", result.stdout)
            
        if match:
            return float(match.group(1))
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
    
    return 0.0

import threading
import time

class VRAMTracker:
    def __init__(self):
        self.peak_vram = 0.00
        self.running = False
        self.thread = None

    def _poll(self):
        while self.running:
            try:
                pids = subprocess.check_output(["pgrep", "-f", "target/release/shimmer"]).decode().strip().split('\n')
                if pids and pids[0]:
                    pid = pids[0]
                    output = subprocess.check_output(["ps", "-p", pid, "-o", "rss="]).decode().strip()
                    current = int(output) / 1024.0 # MB
                    if current > self.peak_vram:
                        self.peak_vram = current
            except Exception:
                pass
            time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_vram

def main():
    """
    Main entry point for parameter sweeping.
    Builds the project in release mode and tests predefined combinations 
    of n-gram and draft sizes, reporting the optimal configuration.
    """
    parser = argparse.ArgumentParser(description="Sweep Shimmer PLD parameters")
    parser.add_argument("--query", type=str, default="Explain the history of the CPU.", help="Prompt to evaluate")
    args = parser.parse_args()

    print(f"Building Shimmer in release mode...")
    subprocess.run(["cargo", "build", "--release"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"\nStarting Parameter Sweep for query: '{args.query}'")
    print(f"{'N-Gram':<10} | {'Draft Size':<15} | {'TPS':<10} | {'Peak VRAM (MB)':<15}")
    print("-" * 55)
    
    best_tps = 0.0
    best_params = (0, 0)
    best_vram = 0.0
    
    for n in NGRAM_SIZES:
        for m in DRAFT_SIZES:
            tracker = VRAMTracker()
            tracker.start()
            tps = run_benchmark(n, m, args.query)
            peak_vram = tracker.stop()
            print(f"{n:<10} | {m:<15} | {tps:<10.2f} | {peak_vram:<15.2f}", flush=True)
            
            if tps > best_tps:
                best_tps = tps
                best_params = (n, m)
                best_vram = peak_vram
            
            time.sleep(1)
                
    print("-" * 55)
    print(f"OPTIMAL CONFIGURATION: N-Gram={best_params[0]}, Draft Size={best_params[1]} -> {best_tps:.2f} TPS (VRAM: {best_vram:.2f} MB)")

if __name__ == "__main__":
    main()
