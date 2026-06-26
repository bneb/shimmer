#!/usr/bin/env python3
"""Shimmer SWE-bench Evaluation Harness.

Usage:
    python3 benchmarks/agentic/generate_swe_bench.py --hybrid --sample 3 --seed 42 \\
        --model models/gemma4-12b.gguf --sample-cfg "temp=0.0,topk=0,repp=1.0"
"""

import argparse, os, sys

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models", "gemma4-12b.gguf")
DEFAULT_SAMPLE_CFG = "temp=0.0,topk=0,repp=1.0"


def main():
    parser = argparse.ArgumentParser(description="Shimmer SWE-bench Evaluation Harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--sample", type=int, default=3, help="Number of instances")
    parser.add_argument("--sample-cfg", default=DEFAULT_SAMPLE_CFG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--agentless", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--instance", type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.model = os.path.abspath(args.model)
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return 1

    from harness.cli import run_eval
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
