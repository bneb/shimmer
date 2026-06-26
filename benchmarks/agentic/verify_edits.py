#!/usr/bin/env python3
"""Validate already-applied edits against SWE-bench test suites.

Reads eval_results.json, runs FAIL_TO_PASS + PASS_TO_PASS tests
on each clone, and reports the pass rate.  No model inference needed.
"""

import json
import os
import subprocess
import sys
import time
from datasets import load_dataset

CLONE_BASE = "/tmp/swe_lite"
TIMEOUT = 120  # seconds per test suite


def run_tests(clone_dir, test_list):
    """Run a list of pytest test names. Returns (passed: bool, output: str)."""
    if not test_list:
        return None, "(no tests)"
    cmd = ["python3", "-m", "pytest", "-x", "--timeout=60"] + test_list
    try:
        proc = subprocess.run(
            cmd, cwd=clone_dir, capture_output=True, text=True, timeout=TIMEOUT
        )
        return proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return None, "pytest not found"


def main():
    # Load eval results
    with open("data/eval_results.json") as f:
        data = json.load(f)

    # Load SWE-bench dataset for test metadata
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks = {row["instance_id"]: row for row in ds}

    results = []
    for r in data["results"]:
        iid = r["instance_id"]
        applied = r["edits_applied"]
        clone_dir = os.path.join(CLONE_BASE, iid)

        if applied == 0:
            results.append({**r, "tests_pass": None, "verify_note": "no edits applied"})
            continue

        if not os.path.isdir(clone_dir):
            results.append({**r, "tests_pass": None, "verify_note": "clone not found"})
            continue

        task = tasks.get(iid)
        if not task:
            results.append({**r, "tests_pass": None, "verify_note": "task not in dataset"})
            continue

        # FAIL_TO_PASS and PASS_TO_PASS are JSON strings in SWE-bench
        ftp_raw = task.get("FAIL_TO_PASS", "[]")
        ftp = json.loads(ftp_raw) if isinstance(ftp_raw, str) else ftp_raw
        ftp = ftp[:10]  # cap at 10 tests
        ftp_passed, ftp_log = run_tests(clone_dir, ftp)

        ptp_raw = task.get("PASS_TO_PASS", "[]")
        ptp = json.loads(ptp_raw) if isinstance(ptp_raw, str) else ptp_raw
        ptp = ptp[:10]
        ptp_passed, ptp_log = run_tests(clone_dir, ptp)

        if ftp_passed is None:
            passed = None
            note = "no pytest"
        elif not ftp:
            passed = None
            note = "no FAIL_TO_PASS tests"
        else:
            passed = ftp_passed and (ptp_passed is not False)

        print(f"  {'PASS' if passed else 'FAIL' if passed is False else 'N/A'}"
              f"  {iid:45s}  ftp={'✓' if ftp_passed else '✗' if ftp_passed is False else '?'}"
              f"  ptp={'✓' if ptp_passed else '✗' if ptp_passed is False else '?'}")

        results.append({
            **r,
            "tests_pass": passed,
            "verify_note": note if passed is None else None,
            "ftp_passed": ftp_passed,
            "ptp_passed": ptp_passed,
        })

    # Summary
    passing = [r for r in results if r.get("tests_pass") is True]
    failing = [r for r in results if r.get("tests_pass") is False]
    unknown = [r for r in results if r.get("tests_pass") is None]

    print(f"\n{'='*60}")
    print(f"Verification complete: {len(results)} instances")
    print(f"  Tests passing:  {len(passing)}/{len(results)}")
    print(f"  Tests failing:  {len(failing)}/{len(results)}")
    print(f"  Unknown:        {len(unknown)}/{len(results)}")

    # Save
    with open("data/eval_results.json", "w") as f:
        data["results"] = results
        json.dump(data, f, indent=2)
    print(f"\nResults saved to data/eval_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
