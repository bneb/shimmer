#!/usr/bin/env python3
import os
import subprocess
import time

FLASK_CLONE_DIR = "/tmp/flask_swe_native"
BASE_COMMIT = "7ee9ceb71e868944a46e1ff00b506772a53a4f1d"

PROMPT = f"""
There is a bug in the Flask repository where things do not work correctly if a Blueprint is given an empty name (e.g. #4944).
It would be helpful if a `ValueError` was raised when trying to do that.

The code is located in `src/flask/blueprints.py`.

Please explore the code, find where Blueprints are initialized, and fix the bug so that passing an empty name or a name containing a dot raises a ValueError.
You MUST use your tools (rg, cat, sed) to edit the file directly to fix the bug.
"""

def setup_repo():
    print(f"Setting up Flask repository at {FLASK_CLONE_DIR}...")
    if not os.path.exists(FLASK_CLONE_DIR):
        subprocess.run(["git", "clone", "https://github.com/pallets/flask.git", FLASK_CLONE_DIR])
    subprocess.run(["git", "reset", "--hard", BASE_COMMIT], cwd=FLASK_CLONE_DIR)
    subprocess.run(["git", "clean", "-fd"], cwd=FLASK_CLONE_DIR)
    
    hidden_test = """import sys
sys.path.insert(0, './src')
from flask.blueprints import Blueprint

try:
    Blueprint("", __name__)
    print("FAILED: Did not raise ValueError")
    sys.exit(1)
except ValueError:
    print("PASSED: Raised ValueError")
    sys.exit(0)
except Exception as e:
    print(f"FAILED: Raised wrong exception {e}")
    sys.exit(1)
"""
    with open(os.path.join(FLASK_CLONE_DIR, "verify.py"), "w") as f:
        f.write(hidden_test)

if __name__ == "__main__":
    setup_repo()
    print("Starting Shimmer Native Agent...", flush=True)
    
    start_time = time.time()
    
    # We pass the prompt directly to Shimmer without --speculative
    full_prompt = f"cd {FLASK_CLONE_DIR} && " + PROMPT
    
    process = subprocess.Popen(
        ["/Users/kevin/projects/shimmer/target/release/shimmer", "--main-model", "/Users/kevin/projects/shimmer/models/qwen_q3.gguf", "--prompt", full_prompt],
        cwd=FLASK_CLONE_DIR
    )
    process.wait()
    
    end_time = time.time()
            
    print("\\n\\n--- Execution Complete ---", flush=True)
    import sys
    test_result = subprocess.run(
        [sys.executable, "verify.py"],
        cwd=FLASK_CLONE_DIR,
        capture_output=True,
        text=True
    )
    passed = test_result.returncode == 0
    duration = end_time - start_time
    print(f"\\n--- Benchmark Results ---", flush=True)
    print(f"Latency: {duration:.2f}s", flush=True)
    print(f"Hidden Test Passed: {passed}", flush=True)
    if not passed:
        print(f"Test output:\\n{test_result.stdout}\\n{test_result.stderr}", flush=True)

