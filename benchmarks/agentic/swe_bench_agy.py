#!/usr/bin/env python3
import os
import subprocess
import time

FLASK_CLONE_DIR = "/tmp/flask_swe_agy"
BASE_COMMIT = "7ee9ceb71e868944a46e1ff00b506772a53a4f1d"

PROMPT = f"""
There is a bug in the Flask repository where things do not work correctly if a Blueprint is given an empty name (e.g. #4944).
It would be helpful if a `ValueError` was raised when trying to do that.

The code is located in `src/flask/blueprints.py` inside the repository at {FLASK_CLONE_DIR}.

Please explore the code, find where Blueprints are initialized, and fix the bug so that passing an empty name or a name containing a dot raises a ValueError.
You MUST use your tools to edit the file directly to fix the bug.
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
    print("Repository is ready.", flush=True)

def wait_for_shimmer():
    print("Waiting for Shimmer REST API on port 8080...", flush=True)
    import socket
    for _ in range(60):
        try:
            with socket.create_connection(("localhost", 8080), timeout=1):
                print("Shimmer API is ready!", flush=True)
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            pass
        time.sleep(1)
    print("Timeout waiting for Shimmer API.", flush=True)
    return False

def run_evaluation():
    print("Starting Shimmer API Server...", flush=True)
    shimmer_log = open("shimmer.log", "w")
    shimmer_proc = subprocess.Popen(
        ["cargo", "run", "--release", "--bin", "shimmer", "--", "--main-model", "/Users/kevin/projects/shimmer/models/qwen_q3.gguf", "--serve"],
        stdout=shimmer_log,
        stderr=subprocess.STDOUT,
        cwd="/Users/kevin/projects/shimmer"
    )
    
    try:
        if not wait_for_shimmer():
            return
            
        print("Running agy evaluation. Writing output to agy_eval.log...", flush=True)
        start_time = time.time()
        
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = "http://localhost:8080/v1"
        env["OPENAI_API_KEY"] = "dummy"
        
        agy_cmd = [
            "agy", "--dangerously-skip-permissions", 
            "--model", "openai/default",
            "-p", PROMPT
        ]
        
        with open("agy_eval.log", "w") as log_file:
            process = subprocess.Popen(
                agy_cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/Users/kevin/projects/shimmer"
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
            
    finally:
        print("Killing Shimmer API Server...", flush=True)
        shimmer_proc.terminate()
        shimmer_proc.wait()

if __name__ == "__main__":
    setup_repo()
    run_evaluation()
