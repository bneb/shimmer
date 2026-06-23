#!/usr/bin/env python3
import os
import subprocess
import time
import json
import urllib.request
import urllib.error

FLASK_CLONE_DIR = "/tmp/flask_swe_rest"
BASE_COMMIT = "7ee9ceb71e868944a46e1ff00b506772a53a4f1d"

SYSTEM_PROMPT = """You are an elite, pragmatic AI coding assistant. You provide direct, correct answers without conversational filler.
You have access to a tool execution engine.
To use a tool, output exactly this JSON syntax:
```json
{"name": "tool_name", "arguments": ["arg1", "arg2"]}
```

Example:
```json
{"name": "rg", "arguments": ["500 error", "logs/"]}
```

Wait for the system to inject the results before continuing your answer. For mutating commands, you must await user confirmation. NEVER chain commands with pipes (|) or redirects (>). Use one tool at a time.
Available tools: rg, fd, cat, git, sed"""

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

def wait_for_shimmer():
    print("Waiting for Shimmer REST API on port 8080...", flush=True)
    import socket
    for _ in range(60):
        try:
            with socket.create_connection(("localhost", 8080), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            pass
        time.sleep(1)
    return False

def run_agent_loop():
    print("\\n--- Starting single REST orchestrator request ---", flush=True)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT}
    ]
    
    req_data = json.dumps({
        "model": "qwen",
        "messages": messages,
    }).encode("utf-8")
    
    req = urllib.request.Request("http://localhost:8080/v1/chat/completions", data=req_data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            res_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"API Error: {e}")
        return False
        
    choice = res_data["choices"][0]
    message = choice["message"]
    
    content = message.get("content", "")
    print(f"Final Agent Output:\\n{content}", flush=True)
    return True

if __name__ == "__main__":
    setup_repo()
    print("Starting Shimmer API Server...", flush=True)
    shimmer_log = open("shimmer.log", "w")
    shimmer_proc = subprocess.Popen(
        ["/Users/kevin/projects/shimmer/target/release/shimmer", "--main-model", "/Users/kevin/.ollama/models/blobs/qwen3-coder-30b-a3b-Q4_K_M.gguf", "--serve"],
        stdout=shimmer_log,
        stderr=subprocess.STDOUT,
        cwd=FLASK_CLONE_DIR
    )
    
    try:
        if wait_for_shimmer():
            print("Shimmer API is ready! Sending request...", flush=True)
            start_time = time.time()
            run_agent_loop()
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
