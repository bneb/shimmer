#!/usr/bin/env python3
"""
SWE-bench Evaluation Script (Multi-Task Sample).

This script demonstrates how to run a sample of multiple SWE-bench exercises
in a single benchmark run. It iterates through a list of predefined tasks,
sets up the repository for each, runs the agent, and aggregates the results.
"""

import os
import subprocess
import time
import re
import threading
import sys

# Define a sample of 3 tasks (we use Flask for all 3 here to avoid excessive cloning, 
# but they represent 3 distinct SWE-bench exercises).
TASKS = [
    {
        "id": "flask-4944",
        "repo_url": "https://github.com/pallets/flask.git",
        "base_commit": "7ee9ceb71e868944a46e1ff00b506772a53a4f1d",
        "prompt": "Things do not work correctly if a Blueprint is given an empty name (e.g. #4944).\nIt would be helpful if a `ValueError` was raised when trying to do that in `src/flask/blueprints.py`.",
        "test_script": "import sys\nsys.path.insert(0, './src')\nfrom flask.blueprints import Blueprint\ntry:\n    Blueprint('', __name__)\n    sys.exit(1)\nexcept ValueError:\n    sys.exit(0)\n",
        "edit_file": "src/flask/blueprints.py"
    },
    {
        "id": "flask-app-mock",
        "repo_url": "https://github.com/pallets/flask.git",
        "base_commit": "7ee9ceb71e868944a46e1ff00b506772a53a4f1d",
        "prompt": "In `src/flask/app.py`, the `Flask.__init__` method needs a new default attribute `self.custom_greeting = 'Hello Flask!'`. Please add it near the end of the `__init__` method.",
        "test_script": "import sys\nsys.path.insert(0, './src')\nfrom flask.app import Flask\napp = Flask(__name__)\nif getattr(app, 'custom_greeting', None) == 'Hello Flask!':\n    sys.exit(0)\nelse:\n    sys.exit(1)\n",
        "edit_file": "src/flask/app.py"
    },
    {
        "id": "flask-config-mock",
        "repo_url": "https://github.com/pallets/flask.git",
        "base_commit": "7ee9ceb71e868944a46e1ff00b506772a53a4f1d",
        "prompt": "In `src/flask/config.py`, please add a `clear_all(self)` method to the `Config` class. It should simply call `self.clear()`.",
        "test_script": "import sys\nsys.path.insert(0, './src')\nfrom flask.config import Config\nc = Config('/')\nc['A'] = 1\ntry:\n    c.clear_all()\n    if len(c) == 0:\n        sys.exit(0)\n    else:\n        sys.exit(1)\nexcept AttributeError:\n    sys.exit(1)\n",
        "edit_file": "src/flask/config.py"
    }
]

CLONE_DIR = "/tmp/swe_bench_workspace"

class VRAMTracker:
    def __init__(self):
        self.peak_vram = 0.0
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

def setup_task(task):
    print(f"\\n[{task['id']}] Setting up repository...")
    if not os.path.exists(CLONE_DIR):
        subprocess.run(["git", "clone", task["repo_url"], CLONE_DIR])
    else:
        # If repo url changes, we'd delete and reclone. Assuming same for simplicity.
        subprocess.run(["git", "fetch", "origin"], cwd=CLONE_DIR)
        
    subprocess.run(["git", "reset", "--hard", task["base_commit"]], cwd=CLONE_DIR)
    subprocess.run(["git", "clean", "-fd"], cwd=CLONE_DIR)
    
    # Write the hidden test script
    with open(os.path.join(CLONE_DIR, "verify.py"), "w") as f:
        f.write(task["test_script"])

def run_task(task):
    setup_task(task)
    
    print(f"[{task['id']}] Running Shimmer evaluation...")
    vram_tracker = VRAMTracker()
    vram_tracker.start()
    start_time = time.time()
    
    # Format the prompt with XML instructions
    instruction = f"""{task['prompt']}
To fix this, output an XML edit block exactly like this:
<edit file="{task['edit_file']}">
<search>
[exact lines to replace]
</search>
<replace>
[new lines]
</replace>
</edit>
"""
    full_prompt = instruction
    
    response_text = ""
    tokens = 0
    
    for attempt in range(3):
        process = subprocess.Popen(
            [os.path.abspath("target/release/shimmer"), "--prompt", full_prompt],
            cwd=CLONE_DIR,
            stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1
        )
        for line in process.stdout:
            response_text += line
            tokens += len(re.findall(r"\w+|[^\w\s]", line))
        process.wait()
        with open("agent_responses.log", "a") as f:
            f.write(f"=== {task['id']} ===\n{response_text}\n=================\n")
        
        # If it didn't crash, we evaluate
        if process.returncode == 0:
            break
        print(f"[{task['id']}] Attempt {attempt+1} crashed. Retrying...")
    
    end_time = time.time()
    peak_vram = vram_tracker.stop()
    
    # Apply edits
    edit_blocks = re.findall(r'<edit file="(.*?)">.*?<search>(.*?)</search>.*?<replace>(.*?)</replace>.*?</edit>', response_text, re.DOTALL)
    for file_path, search, replace in edit_blocks:
        full_path = os.path.join(CLONE_DIR, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f: content = f.read()
            if search.strip('\\n') in content:
                content = content.replace(search.strip('\\n'), replace.strip('\\n'))
                with open(full_path, 'w') as f: f.write(content)
                
    # Run test
    test_result = subprocess.run([sys.executable, "verify.py"], cwd=CLONE_DIR, capture_output=True, text=True)
    passed = (test_result.returncode == 0)
    duration = end_time - start_time
    
    return {
        "id": task["id"],
        "passed": passed,
        "latency": duration,
        "tps": tokens / duration if duration > 0 else 0,
        "vram": peak_vram
    }

if __name__ == "__main__":
    results = []
    print("Starting Multi-Task SWE-bench Evaluation...")
    for t in TASKS:
        results.append(run_task(t))
        
    print("\\n\\n=== FINAL BENCHMARK RESULTS ===")
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Total Score: {passed_count}/{len(TASKS)} ({(passed_count/len(TASKS))*100:.1f}%)")
    print(f"{'Task ID':<15} | {'Status':<6} | {'Latency':<8} | {'TPS':<8} | {'VRAM (MB)'}")
    print("-" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<15} | {status:<6} | {r['latency']:<8.2f}s | {r['tps']:<8.2f} | {r['vram']:.2f}")
