#!/usr/bin/env python3
"""
SWE-bench Evaluation Script (Ollama Baseline).

This module automates the setup and evaluation of a SWE-bench task (Flask bug #4944) 
using an Ollama-backed LLM. It clones the repository, checks out a specific commit, 
prompts the model for an XML-formatted patch, applies it, and runs a verification script.

Example usage:
    python swe_bench_ollama.py
"""

import os
import subprocess
import time

FLASK_CLONE_DIR = "/Users/kevin/projects/flask_swe_5014"
BASE_COMMIT = "7ee9ceb71e868944a46e1ff00b506772a53a4f1d"

PROMPT = """
Things do not work correctly if a Blueprint is given an empty name (e.g. #4944).
It would be helpful if a `ValueError` was raised when trying to do that.

Here is the relevant part of `src/flask/blueprints.py`:
```python
        super().__init__(
            import_name=import_name,
            static_folder=static_folder,
            static_url_path=static_url_path,
            template_folder=template_folder,
            root_path=root_path,
        )

        if "." in name:
            raise ValueError("'name' may not contain a dot '.' character.")

        self.name = name
```

To fix this, please output an XML edit block to modify `src/flask/blueprints.py`. Use the following exact format:
<edit file="src/flask/blueprints.py">
<search>
[exact lines of code to replace]
</search>
<replace>
[new lines of code]
</replace>
</edit>
"""

def setup_repo():
    """
    Sets up the target repository for SWE-bench evaluation.
    
    Clones the Flask repository if missing, checks out the base buggy commit,
    and creates both a bug prompt file and a hidden test verification script.
    """
    print(f"Setting up Flask repository at {FLASK_CLONE_DIR}...")
    if not os.path.exists(FLASK_CLONE_DIR):
        subprocess.run(["git", "clone", "https://github.com/pallets/flask.git", FLASK_CLONE_DIR])
    
    subprocess.run(["git", "checkout", BASE_COMMIT], cwd=FLASK_CLONE_DIR)
    
    with open(os.path.join(FLASK_CLONE_DIR, "bug_prompt.txt"), "w") as f:
        f.write(PROMPT)
    
    # Add the hidden test script
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
        
    print("Repository is ready at the buggy commit (7ee9ceb7).")

import requests
import json

import threading

class VRAMTracker:
    """
    Estimates VRAM/memory usage of the running Ollama process during execution.
    """
    def __init__(self):
        self.peak_vram = 0.0
        self.running = False
        self.thread = None

    def _poll(self):
        while self.running:
            try:
                pids = subprocess.check_output(["pgrep", "-f", "ollama"]).decode().strip().split('\n')
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

def run_evaluation():
    """
    Executes the SWE-bench evaluation workflow via Ollama.
    
    Sends the prompt to the Ollama API, streams the response, parses the resulting 
    XML edit block, applies the patch to the target file, and runs the hidden 
    verification script. Prints benchmark metrics including TPS and memory usage.
    """
    print("Running SWE task evaluation via Ollama...")
    
    vram_tracker = VRAMTracker()
    vram_tracker.start()
    start_time = time.time()
    
    full_prompt = f"cd {FLASK_CLONE_DIR} && " + PROMPT
    
    req = {
        "model": "gemma4:12b",
        "prompt": full_prompt,
        "stream": True
    }
    
    tokens = 0
    response_text = ""
    try:
        with requests.post("http://localhost:11434/api/generate", json=req, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        tokens += 1
                        token_str = data["response"]
                        response_text += token_str
                        print(token_str, end="", flush=True)
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")
        
    end_time = time.time()
    peak_vram = vram_tracker.stop()
    
    print("\n\n--- Execution Complete ---")
    
    # Extract edit blocks and execute them
    import re
    edit_blocks = re.findall(r'<edit file="(.*?)">.*?<search>(.*?)</search>.*?<replace>(.*?)</replace>.*?</edit>', response_text, re.DOTALL)
    for file_path, search, replace in edit_blocks:
        print(f"\\nApplying edit to {file_path}...")
        full_path = os.path.join(FLASK_CLONE_DIR, file_path)
        with open(full_path, 'r') as f:
            content = f.read()
        search_str = search.strip('\\n')
        replace_str = replace.strip('\\n')
        if search_str in content:
            content = content.replace(search_str, replace_str)
            with open(full_path, 'w') as f:
                f.write(content)
            print("Edit applied successfully.")
        else:
            print("Edit failed: Search string not found.")
        
    # Run the verify script
    import sys
    test_result = subprocess.run(
        [sys.executable, "verify.py"],
        cwd=FLASK_CLONE_DIR,
        capture_output=True,
        text=True
    )
    
    passed = test_result.returncode == 0
    duration = end_time - start_time
    tps = tokens / duration if duration > 0 else 0
    
    print(f"\\n--- Benchmark Results ---")
    print(f"Latency: {duration:.2f}s")
    print(f"Throughput (TPS): {tps:.2f} tokens/s")
    print(f"Total Tokens: {tokens}")
    print(f"Peak VRAM: {peak_vram:.2f} MB")
    print(f"Hidden Test Passed: {passed}")
    if not passed:
        print(f"Test output:\\n{test_result.stdout}\\n{test_result.stderr}")

if __name__ == "__main__":
    setup_repo()
    run_evaluation()
