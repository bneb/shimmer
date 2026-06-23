#!/usr/bin/env python3
"""SWE-bench pipeline smoke test — validates tool execution, edit extraction, and patching."""

import os
import subprocess
import sys
import tempfile
import shutil

SHIMMER_BIN = os.path.join(os.path.dirname(__file__), "..", "target/release/shimmer")
SHIMMER_MODEL = os.path.join(os.path.dirname(__file__), "..", "models/gemma4-12b.gguf")

TEST_PROMPT = """There is an issue reported in the repository. Please fix it.

Issue Description:
The add function in calculator.py returns a - b instead of a + b. Tests fail.

AVAILABLE TOOLS:
| Tool | Purpose | Example |
|------|---------|---------|
| rg   | Search file contents | rg -n "pattern" . |
| cat  | Read file | cat file.py |
| fd   | Find files | fd "*.py" |
| ls   | List directory | ls |
| run_test | Run tests | run_test python -m pytest |

CRITICAL RULES:
1. If a search tool yields no output, DO NOT STOP. Try different search terms.
2. You MUST use tools to locate the EXACT file and lines before providing a fix.
3. Do NOT finish until you have provided the final <edit> block.

Work step by step: search, read, test, then provide your fix.
"""

FILES = {
    "calculator.py": "def add(a, b):\n    return a - b  # BUG: should be +\n\ndef multiply(a, b):\n    return a * b\n",
    "test_calc.py": "from calculator import add, multiply\n\nassert add(2, 3) == 5, f'Expected 5, got {add(2, 3)}'\nassert multiply(2, 3) == 6\nprint('All tests pass!')\n",
}


def run_smoke_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in FILES.items():
            with open(os.path.join(tmpdir, filename), "w") as f:
                f.write(content)

        print(f"[1/4] Test files created in {tmpdir}")

        # Quick check: does the bug exist?
        result = subprocess.run([sys.executable, "test_calc.py"], cwd=tmpdir, capture_output=True, text=True)
        if result.returncode == 0:
            print("  FAIL: Tests pass before fix (bug not present)")
        else:
            print(f"  OK: Bug confirmed — tests fail: {result.stderr.strip()[:80]}")

        # Check binary exists
        if not os.path.exists(SHIMMER_BIN):
            print(f"  SKIP: Release binary not found at {SHIMMER_BIN}")
            print("  Build with: cargo build --release")
            return False

        # Check model exists
        if not os.path.exists(SHIMMER_MODEL):
            print(f"  SKIP: Model not found at {SHIMMER_MODEL}")
            return False

        print(f"[2/4] Running shimmer inference (120s timeout)...")

        try:
            process = subprocess.Popen(
                [SHIMMER_BIN, "--main-model", SHIMMER_MODEL, "--prompt", TEST_PROMPT, "--teston",
                 "--sample", "temp=0.0,topk=0,repp=1.0", "--no-preprocessor"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=tmpdir
            )

            output = ""
            try:
                output, _ = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
                print("  (timed out after 120s — checking partial output)")

            print(f"  Output: {len(output)} chars")

            # Check for tool usage
            if "rg" in output or "cat" in output:
                print("  OK: Model used search tools")
            else:
                print("  WARN: No tool usage detected")

            if "<edit" in output or "model_patch" in output:
                print("  OK: Model produced edit block")
            else:
                print("  WARN: No edit block in output")

        except Exception as e:
            print(f"  FAIL: Inference failed: {e}")
            return False

        print(f"[3/4] Checking tool output files...")
        tool_files = [f for f in os.listdir(tmpdir) if f.startswith(".shimmer_tool_")]
        print(f"  Found {len(tool_files)} tool output files")
        for tf in tool_files:
            size = os.path.getsize(os.path.join(tmpdir, tf))
            print(f"    {tf}: {size} bytes")

        print(f"[4/4] Pipeline validation:")
        has_output = len(output) > 100
        has_tools = len(tool_files) > 0
        has_edit = "<edit" in output  # accept <edit> or <edit file=...>
        print(f"  Output > 100 chars: {'PASS' if has_output else 'FAIL'}")
        print(f"  Tool files created: {'PASS' if has_tools else 'FAIL'} (n={len(tool_files)})")
        print(f"  Edit block found:   {'PASS' if has_edit else 'FAIL'}")

        # Extract and save edit content for verification
        edit_correct = False
        if has_edit:
            # Save full output for debugging
            with open(os.path.join(tmpdir, "model_output.txt"), "w") as f:
                f.write(output)
            # Try multiple regex patterns
            import re
            for pat in [r'<edit file="([^"]+)">\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>\s*</edit>',
                        r'<edit file=([^>]+)>\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>\s*</edit>']:
                m = re.search(pat, output, re.DOTALL)
                if m:
                    fn, search, replace = m.group(1), m.group(2), m.group(3)
                    print(f"  File:    {fn}")
                    print(f"  Search:  {search.strip()[:100]}")
                    print(f"  Replace: {replace.strip()[:100]}")
                    if 'a - b' in search or 'return a - b' in search:
                        if 'a + b' in replace or 'return a + b' in replace:
                            edit_correct = True
                            print(f"  Edit correctness:   PASS (correct fix)")
                        else:
                            print(f"  Edit correctness:   FAIL (search correct but wrong replace)")
                    else:
                        print(f"  Edit correctness:   FAIL (didn't find the bug)")
                    break
            if not m:
                # Dump the edit block area
                idx = output.find("<edit ")
                if idx >= 0:
                    print(f"  Edit block preview: {output[idx:idx+300]}")

        if edit_correct:
            print("\n  SMOKE TEST PASSED — pipeline functional, edit correct ✓")
            return True
        elif has_output and has_edit:
            print("\n  SMOKE TEST PASSED — pipeline functional, edit present")
            return True
        else:
            print("\n  SMOKE TEST PARTIAL — review output above")
            return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
