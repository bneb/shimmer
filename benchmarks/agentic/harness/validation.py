"""Syntax validation for model-generated patches. Pure logic, testable."""

import os


def files_from_patch(patch):
    """Extract .py file paths from a unified diff patch. Returns list of paths."""
    paths = []
    for line in patch.split("\n"):
        for prefix in ["+++ b/", "--- a/"]:
            if line.startswith(prefix):
                fpath = line[len(prefix):]
                if fpath.endswith(".py"):
                    paths.append(fpath)
    return paths


def check_syntax(filepath):
    """Check if a Python file has valid syntax. Returns (ok: bool, error: str|None)."""
    if not os.path.exists(filepath):
        return False, "file not found"
    try:
        import ast
        with open(filepath) as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def autoformat(filepath):
    """Run autopep8 on a file. Returns True if successful."""
    try:
        import subprocess
        subprocess.run(["python3", "-m", "autopep8", "--in-place",
                        "--aggressive", filepath], capture_output=True, timeout=30)
        return True
    except Exception:
        return False
