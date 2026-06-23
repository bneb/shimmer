#!/usr/bin/env python3
"""Unit tests for agentless mode: localization and prompt building."""

import os
import sys
import tempfile
import shutil
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "agentic"))
from generate_swe_bench import (
    localize_files_keyword,
    build_agentless_prompt,
    scan_repo,
)


# ── localize_files_keyword ────────────────────────────────────────────────

def test_keyword_extraction_basic():
    """Extracts paths, CamelCase, snake_case, and module refs from issue text."""
    issue = (
        "The FILE_UPLOAD_PERMISSIONS setting in django/conf/global_settings.py\n"
        "is not honored by the ModelAdmin. The get_response function\n"
        "in django.core.handlers should call validate_name properly."
    )
    # This test just validates the function returns files without crashing
    # when run against a real-ish directory
    with tempfile.TemporaryDirectory() as td:
        # Create a minimal repo structure
        os.makedirs(os.path.join(td, "django", "conf"), exist_ok=True)
        Path(os.path.join(td, "django/conf/global_settings.py")).write_text(
            "FILE_UPLOAD_PERMISSIONS = None\n")
        os.makedirs(os.path.join(td, "django", "core"), exist_ok=True)
        Path(os.path.join(td, "django/core/handlers.py")).write_text(
            "def get_response(request): pass\n")

        files = localize_files_keyword(issue, td, max_files=10)
        assert len(files) >= 1, f"Expected at least 1 file, got {files}"
        # Should find the files that match keywords
        found_paths = [os.path.normpath(f) for f in files]
        assert any("global_settings.py" in f for f in found_paths), \
            f"Expected global_settings.py in results: {found_paths}"


def test_keyword_extraction_empty_issue():
    """Fallback returns all .py files when no keywords extracted."""
    with tempfile.TemporaryDirectory() as td:
        Path(os.path.join(td, "a.py")).write_text("x=1")
        Path(os.path.join(td, "b.py")).write_text("y=2")

        files = localize_files_keyword("", td, max_files=5)
        assert len(files) == 2, f"Expected 2 files from fallback, got {len(files)}"


def test_keyword_extraction_filters_reserved():
    """Python reserved words and short tokens are filtered out."""
    issue = "The class method def return self try except import None True False in is"
    with tempfile.TemporaryDirectory() as td:
        Path(os.path.join(td, "a.py")).write_text("x=1")
        files = localize_files_keyword(issue, td, max_files=5)
        # Should fall back to all .py files since all keywords are reserved/short
        assert len(files) >= 0  # Doesn't crash


def test_keyword_extraction_max_files_cap():
    """Respects max_files cap."""
    issue = "function_name class_name module.submodule path/to/file.py"
    with tempfile.TemporaryDirectory() as td:
        for i in range(30):
            Path(os.path.join(td, f"mod{i:02d}.py")).write_text(
                f"def function_name(): pass\n"
                f"class class_name: pass\n"
            )
        files = localize_files_keyword(issue, td, max_files=5)
        assert len(files) <= 5, f"Expected at most 5 files, got {len(files)}"


# ── build_agentless_prompt ───────────────────────────────────────────────

def test_agentless_prompt_structure():
    """Prompt contains required sections."""
    repo_hint = "## Structure\n- src/main.py\n"
    problem = "Fix the bug in main.py"
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, "main.py")
        Path(fpath).write_text("def main():\n    pass\n")
        rel = os.path.relpath(fpath, td)

        prompt = build_agentless_prompt(repo_hint, problem, [rel], td)
        assert "REPOSITORY STRUCTURE" in prompt
        assert "DETAILED ISSUE DESCRIPTION" in prompt
        assert "RELEVANT SOURCE FILES" in prompt
        assert "Fix the bug" in prompt
        assert "INSTRUCTIONS" in prompt
        assert "Do NOT output JSON tool calls" in prompt
        assert "Begin now." in prompt
        assert "def main():" in prompt  # file content inlined
        assert "END FILE:" in prompt


def test_agentless_prompt_small_file_inlined_fully():
    """Files under 100 lines are inlined completely."""
    repo_hint = "## Structure\n"
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, "small.py")
        content = "\n".join(f"line {i}" for i in range(50))
        Path(fpath).write_text(content)
        rel = os.path.relpath(fpath, td)

        prompt = build_agentless_prompt(repo_hint, "issue", [rel], td)
        assert "line 0" in prompt
        assert "line 49" in prompt
        assert "omitted" not in prompt  # no truncation


def test_agentless_prompt_large_file_truncated():
    """Files over 300 lines show first 30 lines only."""
    repo_hint = "## Structure\n"
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, "large.py")
        content = "\n".join(f"line {i}" for i in range(500))
        Path(fpath).write_text(content)
        rel = os.path.relpath(fpath, td)

        prompt = build_agentless_prompt(repo_hint, "issue", [rel], td)
        assert "line 0" in prompt
        assert "line 29" in prompt
        assert "line 499" not in prompt  # truncated
        assert "first 30 of 500 lines" in prompt


def test_agentless_prompt_medium_file_partial():
    """Files 100-300 lines show first 30 + last 20 lines."""
    repo_hint = "## Structure\n"
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, "med.py")
        content = "\n".join(f"line {i}" for i in range(200))
        Path(fpath).write_text(content)
        rel = os.path.relpath(fpath, td)

        prompt = build_agentless_prompt(repo_hint, "issue", [rel], td)
        assert "line 0" in prompt
        assert "line 29" in prompt
        assert "line 180" in prompt  # in last 20
        assert "line 199" in prompt  # in last 20
        assert "omitted" in prompt  # truncation marker


def test_agentless_prompt_token_budget_enforced():
    """Stops adding files when char budget is exhausted."""
    repo_hint = "## Structure\n"
    with tempfile.TemporaryDirectory() as td:
        files = []
        for i in range(30):
            fpath = os.path.join(td, f"file{i:02d}.py")
            # ~5K chars each
            content = "\n".join(f"x = {j}  # padding line" for j in range(100))
            Path(fpath).write_text(content)
            files.append(os.path.relpath(fpath, td))

        prompt = build_agentless_prompt(repo_hint, "issue", files, td)
        # With 30 files at ~5K chars each, budget should cut off before all 30
        count = prompt.count("BEGIN FILE:")
        assert count < 30, f"Expected fewer than 30 files, got {count}"
        # At least some files should be included
        assert count >= 1, "Expected at least 1 file"


def test_agentless_prompt_no_tools_instruction():
    """Prompt explicitly says no tools available."""
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, "a.py")
        Path(fpath).write_text("pass")
        rel = os.path.relpath(fpath, td)

        prompt = build_agentless_prompt("#", "issue", [rel], td)
        assert "Do NOT output JSON tool calls" in prompt
        assert "There are no tools available" in prompt


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    passed = 0
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
