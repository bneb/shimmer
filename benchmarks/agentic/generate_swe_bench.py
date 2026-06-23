#!/usr/bin/env python3
"""Shimmer SWE-bench Evaluation Harness.

Runs N SWE-bench Lite instances against a Shimmer model, collects
per-instance stats, verifies patches against test suites, and
outputs structured JSON results for comparison across models.
"""

import os, json, subprocess, time, re, sys, argparse, shutil, threading, random
from pathlib import Path
from datasets import load_dataset

# ── Configuration ────────────────────────────────────────────────────────────

SHIMMER_CWD = "/Users/kevin/projects/shimmer"
PREDICTIONS_FILE = f"{SHIMMER_CWD}/data/predictions.jsonl"
RESULTS_FILE = f"{SHIMMER_CWD}/data/eval_results.json"
CLONE_BASE = "/tmp/swe_lite"
INSTANCE_TIMEOUT = 900  # 15 minutes per instance
TEST_VERIFY_TIMEOUT = 120  # 2 minutes for test verification

DEFAULT_SAMPLE_CFG = "temp=0.0,topk=0,repp=1.0"  # Greedy; required for reliable tool detection
DEFAULT_MODEL = f"{SHIMMER_CWD}/models/qwen2.5-coder-7b.gguf"


# ── Shimmer Runner ───────────────────────────────────────────────────────────

def run_shimmer(prompt, cwd, model_path, sample_config, no_tools=False):
    """Run shimmer inference with wall-clock timeout. Returns output text."""
    cmd = [
        f"{SHIMMER_CWD}/target/release/shimmer",
        "--main-model", model_path,
        "--prompt", prompt,
        "--no-blind-edit-blocker",
        "--sample", sample_config,
    ]
    if no_tools:
        cmd.append("--no-tools")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, cwd=cwd)
    response_chunks = []
    done = threading.Event()
    def reader():
        for line in process.stdout:
            print(line, end="", flush=True)
            response_chunks.append(line)
        done.set()
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout=INSTANCE_TIMEOUT)
    if not done.is_set():
        # SIGKILL first (GPU kernels ignore SIGTERM)
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            import signal
            os.kill(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"\n[TIMEOUT after {INSTANCE_TIMEOUT}s — process unkillable (GPU hang)]", flush=True)
        print(f"\n[TIMEOUT after {INSTANCE_TIMEOUT}s]", flush=True)
        response_chunks.append("\n[TIMEOUT]\n")
    return "".join(response_chunks)


# ── Repo Scanner ─────────────────────────────────────────────────────────────

def scan_repo(clone_dir):
    """Scan repo for structure, source files, and test files. Returns prompt-ready string."""
    lines = ["## Repository Structure", "```"]
    try:
        for entry in sorted(os.listdir(clone_dir)):
            if entry.startswith('.') or entry in ('target', 'node_modules'): continue
            p = os.path.join(clone_dir, entry)
            prefix = "[dir] " if os.path.isdir(p) else "[file]"
            lines.append(f"  {prefix} {entry}")
    except Exception: pass
    lines.append("```")

    src_files = []
    for root, dirs, files in os.walk(clone_dir):
        depth = root[len(clone_dir):].count(os.sep)
        if depth > 2: continue
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules','target','.git')]
        for f in files:
            if f.endswith('.py'):
                src_files.append(os.path.relpath(os.path.join(root, f), clone_dir))
    if src_files:
        lines.append(f"\n## Source Files ({len(src_files)} .py files)")
        for sf in sorted(src_files)[:50]:
            lines.append(f"  - {sf}")
        if len(src_files) > 50:
            lines.append(f"  ... and {len(src_files)-50} more")
    test_files = [f for f in src_files if 'test' in f.lower()]
    if test_files:
        lines.append(f"\n## Test Files ({len(test_files)} found)")
        for tf in sorted(test_files)[:15]:
            lines.append(f"  - {tf}")
    return "\n".join(lines) + "\n"


_RESERVED_WORDS = {"and", "as", "assert", "break", "class", "continue", "def",
                  "del", "elif", "else", "except", "finally", "for", "from",
                  "global", "if", "import", "in", "is", "lambda", "nonlocal",
                  "not", "or", "pass", "raise", "return", "try", "while", "with",
                  "yield", "True", "False", "None", "self", "cls"}


def _extract_keywords(issue_text):
    """Extract Python identifiers and file paths from issue text."""
    keywords = set()
    for m in re.finditer(r'["\']([\w./-]+\.py)["\']', issue_text):
        keywords.add(m.group(1))
    for m in re.finditer(r'([\w./-]+\.py)', issue_text):
        p = m.group(1)
        if "/" in p and not p.startswith("."):
            keywords.add(p)
    for m in re.finditer(r'\b([A-Z][a-zA-Z0-9]{2,})\b', issue_text):
        keywords.add(m.group(1))
    for m in re.finditer(r'\b([A-Z][A-Z0-9_]{3,})\b', issue_text):
        keywords.add(m.group(1))
    for m in re.finditer(r'\b([a-z]+(?:_[a-z0-9]+){1,})\b', issue_text):
        kw = m.group(1)
        if len(kw) >= 5:
            keywords.add(kw)
    for m in re.finditer(r'\b([a-z_]+(?:\.[a-z_]+)+)\b', issue_text):
        kw = m.group(1)
        if any(c.isalpha() for c in kw):
            keywords.add(kw)
    for m in re.finditer(r'(?:from|import)\s+([\w.]+)', issue_text):
        keywords.add(m.group(1))
    return {k for k in keywords if k not in _RESERVED_WORDS and len(k) >= 3}


_VENDORED_DIRS = {"site-packages", "node_modules", "venv",
                  ".env", "build", "dist", ".eggs", ".git",
                  "__pycache__", ".tox", ".mypy_cache"}


def _walk_py_files(clone_dir):
    """Return sorted list of relative paths to all .py files in clone_dir."""
    src_files = []
    for root, dirs, files in os.walk(clone_dir):
        dirs[:] = [d for d in dirs
                   if not d.startswith(".") and d not in _VENDORED_DIRS]
        for f in files:
            if f.endswith(".py"):
                src_files.append(os.path.relpath(
                    os.path.join(root, f), clone_dir))
    return sorted(src_files)


def _score_files_by_grep(keywords, clone_dir):
    """Grep each keyword in clone_dir. Returns {rel_path: match_count}."""
    scores = {}
    # Pass 1: path-like keywords get bonus for exact file existence
    for kw in sorted(keywords, key=len, reverse=True):
        if "/" in kw and kw.endswith(".py"):
            full = os.path.join(clone_dir, kw)
            if os.path.isfile(full):
                scores[kw] = scores.get(kw, 0) + 3
    # Pass 2: grep for keyword content (top 30 longest keywords)
    for kw in sorted(keywords, key=len, reverse=True)[:30]:
        search_terms = [kw]
        if "/" in kw:
            fname = os.path.basename(kw)
            if fname != kw and len(fname) >= 3:
                search_terms.append(fname)
        for term in search_terms:
            try:
                result = subprocess.run(
                    ["rg", "-l", "--no-heading", "--", term, clone_dir],
                    capture_output=True, text=True, timeout=15)
                for path in result.stdout.strip().split("\n"):
                    if not path or not path.endswith(".py"):
                        continue
                    rel = os.path.relpath(path, clone_dir)
                    if any(p in _VENDORED_DIRS for p in rel.split(os.sep)):
                        continue
                    scores[rel] = scores.get(rel, 0) + 1
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
    return scores


def localize_files_keyword(issue_text, clone_dir, max_files=20):
    """Localize relevant files using keyword extraction + rg grep.

    Returns ranked file paths relative to clone_dir, capped at max_files.
    """
    keywords = _extract_keywords(issue_text)
    if not keywords:
        print("  [localize] No keywords extracted — using all source files")
        return _walk_py_files(clone_dir)[:max_files]

    print(f"  [localize] {len(keywords)} keywords extracted: "
          f"{', '.join(sorted(keywords, key=len, reverse=True)[:10])}")

    scores = _score_files_by_grep(keywords, clone_dir)
    if not scores:
        print("  [localize] rg found no matches — falling back to all source files")
        return _walk_py_files(clone_dir)[:max_files]

    ranked = sorted(scores.items(), key=lambda x: (-x[1], len(x[0])))
    files = [p for p, _ in ranked[:max_files]]
    print(f"  [localize] Top {len(files)} files (of {len(scores)} matched): "
          f"{', '.join(f[:50] for f in files[:5])}")
    return files


def build_prompt(repo_hint, problem):
    """Construct the SWE-bench prompt with repo context and convergence rules."""
    return f"""There is an issue reported in the repository. Please fix it.

{repo_hint}
Issue Description:
{problem}

AVAILABLE TOOLS (USE JSON FORMAT):
- rg: {{"name": "rg", "arguments": ["pattern", "path"]}}
- cat: {{"name": "cat", "arguments": ["file.py"]}} or {{"name": "cat", "arguments": ["file.py", "10", "50"]}}
- fd: {{"name": "fd", "arguments": ["pattern"]}}
- ls: {{"name": "ls", "arguments": ["path"]}}
- git: {{"name": "git", "arguments": ["diff"]}}

INVESTIGATE BEFORE EDITING:
1. FIRST: Use `rg` to search for the function, class, or setting name from the issue.
2. THEN: Use `cat` to read the files that contain matches.
3. LAST: Provide your fix using `<edit>` tags with EXACT lines copied from `cat` output.

CRITICAL RULES:
1. File paths MUST exactly match one entry in the Source Files list above. Copy verbatim.
2. COPY-PASTE exact lines from `cat` output into your `<search>` block. Do not re-type code.
3. Your `<search>` must match exactly once in the file, including all whitespace and indentation.

Example of a correct edit block:
<edit file="lib/colors.py">
<search>
def inverse(value):
    return 1.0-value
</search>
<replace>
def inverse(value):
    return value
</replace>
</edit>

SELF-CHECK BEFORE SUBMITTING:
Before your final <edit>, pause and ask yourself:
- Does the changed code actually do what the issue asks for?
- Is the change consistent with how the rest of the file works?
- Would this break anything?
If unsure, use `cat` or `rg` to verify before editing.

Begin now.
"""


_PROMPT_CHARS_BUDGET = 48000  # ~12K tokens, leaves room for output

_HYBRID_FOOTER = [
    "",
    "INSTRUCTIONS:",
    "- Source files with line numbers are shown above for orientation.",
    "- You MAY use up to 3 read-only tools (rg, cat, ls) to verify",
    "  exact lines before editing. Stop investigating after 3 tools.",
    "- When you have the right lines, produce your fix using XML <edit> tags.",
    "- Do NOT output JSON tool calls after your 3rd tool.",
    "",
    "CRITICAL RULES:",
    "1. File paths in <edit file=\"...\"> must match a path shown above.",
    "2. Copy exact lines from the LEFT side into <search>.",
    "   Do NOT include '# L294'-style line-number comments in your search.",
    "3. Your <search> must match exactly once in the file.",
    "4. Pay attention to line numbers: place new code near related settings.",
    "",
    "Example of a correct edit block:",
    "<edit file=\"path/to/__demo_placeholder__.py\">",
    "<search>",
    "def __demo_placeholder_func(value):",
    "    return 1.0 - value",
    "</search>",
    "<replace>",
    "def __demo_placeholder_func(value):",
    "    return value",
    "</replace>",
    "</edit>",
    "",
    "Begin now.",
]

_AGENTLESS_FOOTER = [
    "",
    "INSTRUCTIONS:",
    "- All relevant code is shown above. You do NOT need to use any tools.",
    "- Do NOT output JSON tool calls. Tools are not available.",
    "- Provide your fix directly using XML <edit> tags.",
    "",
    "CRITICAL RULES:",
    "1. File paths in <edit file=\"...\"> must match a path shown above.",
    "2. Copy exact lines from the LEFT side of the code above into <search>.",
    "   Do NOT include '# L294'-style line-number comments in your search.",
    "3. Your <search> must match exactly once in the file.",
    "4. Pay attention to line numbers: place new code near related settings,",
    "   not at the top of the file. The '# L294' comments tell you where",
    "   each line actually lives in the file.",
    "",
    "Example of a correct edit block:",
    "<edit file=\"path/to/__demo_placeholder__.py\">",
    "<search>",
    "def __demo_placeholder_func(value):",
    "    return 1.0 - value",
    "</search>",
    "<replace>",
    "def __demo_placeholder_func(value):",
    "    return value",
    "</replace>",
    "</edit>",
    "",
    "SELF-CHECK BEFORE SUBMITTING:",
    "- Does the changed code actually do what the issue asks for?",
    "- Is the change consistent with how the rest of the file works?",
    "- Would this break anything?",
    "- Did you place the edit at the correct line number (not just at the top)?",
    "",
    "IMPORTANT: There are no tools available. Only output the edit block.",
    "Begin now.",
]


def _read_file_snippet(fpath, clone_dir, keywords=None):
    """Read a file and return a snippet with line-number annotations.

    Line numbers appear as right-hand comments (``# L294``) so the model
    can copy clean code into <search> blocks while knowing exact positions.

    When keywords are provided, context is shown around keyword matches
    rather than just the top of the file.

    Returns (snippet: str, label: str) or raises OSError.
    """
    content = Path(os.path.join(clone_dir, fpath)).read_text()
    content_lines = content.split("\n")
    n_lines = len(content_lines)

    if n_lines <= 100:
        return _annotate_lines(content_lines, 1, n_lines), fpath

    # Find lines matching keywords to anchor context windows
    match_lines = set()
    if keywords:
        for kw in keywords:
            kw_lower = kw.lower()
            for i, line in enumerate(content_lines):
                if kw_lower in line.lower():
                    match_lines.add(i)

    if match_lines:
        # Build context windows around each keyword match
        ctx_radius = 12
        windows = []
        added = set()
        for ml in sorted(match_lines):
            start = max(0, ml - ctx_radius)
            end = min(n_lines, ml + ctx_radius + 1)
            # Merge overlapping windows
            key = (start, end)
            if key not in added:
                added.add(key)
                windows.append((start, end))
        # Merge adjacent/overlapping windows
        merged = []
        for start, end in sorted(windows):
            if merged and start <= merged[-1][1] + 3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        # Build snippet from merged windows
        parts = []
        for i, (start, end) in enumerate(merged):
            if i > 0:
                skipped = start - merged[i - 1][1]
                parts.append(f"\n... ({skipped} lines between matches) ...\n")
            chunk = _annotate_lines(content_lines[start:end], start + 1, end)
            parts.append(chunk)
        snippet = "\n".join(parts)
        label = f"{fpath} (matched context)"
        return snippet, label

    # No keyword matches: show top + bottom with line numbers
    if n_lines <= 300:
        top = _annotate_lines(content_lines[:30], 1, 30)
        bot = _annotate_lines(content_lines[-20:], n_lines - 19, n_lines)
        snippet = top + f"\n... ({n_lines - 50} lines omitted) ...\n" + bot
        return snippet, f"{fpath} (lines 1-30, {n_lines-19}-{n_lines})"
    else:
        top = _annotate_lines(content_lines[:30], 1, 30)
        snippet = top + f"\n... (file continues for {n_lines - 30} more lines) ..."
        return snippet, f"{fpath} (first 30 of {n_lines} lines)"


def _annotate_lines(lines, start_num, _end_num):
    """Add right-hand line-number comments to a list of code lines.

    Code stays clean on the left for copy-paste into <search> blocks.
    Line numbers as ``# L294`` comments on the right for positional reference.
    """
    annotated = []
    max_line = start_num + len(lines) - 1
    num_width = len(str(max_line))
    for i, line in enumerate(lines):
        line_num = start_num + i
        annotated.append(f"{line:<72}  # L{line_num:>{num_width}}")
    return "\n".join(annotated)


def _build_agentless_header(repo_hint, problem):
    """Return the initial lines for an agentless prompt (before file contents)."""
    return [
        "There is an issue reported in the repository. Please fix it.",
        "",
        "[REPOSITORY STRUCTURE]",
        repo_hint.strip(),
        "",
        "[DETAILED ISSUE DESCRIPTION]",
        problem.strip(),
        "",
        "[RELEVANT SOURCE FILES]",
        "Key parts of the files most relevant to this issue.",
        "Line numbers appear as right-hand comments (e.g. '# L294').",
        "They show the actual position in the file — use them to place",
        "your edit near related code, not at the top of the file.",
        "Copy the code from the LEFT side of each line into <search>.",
        "Do NOT include the line-number comments in your <search> block.",
        "Do NOT search for additional files or use any tools.",
        "",
    ]


def _add_file_block(lines, fpath, snippet, label):
    """Append a formatted file block to lines. Returns char count of block."""
    block = (f"\n--- BEGIN FILE: {label} ---\n"
             + snippet
             + f"\n--- END FILE: {fpath} ---\n")
    lines.append(block.rstrip())
    return len(block)


def build_fill_prompt(repo_hint, problem, clone_dir):
    """Build a prompt asking the model only for replacement code.

    Returns (prompt, anchor_info) where anchor_info is None or a dict
    with file, start_line, end_line, and original_lines for constructing
    the edit from the model's raw output.
    """
    keywords = _extract_keywords(problem)
    if not keywords:
        prompt = build_agentless_prompt(repo_hint, problem, _walk_py_files(clone_dir)[:5], clone_dir)
        return prompt, None

    files = localize_files_keyword(problem, clone_dir, max_files=3)
    if not files:
        prompt = build_agentless_prompt(repo_hint, problem, _walk_py_files(clone_dir)[:5], clone_dir)
        return prompt, None

    # Find the best keyword match
    best_file, best_line = None, None
    for fpath in files:
        try:
            content = Path(os.path.join(clone_dir, fpath)).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for kw in sorted(keywords, key=len, reverse=True):
            kw_lower = kw.lower()
            for i, line in enumerate(content.split("\n")):
                if kw_lower in line.lower():
                    best_file, best_line = fpath, i
                    break
            if best_file:
                break
        if best_file:
            break

    if not best_file:
        prompt = build_agentless_prompt(repo_hint, problem, files, clone_dir)
        return prompt, None

    # Show +/- 15 lines around the match
    content = Path(os.path.join(clone_dir, best_file)).read_text()
    lines = content.split("\n")
    start = max(0, best_line - 15)
    end = min(len(lines), best_line + 16)
    context = _annotate_lines(lines[start:end], start + 1, end)

    prompt = [
        "A bug was reported. Fix it by changing the highlighted code.",
        "",
        "[ISSUE]",
        problem.strip(),
        "",
        f"[CODE TO CHANGE — {best_file}, lines {start+1}-{end}]",
        "Output ONLY the corrected code for the lines that need to change.",
        "Keep everything else exactly as shown.",
        "",
        context,
        "",
        "Output only the corrected lines. No XML, no explanation.",
        "",
        "Begin now.",
    ]
    anchor = {
        "file": best_file,
        "start": start,
        "end": end,
        "original": "\n".join(lines[start:end]),
    }
    return "\n".join(prompt), anchor


def build_agentless_prompt(repo_hint, problem, localized_files, clone_dir, hybrid=False):
    """Build an agentless or hybrid repair prompt with source files inlined."""
    # Extract keywords for context-aware snippet selection
    keywords = _extract_keywords(problem)
    lines = _build_agentless_header(repo_hint, problem)
    chars_used = len("\n".join(lines))

    for fpath in localized_files:
        try:
            snippet, label = _read_file_snippet(fpath, clone_dir, keywords)
        except (OSError, UnicodeDecodeError):
            continue

        block_len = (len(fpath) + len(snippet) + len(label) + 30)

        if chars_used + block_len > _PROMPT_CHARS_BUDGET:
            try:
                content = Path(os.path.join(clone_dir, fpath)).read_text()
                tiny_lines = content.split("\n")[:15]
                tiny = _annotate_lines(tiny_lines, 1, len(tiny_lines))
                chars_used += _add_file_block(
                    lines, fpath, tiny,
                    f"{fpath} (first 15 lines)")
            except Exception:
                pass
            break

        chars_used += _add_file_block(lines, fpath, snippet, label)

    footer = _HYBRID_FOOTER if hybrid else _AGENTLESS_FOOTER
    lines.extend(footer)
    return "\n".join(lines)


# ── Patch Extraction ─────────────────────────────────────────────────────────

def find_real_paths(clone_dir, response_text):
    """Extract real .py paths the model discovered."""
    paths = set()
    for m in re.finditer(r'"arguments"\s*:\s*\[(.*?)\]', response_text, re.DOTALL):
        for arg in re.finditer(r'"([^"]+)"', m.group(1)):
            v = arg.group(1)
            if '.' in v and '/' in v and not v.startswith('.'):
                paths.add(v)
    for m in re.finditer(r"File '([^']+)' does not exist", response_text):
        paths.add(m.group(1))
    for m in re.finditer(r'([\w.-]+/)+[\w.-]+\.py', response_text):
        p = m.group(0)
        if os.path.exists(os.path.join(clone_dir, p)):
            paths.add(p)
    return [p for p in paths if os.path.exists(os.path.join(clone_dir, p))]


def score_candidate(clone_dir, path, search_text, _replace_text):
    """Score a candidate file by how well it matches the search content."""
    score = 0
    try:
        content = Path(os.path.join(clone_dir, path)).read_text()
        if search_text.strip() and search_text.strip() in content:
            score += 1000
        first_line = next((l for l in search_text.split('\n') if l.strip()), '')
        if first_line and first_line.strip() in content:
            score += 200
        score += len(set(search_text.split()) & set(content.split()))
    except Exception: pass
    return score


def resolve_path(clone_dir, file_path, search_text, replace_text, response_text):
    """Resolve a model-produced path to a real file on disk."""
    if os.path.exists(os.path.join(clone_dir, file_path)):
        return file_path
    candidates = find_real_paths(clone_dir, response_text)
    candidates.extend([p for p in find_real_paths(clone_dir, response_text)
                       if p.endswith(file_path) or file_path.endswith(p.split('/')[-1])])
    if candidates:
        scored = [(p, score_candidate(clone_dir, p, search_text, replace_text)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scored[0]
        if best_score > 100:
            print(f"  Resolved '{file_path}' -> '{best}' (score {best_score})")
            return best
    return file_path


def apply_fuzzy_edit(content, search_str, replace_str):
    """Try exact match, then normalized, then difflib anchor match.

    Returns (new_content, method) where method is one of:
    exact, normalized, difflib, or failed.
    """
    search_str = search_str.strip()
    replace_str = replace_str.strip()
    if not search_str:
        print(f"  Rejected: empty search after strip (model produced whitespace-only search)")
        return content, "failed"
    if search_str in content:
        return content.replace(search_str, replace_str), "exact"
    return _normalized_match(content, search_str, replace_str)


def _normalize(text):
    """Strip punctuation, underscores, and common plural suffixes for fuzzy matching."""
    import re
    text = text.lower()
    text = re.sub(r'[_\-.]+', ' ', text)
    text = re.sub(r'\bies\b', 'y', text)
    text = re.sub(r'\b([a-z]{3,})s\b', r'\1', text)
    return re.sub(r'\s+', '', text)


def _normalized_match(content, search_str, replace_str):
    """Match by normalizing both search and content lines.

    Strips underscores, punctuation, and plural suffixes so the model's
    slightly-wrong transcription can still find the right location.
    Once located, replaces the ORIGINAL content lines with the replace text.
    """
    import difflib
    content_lines = content.split('\n')
    search_lines = [l for l in search_str.split('\n') if l.strip()]
    replace_lines = replace_str.split('\n')
    if not search_lines:
        return _difflib_match(content, search_str, replace_str)
    # Normalize all lines
    norm_content = [_normalize(l) for l in content_lines]
    norm_search = [_normalize(l) for l in search_lines]
    # Score each position as a potential match for the search block
    best_score, best_idx = 0.0, None
    for i in range(len(content_lines) - len(search_lines) + 1):
        window = norm_content[i:i + len(search_lines)]
        score = sum(difflib.SequenceMatcher(None, w, s).ratio()
                    for w, s in zip(window, norm_search)) / len(search_lines)
        if score > best_score:
            best_score, best_idx = score, i
    if best_idx is not None and best_score > 0.7:
        # Determine end: anchor-indent boundary like difflib
        anchor = content_lines[best_idx]
        anchor_indent = len(anchor) - len(anchor.lstrip())
        max_block = max(len(search_lines), len(replace_lines)) * 2
        end_idx = best_idx + len(search_lines)
        for i in range(best_idx + 1, min(best_idx + max_block, len(content_lines))):
            line = content_lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= anchor_indent:
                end_idx = i
                break
        else:
            end_idx = min(best_idx + max_block, len(content_lines))
        new_lines = (content_lines[:best_idx]
                     + replace_lines
                     + content_lines[end_idx:])
        print(f"  Normalized: matched at line {best_idx+1} (score {best_score:.2f})"
              + f", replaced {end_idx - best_idx} lines with {len(replace_lines)}")
        return '\n'.join(new_lines), "normalized"
    return _difflib_match(content, search_str, replace_str)


def _difflib_match(content, search_str, replace_str):
    """Use the first search line as an anchor to find the edit location.

    When the model can't reproduce exact source lines (common at 12B Q4),
    the function signature or first meaningful line is usually correct even
    when the body is hallucinated. Finds that anchor via difflib, then
    replaces through the end of the original block (bounded by indentation
    or next definition).

    Returns (new_content, 'difflib') or (content, 'failed').
    """
    import difflib
    content_lines = content.split('\n')
    search_lines = search_str.split('\n')
    replace_lines = replace_str.split('\n')
    # Use the first non-empty search line as the anchor
    anchor = next((l.strip() for l in search_lines if l.strip()), None)
    if not anchor:
        return content, "failed"
    # Find the best match for the anchor line in the file
    best_idx, best_ratio = max(
        ((i, difflib.SequenceMatcher(None, cl.strip(), anchor).ratio())
         for i, cl in enumerate(content_lines)),
        key=lambda x: x[1])
    if best_ratio < 0.5:
        print(f"  Difflib: no anchor match for {repr(anchor[:60])} (best ratio {best_ratio:.2f})")
        return content, "failed"
    # Determine the anchor's indentation level
    anchor_indent = len(content_lines[best_idx]) - len(content_lines[best_idx].lstrip())
    # Find end: the next line at same or lesser indent, or max(search_lines, replace_lines) lines
    max_block = max(len(search_lines), len(replace_lines)) * 2
    end_idx = best_idx + len(search_lines)
    for i in range(best_idx + 1, min(best_idx + max_block, len(content_lines))):
        line = content_lines[i]
        if line.strip() and (len(line) - len(line.lstrip())) <= anchor_indent:
            end_idx = i
            break
    else:
        end_idx = min(best_idx + max_block, len(content_lines))
    new_lines = (content_lines[:best_idx]
                 + replace_lines
                 + content_lines[end_idx:])
    print(f"  Difflib: anchored at line {best_idx+1} (ratio {best_ratio:.2f})"
          + f", replaced {end_idx - best_idx} original lines with {len(replace_lines)}")
    return '\n'.join(new_lines), "difflib"


def extract_patch(clone_dir, response_text):
    """Extract and apply edits from model output. Returns (patch, stats)."""
    # Only process text after the last prompt marker (Begin now.)
    # This prevents extracting example <edit> blocks from the prompt.
    begin_idx = response_text.rfind("Begin now.")
    if begin_idx != -1:
        model_output = response_text[begin_idx:]
    else:
        model_output = response_text
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', model_output, flags=re.DOTALL)
    # Two-step extraction: find complete <edit> blocks first, then extract
    # search/replace from within each block. This prevents the regex from
    # spanning across incomplete blocks (e.g., thought-channel drafts).
    edits = []
    for block_match in re.finditer(r'<edit file="([^"]+)">(.*?)</edit>', cleaned, re.DOTALL):
        file_path = block_match.group(1)
        block = block_match.group(2)
        s_match = re.search(r'<search>(.*?)</search>', block, re.DOTALL)
        r_match = re.search(r'<replace>(.*?)</replace>', block, re.DOTALL)
        if s_match and r_match:
            edits.append((file_path, s_match.group(1), r_match.group(1)))
    json_patterns = [
        r'\{\s*"edit"\s*:\s*\{\s*"file"\s*:\s*"([^"]+)"\s*,\s*"search"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"replace"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        r'\{\s*"file"\s*:\s*"([^"]+)"\s*,\s*"search"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"replace"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
    ]
    for pat in json_patterns:
        for fp, s, r in re.findall(pat, cleaned):
            edits.append((fp, s.replace('\\n','\n').replace('\\t','\t').replace('\\"','"'),
                          r.replace('\\n','\n').replace('\\t','\t').replace('\\"','"')))

    # Deduplicate: keep only last occurrence of each (file, search, replace) tuple.
    # Earlier occurrences are often thought-channel drafts; the last is the final answer.
    seen = set()
    deduped = []
    for e in reversed(edits):
        key = (e[0], e[1].strip(), e[2].strip())
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    edits = list(reversed(deduped))

    stats = {"edits_found": len(edits), "edits_applied": 0, "edits_failed": 0,
             "files_touched": len(set(e[0] for e in edits))}
    for file_path, search, replace in edits:
        resolved = resolve_path(clone_dir, file_path, search, replace, response_text)
        full = os.path.join(clone_dir, resolved)
        if not os.path.exists(full):
            print(f"  Edit failed: {resolved} not found"); stats["edits_failed"] += 1; continue
        content = Path(full).read_text()
        new_content, method = apply_fuzzy_edit(content, search, replace)
        if method != "failed":
            Path(full).write_text(new_content)
            stats["edits_applied"] += 1
            print(f"  Applied ({method}): {resolved}")
        else:
            stats["edits_failed"] += 1
            print(f"  Failed: search not found in {resolved}")
    diff = subprocess.run(["git", "diff"], cwd=clone_dir, capture_output=True, text=True)
    return diff.stdout, stats


# ── Test Verification ────────────────────────────────────────────────────────

def verify_patch(clone_dir, instance_id):
    """Run the repo's test suite to verify the applied patch. Returns (passed, log)."""
    # Try common test commands
    candidates = [
        ["python", "-m", "pytest", "-x", "--timeout=60"],
        ["python", "-m", "pytest", "-x"],
        [sys.executable, "-m", "pytest", "-x"],
    ]
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, cwd=clone_dir, capture_output=True, text=True,
                                  timeout=TEST_VERIFY_TIMEOUT)
            passed = proc.returncode == 0
            return passed, proc.stdout[-500:] + "\n" + proc.stderr[-500:]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return None, "No test runner found"


# ── Main Eval Loop ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shimmer SWE-bench Evaluation Harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--sample", type=int, default=3, help="Number of instances to run")
    parser.add_argument("--sample-cfg", default=DEFAULT_SAMPLE_CFG, help="Sampling config")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--no-verify", action="store_true", help="Skip test verification")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions")
    parser.add_argument("--agentless", action="store_true",
                        help="Agentless mode: keyword-based file localization, "
                             "no tool loop, single-shot generation")
    parser.add_argument("--fill", action="store_true",
                        help="Fill mode: show the model the exact code to change, "
                             "ask only for the replacement. No tools, no search blocks.")
    parser.add_argument("--instance", type=str, help="Specific instance ID to run (e.g., sympy__sympy-18532)")
    args = parser.parse_args()

    args.model = os.path.abspath(args.model)
    model_name = os.path.basename(args.model).replace(".gguf", "")
    mode_str = "fill" if args.fill else ("agentless" if args.agentless else ("hybrid" if args.hybrid else "agentic"))
    print(f"Shimmer SWE-bench Eval [{mode_str}]: {model_name} | {args.sample} instances | {args.sample_cfg}")
    print(f"Timeout: {INSTANCE_TIMEOUT}s/instance | Verify: {not args.no_verify}\n")

    dataset = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
    if args.seed:
        random.seed(args.seed)
    
    if args.instance:
        tasks = [t for t in dataset if t["instance_id"] == args.instance]
        if not tasks:
            print(f"Instance {args.instance} not found in dataset!")
            sys.exit(1)
    else:
        tasks = random.sample(dataset, args.sample) if args.sample else dataset[3:30]
        
    print(f"Loaded {len(tasks)} tasks from SWE-bench Lite.")

    predictions = []
    if args.resume and os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE) as f:
            predictions = [json.loads(l) for l in f if l.strip()]
        print(f"Resuming with {len(predictions)} existing predictions.")

    os.makedirs(CLONE_BASE, exist_ok=True)
    results = []
    total_start = time.time()

    for idx, task in enumerate(tasks):
        iid = task["instance_id"]
        if any(p["instance_id"] == iid for p in predictions):
            print(f"[{idx+1}/{len(tasks)}] Skipping {iid} (already done)")
            continue

        print(f"\n{'='*60}\n[{idx+1}/{len(tasks)}] {iid}\n{'='*60}")
        clone_dir = os.path.join(CLONE_BASE, iid)

        # Clone & checkout (with retry on failures)
        for attempt in range(3):
            if not os.path.exists(os.path.join(clone_dir, ".git")):
                if os.path.exists(clone_dir): shutil.rmtree(clone_dir)
                print(f"Cloning {task['repo']} (attempt {attempt+1})...")
                result = subprocess.run(
                    ["git", "clone", f"https://github.com/{task['repo']}.git", clone_dir],
                    capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  Clone failed: {result.stderr.strip()[:120]}")
                    shutil.rmtree(clone_dir, ignore_errors=True)
                    if attempt < 2:
                        print(f"  Retrying in 5s...")
                        time.sleep(5)
                    continue
            result = subprocess.run(["git", "reset", "--hard", task["base_commit"]], cwd=clone_dir,
                                    capture_output=True, text=True)
            if result.returncode == 0:
                break
            print(f"  Checkout failed: {result.stderr.strip()[:120]}")
            shutil.rmtree(clone_dir)
        else:
            print(f"  FAIL: Could not clone {task['repo']} after 3 attempts — skipping")
            continue
        subprocess.run(["git", "clean", "-fd"], cwd=clone_dir, capture_output=True)

        # Scan repo & build prompt
        repo_hint = scan_repo(clone_dir)

        anchor = None
        if args.fill:
            prompt, anchor = build_fill_prompt(repo_hint, task["problem_statement"], clone_dir)
            no_tools = True
        elif args.agentless or args.hybrid:
            localized = localize_files_keyword(
                task["problem_statement"], clone_dir)
            prompt = build_agentless_prompt(
                repo_hint, task["problem_statement"], localized, clone_dir,
                hybrid=args.hybrid)
            no_tools = args.agentless
        else:
            prompt = build_prompt(repo_hint, task["problem_statement"])
            no_tools = False

        mode_label = "fill" if args.fill else ("agentless" if args.agentless else ("hybrid" if args.hybrid else "agentic"))
        print(f"  Context: {len(prompt)} chars ({mode_label})")

        # Run inference
        t0 = time.time()
        response = run_shimmer(prompt, clone_dir, args.model, args.sample_cfg,
                               no_tools=no_tools)
        wall = time.time() - t0

        # Save transcript for debugging
        transcript_path = os.path.join(SHIMMER_CWD, "data", f"transcript_{iid}.log")
        with open(transcript_path, "w") as tf:
            tf.write("========== PROMPT ==========\n")
            tf.write(prompt + "\n")
            tf.write("========== RESPONSE ==========\n")
            tf.write(response + "\n")
        print(f"  Transcript saved to: {transcript_path}")

        # Extract tool call count from response
        tool_count = response.count("[Tool '")

        # In fill mode, construct the edit from the anchor and model output
        if args.fill and anchor:
            model_replace = response
            # Strip everything before and including "Begin now."
            begin_idx = model_replace.rfind("Begin now.")
            if begin_idx != -1:
                model_replace = model_replace[begin_idx + len("Begin now."):]
            model_replace = model_replace.strip()
            if model_replace:
                synthetic = (
                    f'<edit file="{anchor["file"]}">\n'
                    f'<search>\n{anchor["original"]}\n</search>\n'
                    f'<replace>\n{model_replace}\n</replace>\n'
                    f'</edit>'
                )
                response = synthetic
                print(f"  Fill: constructed edit for {anchor['file']} "
                      f"({len(anchor['original'])} -> {len(model_replace)} chars)")

        # Extract patch
        patch, edit_stats = extract_patch(clone_dir, response)
        print(f"\n  Wall: {wall:.0f}s | Tools: {tool_count} | Edits: {edit_stats}")

        # Verify
        verified = None
        if patch.strip() and not args.no_verify:
            print(f"  Verifying with test suite...")
            passed, log = verify_patch(clone_dir, iid)
            verified = passed
            print(f"  Tests: {'PASS' if passed else 'FAIL' if passed is False else 'SKIP'}")

        # Collect stats from shimmer output
        tps_match = re.search(r'Performance:\s+([\d.]+)\s+TPS', response)
        tps = float(tps_match.group(1)) if tps_match else None

        result = {
            "instance_id": iid,
            "model": f"shimmer-{model_name}",
            "mode": "fill" if args.fill else ("agentless" if args.agentless else ("hybrid" if args.hybrid else "agentic")),
            "wall_time_s": round(wall, 1),
            "patch_chars": len(patch),
            "tps": tps,
            "tool_calls": tool_count,
            "edits_found": edit_stats["edits_found"],
            "edits_applied": edit_stats["edits_applied"],
            "edits_failed": edit_stats["edits_failed"],
            "tests_pass": verified,
            "model_patch": patch,
            "sample_cfg": args.sample_cfg,
        }
        results.append(result)

        # Write intermediate predictions
        predictions.append({"instance_id": iid, "model_patch": patch,
                            "model_name_or_path": f"shimmer-{model_name}",
                            "wall_time_s": round(wall, 1)})
        with open(PREDICTIONS_FILE, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

    # finally:
    #     if os.path.exists(clone_dir):
    #         shutil.rmtree(clone_dir)

    # ── Final Report ─────────────────────────────────────────────────────
    total_wall = time.time() - total_start
    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": model_name, "sample_cfg": args.sample_cfg, "instances": len(tasks),
                    "total_wall_s": round(total_wall, 1), "results": results}, f, indent=2)

    nonempty = [r for r in results if r["patch_chars"] > 0]
    passing = [r for r in results if r["tests_pass"] is True]
    print(f"\n{'='*60}")
    print(f"Eval complete: {len(results)} instances in {total_wall:.0f}s")
    print(f"Non-empty patches: {len(nonempty)}/{len(results)} ({len(nonempty)/max(len(results),1)*100:.0f}%)")
    print(f"Tests passing:      {len(passing)}/{len(results)} ({len(passing)/max(len(results),1)*100:.0f}%)")
    for r in results:
        tps_str = f"{r['tps']:.1f}" if r['tps'] else "?"
        status = "PASS" if r['tests_pass'] else "FAIL" if r['tests_pass'] is False else "N/A"
        print(f"  {r['instance_id']:35s} patch={r['patch_chars']:5d} chars  {r['wall_time_s']:5.0f}s  {tps_str} TPS  tools={r['tool_calls']}  tests={status}")
    print(f"\nResults saved: {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
