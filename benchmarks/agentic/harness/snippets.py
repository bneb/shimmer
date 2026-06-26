"""File snippet reading with line-number annotations for agentless prompts."""

import os
from pathlib import Path


def _annotate_lines(lines, start_num, _end_num):
    """Add right-hand line-number comments to code lines."""
    annotated = []
    max_line = start_num + len(lines) - 1
    num_width = len(str(max_line))
    for i, line in enumerate(lines):
        line_num = start_num + i
        annotated.append(f"{line:<80s} # L{line_num:{num_width}d}")
    return "\n".join(annotated)


def _read_file_snippet(fpath, clone_dir, keywords=None):
    """Read a file and return an annotated snippet.

    When keywords are provided, context windows are shown around
    keyword matches. Otherwise the file header (and optionally
    footer) are shown. Returns (snippet, label).
    """
    content = Path(os.path.join(clone_dir, fpath)).read_text()
    content_lines = content.split("\n")
    n_lines = len(content_lines)

    if n_lines <= 100:
        return _annotate_lines(content_lines, 1, n_lines), fpath

    match_lines = set()
    if keywords:
        for kw in keywords:
            kw_lower = kw.lower()
            for i, line in enumerate(content_lines):
                if kw_lower in line.lower():
                    match_lines.add(i)

    if match_lines:
        ctx_radius = 12
        windows = []
        for ml in sorted(match_lines):
            start = max(0, ml - ctx_radius)
            end = min(n_lines, ml + ctx_radius + 1)
            windows.append((start, end))
        # Merge overlapping windows
        merged = []
        for start, end in sorted(windows):
            if merged and start <= merged[-1][1] + 3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        parts = []
        for i, (start, end) in enumerate(merged):
            if i > 0:
                parts.append(f"\n... ({start - merged[i-1][1]} lines between matches) ...\n")
            parts.append(_annotate_lines(content_lines[start:end], start + 1, end))
        return "\n".join(parts), f"{fpath} (matched context)"

    if n_lines <= 300:
        top = _annotate_lines(content_lines[:30], 1, 30)
        bot = _annotate_lines(content_lines[-20:], n_lines - 19, n_lines)
        return (top + f"\n... ({n_lines - 50} lines omitted) ...\n" + bot,
                f"{fpath} (lines 1-30, {n_lines-19}-{n_lines})")

    top = _annotate_lines(content_lines[:30], 1, 30)
    return (top + f"\n... (file continues for {n_lines - 30} more lines) ...",
            f"{fpath} (first 30 of {n_lines} lines)")


def _build_header(repo_hint, problem):
    """Return the standard agentless prompt header lines."""
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
        "Line numbers appear as right-hand comments (e.g. '# L294').",
        "Copy code from the LEFT side of each line into <search>.",
        "Do NOT include the line-number comments in your <search> block.",
        "",
    ]


def _add_file_block(lines, fpath, snippet, label):
    """Append a formatted file block to lines. Returns char count."""
    block = (f"\n--- BEGIN FILE: {label} ---\n"
             + snippet
             + f"\n--- END FILE: {fpath} ---\n")
    lines.append(block.rstrip())
    return len(block)


def scan_repo(clone_dir):
    """Build a text summary of the repository structure."""
    top_entries = sorted(os.listdir(clone_dir))
    lines = ["## Repository Structure", "```"]
    for entry in top_entries:
        if entry.startswith(".") and entry != ".github":
            continue
        p = os.path.join(clone_dir, entry)
        etype = "[dir] " if os.path.isdir(p) else "[file]"
        lines.append(f"  {etype} {entry}")
    lines.append("```")
    # List source files
    src_files = []
    for root, dirs, files in os.walk(clone_dir):
        depth = root[len(clone_dir):].count(os.sep)
        if depth > 4:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                src_files.append(os.path.relpath(os.path.join(root, f), clone_dir))
    lines.append(f"\n## Source Files ({len(src_files)} .py files)")
    for f in sorted(src_files)[:50]:
        lines.append(f"  - {f}")
    if len(src_files) > 50:
        lines.append(f"  ... and {len(src_files) - 50} more")
    return "\n".join(lines)
