"""Prompt builders for agentic, agentless, and hybrid evaluation modes."""

from .snippets import _build_header, _add_file_block, _read_file_snippet
from .keywords import _extract_keywords

_PROMPT_CHARS_BUDGET = 48000

_AGENTLESS_FOOTER = [
    "",
    "INSTRUCTIONS:",
    "- All relevant code is shown above. You do NOT need to use any tools.",
    "- Do NOT output JSON tool calls. There are no tools available.",
    "- Provide your fix using XML <edit> tags.",
    "",
    "RULES:",
    "1. File paths must match a path shown above.",
    "2. Copy lines from the LEFT side of each code block into <search>.",
    "3. Do NOT include '# L294'-style line-number comments in your search.",
    "4. Your <search> must match exactly once in the file.",
    "",
    "Example:",
    "<edit file=\"django/conf/global_settings.py\">",
    "<search>",
    "FILE_UPLOAD_PERMISSIONS = None",
    "</search>",
    "<replace>",
    "FILE_UPLOAD_PERMISSIONS = 0o644",
    "</replace>",
    "</edit>",
    "",
    "Begin now.",
]

_HYBRID_FOOTER = [
    "",
    "You may use up to 3 read-only tools (rg, cat, ls) to verify exact",
    "lines before editing. Stop after 3 tools — produce your edit.",
    "",
    "RULES:",
    "1. File paths must match a path shown in the repository structure above.",
    "2. Keep your <search> SHORT. One distinctive line is often enough.",
    "3. Copy lines EXACTLY from tool output. Do not retype from memory.",
    "4. Do NOT include line-number comments (like '# L294') in your search.",
    "",
    "Example:",
    "<edit file=\"django/conf/global_settings.py\">",
    "<search>",
    "FILE_UPLOAD_PERMISSIONS = None",
    "</search>",
    "<replace>",
    "FILE_UPLOAD_PERMISSIONS = 0o644",
    "</replace>",
    "</edit>",
    "",
    "Begin now.",
]


def build_prompt(repo_hint, problem):
    """Agentic mode prompt with tool instructions."""
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
1. FIRST: Use rg to search for the function/class/setting name from the issue.
2. THEN: Use cat to read the files that contain matches.
3. LAST: Provide your fix using <edit> tags with EXACT lines copied from cat output.

Example:
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

Begin now.
"""


def build_agentless_prompt(repo_hint, problem, localized_files, clone_dir, hybrid=False):
    """Build agentless or hybrid prompt with source files inlined.

    Each file gets a share of the prompt budget.  In hybrid mode the
    footer instructs the model to use up to 3 tools before editing.
    """
    keywords = _extract_keywords(problem)
    lines = _build_header(repo_hint, problem)
    chars_used = len("\n".join(lines))
    remaining = _PROMPT_CHARS_BUDGET - chars_used - 400

    for i, fpath in enumerate(localized_files):
        per_file = max(800, remaining // max(len(localized_files) - i, 1))
        try:
            snippet, label = _read_file_snippet(fpath, clone_dir, keywords)
        except (OSError, UnicodeDecodeError):
            continue
        block_len = _add_file_block(lines, fpath, snippet, label)
        chars_used += block_len
        remaining -= block_len
        if remaining <= 0 and i + 1 < len(localized_files):
            lines.append(f"\n[Budget exhausted — {len(localized_files) - i - 1} "
                         f"more files not shown]")
            break

    footer = _HYBRID_FOOTER if hybrid else _AGENTLESS_FOOTER
    lines.extend(footer)
    return "\n".join(lines)
