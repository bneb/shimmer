"""Patch extraction from model responses."""

import json, os, re, subprocess
from pathlib import Path

from .matching import apply_fuzzy_edit, _resolve_path

def extract_patch(clone_dir, response_text, exact_only=False):
    """Parse XML <edit> blocks from response, apply to disk, return unified diff."""
    pat = r'<edit\s+file="([^"]+)"\s*>\s*<search>\s*(.*?)\s*</search>\s*<replace>\s*(.*?)\s*</replace>\s*</edit>'
    edits = []
    for fp, s, r in re.findall(pat, response_text, re.DOTALL):
        edits.append((fp, s.replace('\\n', '\n'), r.replace('\\n', '\n')))

    # Deduplicate: keep only last occurrence of each (file, search, replace)
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
        resolved = _resolve_path(clone_dir, file_path, search, response_text)
        full = os.path.join(clone_dir, resolved)
        if not os.path.exists(full):
            stats["edits_failed"] += 1
            continue
        content = Path(full).read_text()
        new_content, method, _score = apply_fuzzy_edit(content, search, replace)
        if exact_only and method != "exact":
            stats["edits_failed"] += 1
            continue
        if method != "failed":
            Path(full).write_text(new_content)
            stats["edits_applied"] += 1
        else:
            stats["edits_failed"] += 1
    return _make_diff(clone_dir), stats


def _make_diff(clone_dir):
    """Run git diff to produce a unified diff of all changes."""
    import subprocess
    result = subprocess.run(["git", "diff"], cwd=clone_dir,
                            capture_output=True, text=True)
    return result.stdout
