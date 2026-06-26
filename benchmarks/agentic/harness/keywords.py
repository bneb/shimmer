"""Keyword extraction and file localization for SWE-bench instances."""

import os, re, subprocess

_RESERVED_WORDS = {
    "and", "as", "assert", "break", "class", "continue", "def",
    "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with",
    "yield", "True", "False", "None", "self", "cls",
    "the", "for", "you", "are", "can", "has", "but", "was",
    "all", "any", "its", "use", "via", "one", "two", "new",
    "old", "see", "get", "set", "add", "that", "this", "with",
    "from", "have", "been", "were", "will", "would", "could",
    "should", "also", "into", "description", "therefore",
    "however", "python", "example", "consider", "following",
    "above", "below", "note", "notes",
}

_VENDORED_DIRS = {"site-packages", "node_modules", "venv",
                  ".env", "build", "dist", ".eggs", ".git",
                  "__pycache__", ".tox", ".mypy_cache"}


def _extract_keywords(issue_text):
    """Extract Python identifiers and file paths from issue text."""
    keywords = set()
    for pat in [r'["\']([\w./-]+\.py)["\']', r'([\w./-]+\.py)',
                r'\b([A-Z][a-zA-Z0-9]{2,})\b', r'\b([A-Z][A-Z0-9_]{3,})\b',
                r'\b([a-z]+(?:_[a-z0-9]+){1,})\b',
                r'\b([a-z_]+(?:\.[a-z_]+)+)\b',
                r'(?:from|import)\s+([\w.]+)']:
        for m in re.finditer(pat, issue_text):
            kw = m.group(1)
            if pat.startswith(r'\b([a-z]+(?:_'):
                if len(kw) < 5:
                    continue
            if pat.startswith(r'\b([a-z_]+(?:\.[a-z_]+)+)'):
                if not any(c.isalpha() for c in kw):
                    continue
            keywords.add(kw)
    # Filter path-like keywords for file existence
    result = set()
    for k in keywords:
        if "/" in k and k.endswith(".py"):
            result.add(k)
        elif k.lower() not in _RESERVED_WORDS and len(k) >= 3:
            result.add(k)
    return result


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
    for kw in sorted(keywords, key=len, reverse=True):
        if "/" in kw and kw.endswith(".py"):
            full = os.path.join(clone_dir, kw)
            if os.path.isfile(full):
                scores[kw] = scores.get(kw, 0) + 3
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


def localize_files_keyword(issue_text, clone_dir, max_files=5):
    """Return ranked file paths relevant to the issue, capped at max_files."""
    keywords = _extract_keywords(issue_text)
    if not keywords:
        return _walk_py_files(clone_dir)[:max_files]
    scores = _score_files_by_grep(keywords, clone_dir)
    if not scores:
        return _walk_py_files(clone_dir)[:max_files]
    ranked = sorted(scores.items(), key=lambda x: (-x[1], len(x[0])))
    return [p for p, _ in ranked[:max_files]]
