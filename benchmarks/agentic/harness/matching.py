"""Fuzzy matching and patch extraction for model-generated XML edits."""

import difflib, os, re
from pathlib import Path


def apply_fuzzy_edit(content, search_str, replace_str):
    """Try exact, then normalized, then difflib match. Returns (content, method, score)."""
    search_str = search_str.strip()
    replace_str = replace_str.strip()
    if not search_str:
        return content, "failed", 0.0
    if search_str in content:
        return content.replace(search_str, replace_str), "exact", 1.0
    return _normalized_match(content, search_str, replace_str)


def _normalize(text):
    """Strip punctuation, underscores, and plural suffixes for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[_\-.]+', ' ', text)
    text = re.sub(r'\bies\b', 'y', text)
    text = re.sub(r'\b([a-z]{3,})s\b', r'\1', text)
    return re.sub(r'\s+', '', text)


def _best_normalized_match(content_lines, norm_content, search_lines, norm_search):
    """Find the best normalized match position. Returns (best_idx, best_score)."""
    best_score, best_idx = 0.0, None
    for i in range(len(content_lines) - len(search_lines) + 1):
        window = norm_content[i:i + len(search_lines)]
        score = sum(difflib.SequenceMatcher(None, w, s).ratio()
                    for w, s in zip(window, norm_search)) / len(search_lines)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx, best_score


def _normalized_match(content, search_str, replace_str):
    """Match by normalizing search and content lines, then replace original."""
    content_lines = content.split('\n')
    search_lines = [l for l in search_str.split('\n') if l.strip()]
    replace_lines = replace_str.split('\n')
    if not search_lines:
        return _difflib_match(content, search_str, replace_str)

    norm_content = [_normalize(l) for l in content_lines]
    norm_search = [_normalize(l) for l in search_lines]
    best_idx, best_score = _best_normalized_match(
        content_lines, norm_content, search_lines, norm_search)

    if best_idx is None or best_score <= 0.7:
        return _difflib_match(content, search_str, replace_str)

    anchor = content_lines[best_idx]
    indent = len(anchor) - len(anchor.lstrip())
    max_block = max(len(search_lines), len(replace_lines)) * 6
    end_idx = best_idx + len(search_lines)
    for i in range(best_idx + 1, min(best_idx + max_block, len(content_lines))):
        line = content_lines[i]
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            end_idx = i
            break
    else:
        end_idx = min(best_idx + max_block, len(content_lines))

    match_len = end_idx - best_idx
    if len(replace_lines) < match_len:
        replace_lines = (replace_lines
                         + content_lines[best_idx + len(replace_lines):end_idx])

    new_lines = content_lines[:best_idx] + replace_lines + content_lines[end_idx:]
    return '\n'.join(new_lines), "normalized", best_score


def _difflib_match(content, search_str, replace_str):
    """Anchor-based difflib matching for when the model can't reproduce exact lines."""
    content_lines = content.split('\n')
    search_lines = search_str.split('\n')
    replace_lines = replace_str.split('\n')

    anchor = next((l.strip() for l in search_lines if l.strip()), None)
    if not anchor:
        return content, "failed", 0.0

    best_idx, best_ratio = max(
        ((i, difflib.SequenceMatcher(None, cl.strip(), anchor).ratio())
         for i, cl in enumerate(content_lines)),
        key=lambda x: x[1])
    if best_ratio < 0.5:
        return content, "failed", 0.0

    anchor_indent = len(content_lines[best_idx]) - len(content_lines[best_idx].lstrip())
    if replace_lines and replace_lines[0].strip():
        repl_indent = len(replace_lines[0]) - len(replace_lines[0].lstrip())
        if repl_indent != anchor_indent and anchor_indent > 0:
            delta = anchor_indent - repl_indent
            for j in range(len(replace_lines)):
                if replace_lines[j].strip():
                    replace_lines[j] = (' ' * delta + replace_lines[j]) if delta > 0 \
                        else replace_lines[j][-delta:]

    max_block = max(len(search_lines), len(replace_lines)) * 6
    end_idx = best_idx + len(search_lines)
    for i in range(best_idx + 1, min(best_idx + max_block, len(content_lines))):
        if content_lines[i].strip() and \
           (len(content_lines[i]) - len(content_lines[i].lstrip())) <= anchor_indent:
            end_idx = i
            break
    else:
        end_idx = min(best_idx + max_block, len(content_lines))

    match_len = end_idx - best_idx
    if len(replace_lines) < match_len:
        replace_lines = (replace_lines
                         + content_lines[best_idx + len(replace_lines):end_idx])

    new_lines = content_lines[:best_idx] + replace_lines + content_lines[end_idx:]
    return '\n'.join(new_lines), "difflib", best_ratio


def _find_files_in_response(clone_dir, response_text):
    """Find .py file paths mentioned in the model's response text."""
    paths = set()
    for pat in [r'"arguments"\s*:\s*\[(.*?)\]', r"File '([^']+)' does not exist",
                r'([\w.-]+/)+[\w.-]+\.py']:
        for m in re.finditer(pat, response_text):
            vals = m.groups()
            if len(vals) == 1 and '/' in vals[0] and not vals[0].startswith('.'):
                if os.path.exists(os.path.join(clone_dir, vals[0])):
                    paths.add(vals[0])
    return [p for p in paths if os.path.exists(os.path.join(clone_dir, p))]


def _resolve_path(clone_dir, file_path, search_text, response_text):
    """Resolve a model-produced path to a real file on disk."""
    if os.path.exists(os.path.join(clone_dir, file_path)):
        return file_path
    candidates = _find_files_in_response(clone_dir, response_text)
    candidates.extend([p for p in candidates
                       if p.endswith(file_path) or file_path.endswith(p.split('/')[-1])])
    if candidates:
        scored = []
        for p in candidates:
            try:
                fc = Path(os.path.join(clone_dir, p)).read_text()
                s = 0
                if search_text.strip() and search_text.strip() in fc:
                    s += 1000
                first = next((l for l in search_text.split('\n') if l.strip()), '')
                if first and first.strip() in fc:
                    s += 200
                scored.append((p, s))
            except Exception:
                scored.append((p, 0))
        scored.sort(key=lambda x: -x[1])
        if scored[0][1] > 100:
            return scored[0][0]
    return file_path
