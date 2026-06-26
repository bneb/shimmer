"""CLI entry point and eval loop for SWE-bench evaluation."""

import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

from .engine import run_shimmer, GPU_NOISE_PATTERNS
from .keywords import localize_files_keyword
from .prompts import build_prompt, build_agentless_prompt
from .snippets import scan_repo
from .patch import extract_patch
from .reporting import build_result, format_summary, parse_tps, count_tool_calls, save_results

SHIMMER_CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLONE_BASE = "/tmp/swe_lite"
TEST_VERIFY_TIMEOUT = 120


def _clone_repo(task):
    """Clone a SWE-bench repo and checkout the base commit."""
    d = os.path.join(CLONE_BASE, task["instance_id"])
    for _ in range(3):
        if not os.path.exists(os.path.join(d, ".git")):
            if os.path.exists(d):
                shutil.rmtree(d)
            remote = task['repo'] if '/' not in task['repo'] or task['repo'].startswith('/') or task['repo'].startswith('.') \
                     else f"https://github.com/{task['repo']}.git"
            r = subprocess.run(["git", "clone", remote, d],
                               capture_output=True, text=True)
            if r.returncode != 0:
                time.sleep(5)
                continue
        r = subprocess.run(["git", "reset", "--hard", task["base_commit"]],
                           cwd=d, capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["git", "clean", "-fd"], cwd=d, capture_output=True)
            return d
        shutil.rmtree(d)
    return None


def _build_prompt(task, args, clone_dir):
    """Build the appropriate prompt based on mode flags."""
    repo_hint = scan_repo(clone_dir)
    if args.agentless or args.hybrid:
        localized = localize_files_keyword(task["problem_statement"], clone_dir)
        prompt = build_agentless_prompt(repo_hint, task["problem_statement"],
                                        localized, clone_dir, hybrid=args.hybrid)
        return prompt, not args.hybrid
    return build_prompt(repo_hint, task["problem_statement"]), False


def _ast_check(clone_dir, patch):
    """Auto-format and syntax-check Python files touched by the patch."""
    if not patch.strip():
        return
    for line in patch.split("\n"):
        for prefix in ["+++ b/", "--- a/"]:
            if not line.startswith(prefix):
                continue
            fpath = os.path.join(clone_dir, line[len(prefix):])
            if not fpath.endswith(".py") or not os.path.exists(fpath):
                continue
            try:
                subprocess.run(["python3", "-m", "autopep8", "--in-place",
                                "--aggressive", fpath], capture_output=True, timeout=30)
            except Exception:
                pass
            try:
                import ast
                with open(fpath) as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                print(f"  AST failed: {fpath} line {e.lineno}: {e.msg}")


def _run_one(task, args, model_name):
    """Evaluate a single SWE-bench instance."""
    clone_dir = _clone_repo(task)
    if not clone_dir:
        return None

    prompt, no_tools = _build_prompt(task, args, clone_dir)
    mode_label = "hybrid" if args.hybrid else ("agentless" if args.agentless else "agentic")
    print(f"  Context: {len(prompt)} chars ({mode_label})")

    t0 = time.time()
    response = run_shimmer(prompt, SHIMMER_CWD, args.model, args.sample_cfg,
                           no_tools=no_tools)
    wall = time.time() - t0

    for pattern in GPU_NOISE_PATTERNS:
        response = re.sub(rf'^.*{pattern}.*$', '', response, flags=re.MULTILINE)

    patch, edit_stats = extract_patch(clone_dir, response, exact_only=args.hybrid)
    _ast_check(clone_dir, patch)

    verified = None
    if patch.strip() and not args.no_verify:
        verified, _ = _verify_patch(clone_dir)

    tps = parse_tps(response)
    tool_count = count_tool_calls(response)

    return build_result(task["instance_id"], model_name, mode_label, wall,
                        patch, edit_stats, tps, tool_count, verified,
                        args.sample_cfg)


def _verify_patch(clone_dir):
    """Run pytest to verify the patch. Returns (passed, log)."""
    for cmd in [["python3", "-m", "pytest", "-x", "--timeout=60"],
                ["python3", "-m", "pytest", "-x"]]:
        try:
            p = subprocess.run(cmd, cwd=clone_dir, capture_output=True,
                               text=True, timeout=TEST_VERIFY_TIMEOUT)
            return p.returncode == 0, p.stdout[-500:] + "\n" + p.stderr[-500:]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return None, "No test runner found"


def run_eval(args):
    """Load SWE-bench tasks, run eval, save results."""
    from datasets import load_dataset
    import random
    dataset = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
    if args.instance:
        tasks = [t for t in dataset if t["instance_id"] == args.instance]
    else:
        random.seed(args.seed)
        tasks = random.sample(dataset, args.sample) if args.sample else dataset[3:30]

    print(f"Loaded {len(tasks)} tasks from SWE-bench Lite.")
    os.makedirs(CLONE_BASE, exist_ok=True)

    results = []
    t0 = time.time()
    for idx, task in enumerate(tasks):
        print(f"\n{'='*60}\n[{idx+1}/{len(tasks)}] {task['instance_id']}\n{'='*60}")
        r = _run_one(task, args, os.path.basename(args.model).replace(".gguf", ""))
        if r:
            results.append(r)

    total_wall = time.time() - t0
    for line in format_summary(results, total_wall):
        print(line)
    model_name = os.path.basename(args.model).replace(".gguf", "")
    mode = "hybrid" if args.hybrid else ("agentless" if args.agentless else "agentic")
    path = save_results(results, model_name, mode, args.sample_cfg, args.seed, os.path.join(SHIMMER_CWD, "data"))
    print(f"\nResults saved: {path}")
    return 0
