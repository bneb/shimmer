"""Pure formatting and summary functions — no I/O, 100% testable."""


def build_result(task_id, model_name, mode_label, wall, patch, edit_stats,
                 tps, tool_count, verified, sample_cfg):
    """Build a result dict from an eval run. Pure data transformation."""
    return {
        "instance_id": task_id, "model": f"shimmer-{model_name}",
        "mode": mode_label, "wall_time_s": round(wall, 1),
        "patch_chars": len(patch), "tps": tps,
        "tool_calls": tool_count,
        "edits_found": edit_stats["edits_found"],
        "edits_applied": edit_stats["edits_applied"],
        "edits_failed": edit_stats["edits_failed"],
        "tests_pass": verified, "model_patch": patch,
        "sample_cfg": sample_cfg,
    }


def format_summary(results, total_wall):
    """Format the final eval summary as a list of lines."""
    lines = []
    nonempty = [r for r in results if r["patch_chars"] > 0]
    passing = [r for r in results if r["tests_pass"] is True]
    walls = [r["wall_time_s"] for r in results]

    lines.append(f"{'='*60}")
    lines.append(f"Eval complete: {len(results)} instances in {total_wall:.0f}s")
    lines.append(f"Non-empty patches: {len(nonempty)}/{len(results)}")
    lines.append(f"Tests passing:      {len(passing)}/{len(results)}")
    if walls:
        lines.append(f"Wall clock: min={min(walls):.0f}s  "
                     f"median={sorted(walls)[len(walls)//2]:.0f}s  max={max(walls):.0f}s")

    for r in results:
        tps_str = f"{r['tps']:.1f}" if r['tps'] else "?"
        status = "PASS" if r['tests_pass'] else "FAIL" if r['tests_pass'] is False else "N/A"
        lines.append(f"  {r['instance_id']:35s} patch={r['patch_chars']:5d} chars  "
                     f"{r['wall_time_s']:5.0f}s  {tps_str} TPS  "
                     f"tools={r['tool_calls']}  tests={status}")
    return lines


def parse_tps(response):
    """Extract TPS from shimmer performance output. Returns float or None."""
    import re
    m = re.search(r'Performance:\s+([\d.]+)\s+TPS', response)
    return float(m.group(1)) if m else None


def count_tool_calls(response):
    """Count tool invocations in model output."""
    return response.count("[Tool '")


def save_results(results, model_name, mode, sample_cfg, seed, output_dir):
    """Write eval results to JSON. Pure I/O, testable with temp dirs."""
    import json, os
    data = {
        "model": model_name,
        "mode": mode,
        "sample_cfg": sample_cfg,
        "seed": seed,
        "instances": len(results),
        "total_wall_s": sum(r.get("wall_time_s", 0) for r in results),
        "results": results,
    }
    path = os.path.join(output_dir, "eval_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
