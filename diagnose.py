"""
diagnose.py
-----------
Run this directly against your ~/.claude to see exactly what the parser
finds and why episodes get dropped.

    uv run python diagnose.py
    uv run python diagnose.py --project-filter pge
    uv run python diagnose.py --dump-sample          # print raw JSONL of first file
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude-dir", default=str(Path.home() / ".claude"))
    ap.add_argument("--project-filter", default=None)
    ap.add_argument(
        "--dump-sample",
        action="store_true",
        help="Print the first 15 lines of the first matching JSONL",
    )
    ap.add_argument("--min-tool-calls", type=int, default=2)
    args = ap.parse_args()

    claude_dir = Path(args.claude_dir)
    projects_dir = claude_dir / "projects"

    # ── 1. Basic path check ────────────────────────────────────────────────
    print("=" * 60)
    print("PATH CHECK")
    print("=" * 60)
    print(f"  claude_dir   : {claude_dir}  exists={claude_dir.exists()}")
    print(f"  projects_dir : {projects_dir}  exists={projects_dir.exists()}")

    if not projects_dir.exists():
        print("\nSTOP: projects_dir does not exist.")
        sys.exit(1)

    # ── 2. Project directory inventory ────────────────────────────────────
    print()
    print("=" * 60)
    print("PROJECT DIRECTORIES")
    print("=" * 60)
    all_dirs = sorted(d for d in projects_dir.iterdir() if d.is_dir())
    print(f"  Total project dirs: {len(all_dirs)}")

    matched_dirs = [
        d for d in all_dirs if args.project_filter is None or args.project_filter in d.name
    ]
    print(f"  After filter {args.project_filter!r}: {len(matched_dirs)} dirs")
    for d in matched_dirs:
        jsonls = list(d.glob("*.jsonl"))
        subagent_jsonls = (
            list((d / "subagents").glob("*.jsonl")) if (d / "subagents").exists() else []
        )
        print(f"    {d.name}")
        print(f"      .jsonl files: {len(jsonls)}   subagent .jsonl: {len(subagent_jsonls)}")

    if not matched_dirs:
        print("\nNo directories matched the filter.")
        print("All directory names:")
        for d in all_dirs:
            print(f"  {d.name}")
        sys.exit(1)

    # ── 3. Per-file diagnosis ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("FILE-LEVEL DIAGNOSIS")
    print("=" * 60)

    all_jsonl_paths: list[Path] = []
    for d in matched_dirs:
        all_jsonl_paths.extend(sorted(d.glob("*.jsonl")))
        sub = d / "subagents"
        if sub.exists():
            all_jsonl_paths.extend(sorted(sub.glob("*.jsonl")))

    print(f"  Total .jsonl files to parse: {len(all_jsonl_paths)}")

    if not all_jsonl_paths:
        print("\nSTOP: No .jsonl files found in matched directories.")
        sys.exit(1)

    # Optional raw dump
    if args.dump_sample:
        print()
        print("=" * 60)
        print(f"RAW SAMPLE: {all_jsonl_paths[0].name}")
        print("=" * 60)
        with open(all_jsonl_paths[0], encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= 15:
                    print("  ... (truncated at 15 lines)")
                    break
                try:
                    obj = json.loads(line)
                    # Pretty-print key fields only
                    t = obj.get("type", "?")
                    msg = obj.get("message", {})
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    content_types = []
                    if isinstance(content, list):
                        content_types = [b.get("type", "?") for b in content if isinstance(b, dict)]
                    elif isinstance(content, str):
                        content_types = [f"str:{content[:40]!r}"]
                    print(
                        f"  line {i + 1:2d}  type={t!r:12s}  role={role!r:12s}  "
                        f"content_block_types={content_types}"
                    )
                except json.JSONDecodeError as e:
                    print(f"  line {i + 1:2d}  JSON ERROR: {e}")

    # ── 4. Parse every file with detailed drop reasons ─────────────────────
    print()
    print("=" * 60)
    print("PARSE RESULTS")
    print("=" * 60)

    from parse_session import parse_session, read_jsonl

    kept = 0
    drop_reasons: Counter = Counter()
    type_counter: Counter = Counter()

    for path in all_jsonl_paths:
        raw = read_jsonl(path)

        # Count entry types to detect structural differences
        for entry in raw:
            type_counter[entry.get("type", "MISSING")] += 1

        ep = parse_session(path)
        n_lines = len(raw)
        n_tools = len(ep["tool_calls"])
        has_prompt = bool(ep["task_prompt"])

        reason = None
        if not has_prompt:
            reason = "no_task_prompt"
        elif n_tools < args.min_tool_calls:
            reason = f"tool_calls={n_tools} < min={args.min_tool_calls}"

        if reason:
            drop_reasons[reason] += 1
            print(
                f"  DROP  {path.name[:50]:50s}  lines={n_lines:4d}  "
                f"tools={n_tools}  prompt={has_prompt}  → {reason}"
            )
        else:
            kept += 1
            print(
                f"  KEEP  {path.name[:50]:50s}  lines={n_lines:4d}  "
                f"tools={n_tools}  outcome={ep['outcome']:12s}  "
                f"prompt={ep['task_prompt'][:50]!r}"
            )

    print()
    print(f"  KEPT: {kept}   DROPPED: {sum(drop_reasons.values())}")
    print(f"  Drop reasons: {dict(drop_reasons)}")

    print()
    print("=" * 60)
    print("ENTRY TYPE DISTRIBUTION (across all files)")
    print("=" * 60)
    for t, count in type_counter.most_common():
        print(f"  {t!r:20s}  {count:6d}")

    print()
    print("If all files show tool_calls=0, run with --dump-sample to inspect")
    print("the raw JSONL structure — the entry format may differ from what")
    print("the parser expects.")


if __name__ == "__main__":
    main()
