"""
watch_and_learn.py
==================
Daemon that watches ~/.claude/projects/ for new/completed sessions,
parses them as they close, and incrementally grows the training corpus.

When corpus reaches a milestone (default: every 25 new episodes), it
triggers a lightweight GEPA optimization pass to update the SKILL.md.

This creates a continuous improvement loop:
  new sessions → parse → add to corpus → re-optimize → update SKILL.md → better sessions

Usage
-----
    uv run python watch_and_learn.py \\
        --skill-file .claude/skills/banking/SKILL.md \\
        --project-filter banking \\
        --optimize-every 25 \\
        --reflection-lm anthropic/claude-sonnet-4-6

Run in background:
    nohup uv run python watch_and_learn.py ... &
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parse_session import DEFAULT_CLAUDE_DIR, build_corpus, parse_session


def _get_warm_seed(output_dir: Path, target: str, original_seed: str) -> str:
    """Return the previous run's best_candidate.md as the warm seed, or
    the original_seed if no prior output exists.

    The watcher writes the optimized SKILL.md to <output_dir>/<target>/best_candidate.md
    after each successful optimization. The next run reads this file (if it exists)
    as the seed, so GEPA refines from the prior best instead of rediscovering.

    Args:
        output_dir: Root output directory (e.g., Path("outputs")).
        target: Target name (e.g., "skill" or "banking" — derived from --skill-file stem
            or --project-filter).
        original_seed: The seed to fall back to when no prior best exists.

    Returns:
        The prior best_candidate.md content (if it exists), otherwise original_seed.
    """
    best_path = output_dir / target / "best_candidate.md"
    if best_path.exists():
        try:
            content = best_path.read_text(encoding="utf-8")
            print(f"[watch] Warm-restarting from previous best ({len(content)} chars)")
            return content
        except (OSError, UnicodeDecodeError) as exc:
            print(
                f"[watch] WARN: could not read {best_path} ({exc}); "
                f"falling back to original seed",
                file=sys.stderr,
            )
            return original_seed
    return original_seed


def watch_and_learn(
    skill_file: Path,
    claude_dir: Path,
    project_filter: str | None,
    optimize_every: int,
    reflection_lm: str,
    max_evals_per_run: int,
    poll_interval: float,
    output_dir: Path | None = None,
    target: str | None = None,
) -> None:
    # Output directory for warm-restart artifacts. Default: outputs/<skill_file.stem>
    if output_dir is None:
        output_dir = Path("outputs")
    if target is None:
        # Derive target from skill_file stem (e.g., .claude/skills/banking/SKILL.md → "SKILL")
        # Fall back to project_filter if skill_file has no recognizable stem.
        target = skill_file.stem or project_filter or "default"
    print(f"[watch] Warm-restart output: {output_dir}/{target}/best_candidate.md")
    print(f"[watch] Watching {claude_dir}/projects/ for new sessions")
    print(f"[watch] Skill file: {skill_file}")
    print(f"[watch] Will re-optimize every {optimize_every} new episodes\n")

    # Load existing corpus
    corpus: list[dict] = build_corpus(claude_dir, project_filter, min_tool_calls=2)
    print(f"[watch] Loaded {len(corpus)} existing episodes")

    # Track (path, size) pairs across polls. A file is "stable" when its size
    # matches the previous poll's size for two consecutive cycles.
    known_files: dict[str, int] = {}  # path → size at last poll
    stable_files: set[str] = set()  # paths stable across two consecutive polls
    # Pre-existing files (loaded from corpus at startup) are not in known_files
    # yet; treat them with the mtime fallback on first encounter.
    preexisting_paths: set[str] = {ep["source_path"] for ep in corpus}
    episodes_since_last_opt = 0
    current_skill = skill_file.read_text(encoding="utf-8") if skill_file.exists() else ""

    projects_dir = claude_dir / "projects"

    while True:
        # Scan for new JSONL files
        new_files: list[Path] = []
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                if project_filter and project_filter not in project_dir.name:
                    continue

                # Scan both top-level and subagent JSONL files. The subagent glob is wrapped
                # in try/except to tolerate unreadable subagents/ directories (e.g. on shared
                # filesystems with restrictive permissions) without crashing the long-running
                # daemon. Top-level glob is NOT wrapped because the project_dir is already
                # guarded by is_dir() and we want PermissionError on the project root to fail fast.
                def _safe_subagent_glob(subdir: Path) -> list[Path]:
                    try:
                        return list(subdir.glob("*.jsonl"))
                    except (PermissionError, OSError) as exc:
                        print(f"[watch] WARN: could not scan {subdir}: {exc}")
                        return []

                candidate_files: list[Path] = list(
                    project_dir.glob("*.jsonl")
                ) + _safe_subagent_glob(project_dir / "subagents")

                for jsonl in candidate_files:
                    path_str = str(jsonl)
                    try:
                        current_size = jsonl.stat().st_size
                    except (FileNotFoundError, OSError):
                        # File was deleted between glob and stat — skip silently
                        continue

                    prev_size = known_files.get(path_str)

                    if prev_size is None:
                        # First poll for this file. Use mtime guard for pre-existing files
                        # (where we have no size history); new files need a second poll to confirm stability.
                        if path_str in preexisting_paths:
                            # Pre-existing: trust the mtime guard (legacy behavior, conservative)
                            if time.time() - jsonl.stat().st_mtime > 10:
                                known_files[path_str] = current_size
                                if path_str not in stable_files:
                                    new_files.append(jsonl)
                        else:
                            # New file: record size, wait for second poll to confirm stability
                            known_files[path_str] = current_size
                    elif prev_size == current_size and path_str not in stable_files:
                        # Size unchanged from previous poll → mark as stable and add to new_files
                        stable_files.add(path_str)
                        new_files.append(jsonl)
                    else:
                        # Size changed: update record; wait for next poll to confirm stability
                        known_files[path_str] = current_size

        for path in new_files:
            path_str = str(path)
            if path_str in stable_files:
                continue
            try:
                ep = parse_session(path)
                if not ep["task_prompt"] or len(ep["tool_calls"]) < 2:
                    stable_files.add(path_str)
                    continue

                corpus.append(ep)
                stable_files.add(path_str)
                episodes_since_last_opt += 1

                outcome_icon = {
                    "success": "✓",
                    "error": "✗",
                    "interrupted": "⚡",
                    "unknown": "?",
                }.get(ep["outcome"], "?")
                print(
                    f"[watch] {outcome_icon} New episode: {ep['task_prompt'][:60]!r}"
                    f" | tools={len(ep['tool_calls'])} dur={ep['duration_s'] or '?':.0f}s"
                )

                if episodes_since_last_opt >= optimize_every:
                    print(
                        f"\n[watch] Milestone reached ({episodes_since_last_opt} new episodes). Re-optimizing..."
                    )
                    # Warm-restart: use prior best_candidate.md as the seed if it
                    # exists, otherwise fall back to current_skill (the current
                    # skill_file content).
                    # Note: current_skill is str for single-component targets (the only
                    # warm-restart case); multi-component targets skip warm restart.
                    assert isinstance(current_skill, str), "warm restart only supports str seed"
                    seed = _get_warm_seed(
                        output_dir=output_dir,
                        target=target,
                        original_seed=current_skill,
                    )
                    current_skill = _run_optimization(
                        corpus,
                        seed,
                        skill_file,
                        reflection_lm,
                        max_evals_per_run,
                        output_dir,
                        target,
                        max_backups=5,
                    )
                    episodes_since_last_opt = 0
                    print(f"[watch] SKILL.md updated ({len(current_skill)} chars)\n")

            except Exception as exc:
                print(f"[watch] Error parsing {path}: {exc}", file=sys.stderr)
                stable_files.add(path_str)

        time.sleep(poll_interval)


def _run_optimization(
    corpus: list[dict],
    current_skill: str | dict[str, str],
    skill_file: Path,
    reflection_lm: str,
    max_evals: int,
    output_dir: Path,
    target: str,
    max_backups: int = 5,
) -> str | dict[str, str]:
    import random

    from evaluator import make_replay_evaluator

    try:
        if importlib.util.find_spec("gepa.optimize_anything") is None:
            raise ImportError("gepa.optimize_anything not available")
        from gepa.optimize_anything import (
            EngineConfig,
            GEPAConfig,
            ReflectionConfig,
            optimize_anything,
        )
    except ImportError:
        print("[watch] gepa not installed — skipping optimization", file=sys.stderr)
        return current_skill

    rng = random.Random(42)
    shuffled = corpus[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * 0.75)
    train_set, val_set = shuffled[:t], shuffled[t:]

    if not train_set:
        return current_skill

    evaluate = make_replay_evaluator(train_set)

    result = optimize_anything(
        seed_candidate=current_skill,
        evaluator=evaluate,
        dataset=train_set,
        valset=val_set or train_set,
        objective=(
            "Optimize the SKILL.md file so Claude Code agents complete tasks "
            "faster with fewer errors and less context window usage."
        ),
        background=(
            "SKILL.md is injected at session start. Focus on: repo-specific commands, "
            "common error patterns, preferred tool sequences, and test strategies. "
            "Keep under 2000 tokens. Be specific and actionable."
        ),
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=max_evals, cache_evaluation=True),
            reflection=ReflectionConfig(reflection_lm=reflection_lm, reflection_minibatch_size=3),
        ),
    )

    best = result.best_candidate
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    # Write backup
    backup = skill_file.with_suffix(f".bak.{int(time.time())}.md")
    if skill_file.exists():
        backup.write_text(skill_file.read_text(encoding="utf-8"), encoding="utf-8")
    # Rotate old backups, keeping only the most recent max_backups
    backup_pattern = skill_file.with_suffix(".bak.*.md")
    backups = sorted(
        backup_pattern.parent.glob(backup_pattern.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_backup in backups[max_backups:]:
        old_backup.unlink(missing_ok=True)
    if isinstance(best, str):
        skill_file.write_text(best, encoding="utf-8")
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"[watch] Backed up to {backup.name}, wrote new skill ({best_score:.3f})")
    # Also write best_candidate.md to <output_dir>/<target>/ so the next watch
    # cycle can warm-restart from this run's best. Skip if the optimizer returned
    # a dict (multi-component target) — that path doesn't write best_candidate.md.
    if isinstance(best, str):
        best_path = output_dir / target / "best_candidate.md"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            best_path.write_text(best, encoding="utf-8")
        except OSError as exc:
            print(
                f"[watch] WARN: could not write {best_path} ({exc}); "
                f"warm-restart will fall back to disk-read of skill_file next run",
                file=sys.stderr,
            )
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-file", required=True, help="SKILL.md to optimize (path)")
    ap.add_argument("--project-filter", default=None)
    ap.add_argument("--claude-dir", default=str(DEFAULT_CLAUDE_DIR))
    ap.add_argument(
        "--optimize-every", type=int, default=25, help="Re-optimize after this many new episodes"
    )
    ap.add_argument("--reflection-lm", default="anthropic/claude-sonnet-4-6")
    ap.add_argument(
        "--max-evals-per-run",
        type=int,
        default=60,
        help="GEPA max_metric_calls for each incremental run",
    )
    ap.add_argument(
        "--poll-interval", type=float, default=15.0, help="Seconds between filesystem polls"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for warm-restart artifacts (default: ./outputs)",
    )
    ap.add_argument(
        "--target",
        default=None,
        help="Target name for warm-restart artifacts (default: skill_file.stem or project_filter)",
    )
    args = ap.parse_args()

    watch_and_learn(
        skill_file=Path(args.skill_file),
        claude_dir=Path(args.claude_dir),
        project_filter=args.project_filter,
        optimize_every=args.optimize_every,
        reflection_lm=args.reflection_lm,
        max_evals_per_run=args.max_evals_per_run,
        poll_interval=args.poll_interval,
        output_dir=args.output_dir,
        target=args.target,
    )
