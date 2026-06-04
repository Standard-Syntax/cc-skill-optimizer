"""
optimize.py
===========
Entry point for optimizing Claude Code skills and config files using GEPA + DSPy.

Seven targets:
  --target skill     Optimize a single SKILL.md file
  --target claude    Optimize the global CLAUDE.md
  --target agent     Optimize an AGENTS.md multi-agent orchestration config
  --target multi     Optimize ALL components in a skill folder simultaneously:
                     SKILL.md + references/*.{md,yaml,yml} + scripts/*.{py,sh}
                     GEPA evolves each file as a named component in one joint
                     optimization run, using dict[str,str] candidate mode.
  --target sections  Optimize WITHIN a single file at section level:
                     Each ## heading becomes its own GEPA component.
                     Sections are evolved independently and reassembled.
                     Use --section-depth to control nesting (default: 2 = ## headings).
  --target nested    Optimize NESTED CLAUDE.md/AGENTS.md files at different directory
                     levels (e.g., root CLAUDE.md + src/CLAUDE.md + test/CLAUDE.md).
                     Discovers all relevant files in a directory tree and optimizes
                     them as a joint multi-component candidate.

Two data source modes:
  (default)         Use real Claude Code session logs from ~/.claude/projects/
  --no-sessions     Use synthetic LLM-as-judge evaluation against hand-written
                    or auto-generated task descriptions. Works with zero sessions.

Usage examples
--------------
# Optimize a single SKILL.md from real session logs:
uv run python optimize.py \
    --target skill \
    --seed-file .claude/skills/banking/SKILL.md \
    --project-filter banking \
    --max-evals 150

# Optimize the ENTIRE skill folder (SKILL.md + references/ + scripts/) jointly:
uv run python optimize.py \
    --target multi \
    --skill-dir .claude/skills/banking \
    --project-filter banking \
    --max-evals 200

# Zero-session multi-component optimization:
uv run python optimize.py \
    --target multi \
    --skill-dir .claude/skills/banking \
    --no-sessions \
    --domain banking \
    --max-evals 80

# Optimize INDIVIDUAL SECTIONS of CLAUDE.md (each ## heading = separate component):
uv run python optimize.py \
    --target sections \
    --seed-file CLAUDE.md \
    --project-filter banking \
    --section-depth 2 \
    --max-evals 200

# Optimize SKILL.md sections including nested subsections (## + ### headings):
uv run python optimize.py \
    --target sections \
    --seed-file .claude/skills/banking/SKILL.md \
    --section-depth 3 \
    --no-sessions \
    --domain banking \
    --max-evals 150

# Optimize NESTED CLAUDE.md/AGENTS.md files at different directory levels:
uv run python optimize.py \
    --target nested \
    --nested-root . \
    --nested-depth 3 \
    --project-filter myproject \
    --max-evals 150

# Zero-session nested file optimization with custom patterns:
uv run python optimize.py \
    --target nested \
    --nested-root . \
    --nested-patterns CLAUDE.md,AGENTS.md \
    --nested-depth 2 \
    --no-sessions \
    --domain banking \
    --max-evals 100

# Zero-session single skill, built-in banking task library:
uv run python optimize.py \
    --target skill \
    --no-sessions \
    --domain banking \
    --seed-file skills/banking-analytics-seed.md \
    --max-evals 80

# Zero-session, structural scoring only (completely free):
uv run python optimize.py \
    --target skill \
    --no-sessions \
    --domain banking \
    --no-judge \
    --max-evals 60

# DSPy two-stage (BootstrapFewShot → GEPA):
uv run python optimize.py \
    --target skill \
    --no-sessions \
    --use-dspy \
    --domain banking \
    --seed-file skills/banking-analytics-seed.md \
    --max-evals 80

# Phase-2: seed from prior synthetic run, refine with real sessions:
uv run python optimize.py \
    --target skill \
    --seed-file outputs/phase1/skill/best_candidate.md \
    --project-filter banking \
    --max-evals 100 \
    --output-dir outputs/phase2/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Literal, cast

# Ensure WARNING level logging is enabled for the module
logging.basicConfig(level=logging.WARNING)

# Imports — handle both "python optimize.py" (from project root) and "python -m optimize" (from src/ dir)
try:
    # Shared DSPy infrastructure for run_dspy_gepa and run_dspy_native_gepa
    # (Phase 17.2: extracted from inline duplicates in both functions)
    from src.dspy_shared import (  # noqa: F401 — used by run_dspy_gepa / run_dspy_native_gepa
        SkillGuidedTask,  # noqa: F401 — base class inside SkillProgram (ruff can't trace dspy.Predict usage)
        SkillProgram,
        _ideal_completion_from_episode,  # noqa: F401 — used inside ep_to_example
        ep_to_example,
    )
    from src.evaluator import make_replay_evaluator
    from src.llm_config import (
        DEFAULT_MODEL,
        DIRECT_OUTPUT_PREFIX,
        EVAL_MAX_TOKENS,
        INFERENCE_PARAMS,
        REFLECTION_MODEL,
        THINKING_CONFIG_REFLECTION,
    )
    from src.parse_session import DEFAULT_CLAUDE_DIR, build_corpus
    from src.section_parser import (
        build_section_tree,
        load_sections_from_file,
        merge_sections,
        parse_sections,
        save_sections_to_file,
    )
    from src.synthetic_evaluator import (
        generate_tasks_for_domain,
        judge_score_task,
        load_task_library,
        make_dspy_synthetic_pipeline,
        make_synthetic_evaluator,
    )
except ImportError:
    # Fallback: add src/ to sys.path so internal module imports (e.g. utils in evaluator)
    # resolve correctly, then retry with bare module names
    _src_path = str(Path(__file__).parent / "src")
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)
    # Shared DSPy helpers (same as the try: block above)
    import llm_config  # noqa: F401 — provides DEFAULT_MODEL, REFLECTION_MODEL, THINKING_CONFIG_* constants
    from dspy_shared import (  # noqa: F401 — used by run_dspy_gepa / run_dspy_native_gepa
        SkillGuidedTask,  # noqa: F401 — base class inside SkillProgram (ruff can't trace dspy.Predict usage)
        SkillProgram,
        _ideal_completion_from_episode,  # noqa: F401 — used inside ep_to_example
        ep_to_example,
    )
    from evaluator import make_replay_evaluator  # noqa: F401
    from llm_config import (  # noqa: F401
        DEFAULT_MODEL,
        DIRECT_OUTPUT_PREFIX,
        EVAL_MAX_TOKENS,
        INFERENCE_PARAMS,
        REFLECTION_MODEL,
        THINKING_CONFIG_REFLECTION,
    )
    from parse_session import DEFAULT_CLAUDE_DIR, build_corpus  # noqa: F401
    from section_parser import (  # noqa: F401
        build_section_tree,
        load_sections_from_file,
        merge_sections,
        parse_sections,
        save_sections_to_file,
    )
    from synthetic_evaluator import (  # noqa: F401
        generate_tasks_for_domain,
        judge_score_task,
        load_task_library,
        make_dspy_synthetic_pipeline,
        make_synthetic_evaluator,
    )

# ---------------------------------------------------------------------------
# DSPy thinking guard
# ---------------------------------------------------------------------------


def _model_uses_thinking(model_str: str) -> bool:
    """
    Return True if the model is configured with extended thinking per llm_config.py.

    Extended thinking is INCOMPATIBLE with the temperature kwarg in dspy.LM calls.
    Per llm_config.py lines 99-105: passing temperature with thinking enabled
    causes an API error at the first request. This helper detects which models
    are currently wired to use thinking configs so that run_dspy_gepa can fail
    fast rather than crash at the first inference call.

    Current thinking-enabled model roster (update as new models are added):
      - minimax/minimax-m3          → THINKING_CONFIG_REFLECTION (thinking enabled)
      - minimax/minimax-m2.7        → THINKING_CONFIG_M2_7        (thinking enabled)
      - minimax/minimax-m2.7-highspeed → THINKING_CONFIG_M2_7   (thinking enabled)

    Models that use THINKING_CONFIG_EVAL (thinking disabled) are NOT included:
      - anthropic/claude-haiku-4-5-20251001
    """
    thinking_enabled_models = (
        "minimax/minimax-m3",
        "minimax/minimax-m2.7",
        "minimax/minimax-m2.7-highspeed",
    )
    return any(model_str.startswith(prefix) for prefix in thinking_enabled_models)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def split_corpus(
    episodes: list[dict],
    train_frac: float = 0.80,
    val_frac: float = 0.20,
    test_frac: float = 0.0,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split episodes into train/val/test sets.

    Default 80/20/0 matches the GEPA FAQ recommendation: "80/20 when total
    datapoints exceed 200; 50/50 when fewer than 200". The unused test slice
    in the legacy 70/20/10 default wasted episodes on small corpora.

    If train_frac + val_frac + test_frac < 1.0, the remaining episodes are
    silently dropped (the legacy behavior — the remainder was implicitly the
    test set). Set test_frac=0.10 explicitly to reproduce the legacy 70/20/10
    split.
    """
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_frac)
    v = int(n * val_frac)
    te = int(n * test_frac)
    # Clamp so indices never exceed n
    end_train = min(t, n)
    end_val = min(t + v, n)
    end_test = min(t + v + te, n)
    return shuffled[:end_train], shuffled[end_train:end_val], shuffled[end_val:end_test]


# ---------------------------------------------------------------------------
# Seed prompts
# ---------------------------------------------------------------------------

SEED_SKILL_MD = """\
# Repository Skills

## Overview
This skill provides guidance for working in this codebase.

## Key Patterns
- Follow existing code conventions
- Run tests after making changes
- Keep changes minimal and focused

## Testing
- Run the test suite before marking tasks complete
- Check for errors in output
"""

SEED_CLAUDE_MD = """\
# Project Instructions

You are working on a software project. Follow these principles:

1. Make focused, minimal changes
2. Always verify your changes with tests
3. Read existing code before writing new code
4. Handle errors explicitly
"""

SEED_AGENTS_MD = """\
# Agent Orchestration

## Roles
- **Researcher**: Gathers information and reads existing code
- **Implementer**: Makes focused code changes
- **Validator**: Runs tests and verifies outcomes

## Workflow
1. Researcher reads relevant files and summarizes the task
2. Implementer makes the minimal changes needed
3. Validator confirms tests pass
"""

OBJECTIVE_BY_TARGET: dict[str, str] = {
    "skill": (
        "Optimize the SKILL.md file so that Claude Code agents complete tasks "
        "faster with fewer errors, fewer tool calls, and less context window usage. "
        "The skill should encode repo-specific patterns, common pitfalls, preferred "
        "commands, and test-running strategies. Every line should be specific and actionable."
    ),
    "claude": (
        "Optimize the CLAUDE.md project instructions so that Claude Code agents "
        "understand the project's conventions, architecture, and workflows deeply "
        "enough to make correct, minimal changes on the first attempt."
    ),
    "agent": (
        "Optimize the AGENTS.md multi-agent orchestration configuration so that "
        "the agent pipeline resolves tasks with minimal redundancy, clear role "
        "boundaries, and efficient handoffs between sub-agents."
    ),
    "multi": (
        "Optimize the complete skill package — SKILL.md, reference documents in "
        "references/, and helper scripts in scripts/ — so that Claude Code agents "
        "complete tasks faster with fewer errors. Each component should reinforce the "
        "others: the skill file should reference the scripts; the reference docs should "
        "cover domain-specific knowledge the skill alludes to; the scripts should be "
        "correct, runnable, and match the patterns described in the skill file. "
        "Optimize all components jointly — improvements in one should guide improvements "
        "in the others."
    ),
    "sections": (
        "Optimize the INDIVIDUAL SECTIONS of the target markdown file so that each "
        "section is maximally effective at guiding the agent. Each ## heading is evolved "
        "independently — one section may become more specific while another becomes more "
        "concise. GEPA handles cross-section interactions via Pareto selection: candidates "
        "that improve some sections without degrading others survive. "
        "Overall goal: tasks complete faster with fewer errors and less context usage."
    ),
    "nested": (
        "Optimize a set of NESTED CLAUDE.md/AGENTS.md files at different directory levels "
        "so that each file provides the most relevant guidance for its scope while the root "
        "file provides project-wide context. The root file should be concise (global rules, "
        "architecture overview), while subdirectory files should be focused (area-specific "
        "patterns, commands, and pitfalls). Each file should be self-contained but reference "
        "siblings when relevant. GEPA evolves each file independently while preserving "
        "cross-references. Overall goal: agents get only the context relevant to their current "
        "work, reducing context waste and improving task success."
    ),
}

BACKGROUND_BY_TARGET: dict[str, str] = {
    "skill": (
        "A SKILL.md file is injected at the start of Claude Code sessions. "
        "It should contain: repository overview, key directories, preferred commands "
        "(build/test/lint), common error patterns and fixes, code conventions, and pitfalls. "
        "Keep under 2000 tokens. Use numbered lists and markdown headers. "
        "Avoid generic advice — every line must be specific and actionable."
    ),
    "claude": (
        "CLAUDE.md is Claude Code's primary project instruction file, read at session start. "
        "It shapes the agent's understanding of the project's goals, tech stack, workflows, "
        "and constraints. Be concise, specific, and prevent common mistakes."
    ),
    "agent": (
        "AGENTS.md defines a multi-agent pipeline: role descriptions, handoff protocols, "
        "shared context formats, and coordination rules. Eliminate duplicate work, define "
        "clear ownership, and specify exactly what each agent produces for the next stage."
    ),
    "multi": (
        "The candidate is a dict where each key is a file path and each value is that "
        "file's content. Components:\n"
        "  skill_md        — SKILL.md injected at session start (keep under 2000 tokens)\n"
        "  references/*    — Markdown/YAML reference docs for domain knowledge\n"
        "  scripts/*       — Python/shell helper scripts the agent can invoke\n\n"
        "Constraints:\n"
        "  - SKILL.md should reference scripts by name when relevant\n"
        "  - Scripts must be syntactically correct and runnable\n"
        "  - Reference docs should be factual and concise\n"
        "  - The whole package is read by the agent at session start"
    ),
    "sections": (
        "The candidate is a dict[str,str] where each key is a section identifier "
        "derived from its markdown heading:\n"
        "  section_overview     — ## Overview\n"
        "  section_commands     — ## Commands\n"
        "  section_commands.subsection_build — ### Build (nested within Commands)\n\n"
        "Each section is the FULL section body including its heading line.\n"
        "When writing new section content:\n"
        "  - Preserve the heading line at the top of the section content\n"
        "  - Use specific tool names, file paths, and command examples\n"
        "  - Keep each section focused on ONE topic\n"
        "  - Remove generic advice; replace with repo-specific guidance\n"
        "  - Target 50-300 words per section; avoid sections over 600 words\n"
        "  - Sections are evaluated as a whole document — coherence matters"
    ),
    "nested": (
        "The candidate is a dict[str,str] where each key is a file path relative to the "
        "project root:\n"
        "  CLAUDE.md        — root-level project instructions (global rules, architecture)\n"
        "  src/CLAUDE.md    — src/-specific rules (src language, frameworks, patterns)\n"
        "  test/CLAUDE.md   — test-specific rules (test commands, fixtures, mocking)\n"
        "  api/AGENTS.md    — api/-specific multi-agent orchestration\n\n"
        "When optimizing nested files:\n"
        "  - Root file should be concise: architecture overview, global conventions, "
        "coding standards that apply everywhere\n"
        "  - Subdirectory files should be focused: area-specific commands, patterns, "
        "pitfalls, and tools\n"
        "  - Avoid duplicating root content in subdirectory files — reference instead\n"
        "  - Each file should be 200-800 chars (focused, actionable)\n"
        "  - Root file can reference subdirectory files for details ('see src/CLAUDE.md')\n"
        "  - Claude Code loads root file always; loads subdirectory files when working there\n"
        "  - Cross-file references should be explicit (file paths, command names)"
    ),
}

SEED_BY_TARGET: dict[str, str] = {
    "skill": SEED_SKILL_MD,
    "claude": SEED_CLAUDE_MD,
    "agent": SEED_AGENTS_MD,
    # multi, sections, nested use dict-based seeds loaded from files;
    # fall back to skill/claude seed if somehow reached without a seed file
    "multi": SEED_SKILL_MD,
    "sections": SEED_SKILL_MD,
    "nested": SEED_CLAUDE_MD,
}


# ---------------------------------------------------------------------------
# GEPA tuning constants (used by run_gepa_optimize_anything and run_gepa_synthetic)
# ---------------------------------------------------------------------------

# Number of threads for parallel metric evaluation.
# Conservative default to respect MiniMax/MiniMax API rate limits.
GEPA_NUM_THREADS_DEFAULT: int = 4

# Reflection minibatch sizes — how many low-scoring episodes the reflection LM
# sees per mutation proposal. Larger = stronger signal but slower per iteration.
# Replay path: 50+ episode corpus → 5 spans failure modes well.
# Synthetic path: 20-task library → 4 is enough without exhausting the library.
GEPA_REFLECTION_MINIBATCH_REPLAY: int = 5
GEPA_REFLECTION_MINIBATCH_SYNTHETIC: int = 4

# Cache evaluation results across GEPA iterations for the same candidate.
GEPA_CACHE_EVALUATION: bool = True


# ---------------------------------------------------------------------------
# Multi-component skill folder loader
# ---------------------------------------------------------------------------

# File extensions considered for each sub-directory
_SKILL_DIR_PATTERNS: dict[str, list[str]] = {
    "": ["SKILL.md", "CLAUDE.md", "AGENTS.md"],  # root of skill dir
    "references": ["*.md", "*.yaml", "*.yml", "*.json"],
    "scripts": ["*.py", "*.sh"],
}

# Maximum bytes per component before truncating (GEPA passes these as strings;
# very large files hit context limits.  Truncate at a safe boundary.)
_MAX_COMPONENT_BYTES = 60_000


def load_skill_dir(skill_dir: Path) -> dict[str, str]:
    """
    Load all optimizable files from a Claude Code skill directory into a
    dict[str, str] candidate suitable for GEPA's multi-component mode.

    Directory layout expected:
        <skill_dir>/
            SKILL.md          → key: "skill_md"
            references/
                glossary.md   → key: "references/glossary.md"
                schema.yaml   → key: "references/schema.yaml"
            scripts/
                setup.py      → key: "scripts/setup.py"
                validate.sh   → key: "scripts/validate.sh"

    Any key with an empty string value is excluded so GEPA doesn't waste
    budget on blank files.
    """
    if not skill_dir.exists():
        raise FileNotFoundError(f"skill_dir not found: {skill_dir}")

    candidate: dict[str, str] = {}

    # Root-level named files
    for fname in _SKILL_DIR_PATTERNS[""]:
        fpath = skill_dir / fname
        if fpath.exists():
            key = fname.lower().replace(".", "_")  # "SKILL.md" → "skill_md"
            content = fpath.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                candidate[key] = content[:_MAX_COMPONENT_BYTES]

    # Sub-directories
    for subdir, patterns in _SKILL_DIR_PATTERNS.items():
        if not subdir:
            continue
        subpath = skill_dir / subdir
        if not subpath.exists():
            continue
        for pattern in patterns:
            for fpath in sorted(subpath.glob(pattern)):
                if not fpath.is_file():
                    continue
                key = f"{subdir}/{fpath.name}"
                content = fpath.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    candidate[key] = content[:_MAX_COMPONENT_BYTES]

    if not candidate:
        raise ValueError(
            f"No optimizable files found in {skill_dir}.\n"
            f"Expected SKILL.md, references/*.md/yaml, or scripts/*.py/sh"
        )

    return candidate


def save_multi_candidate(candidate: dict[str, str], skill_dir: Path, output_dir: Path) -> None:
    """
    Write each component of a dict candidate back to disk.

    For session-backed runs: writes to output_dir/<key> (doesn't touch the source).
    To deploy: copy output_dir/ back over skill_dir/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, content in candidate.items():
        out_path = output_dir / key
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    print(f"[multi] Saved {len(candidate)} components to {output_dir}/")
    for key in sorted(candidate):
        size = len(candidate[key])
        print(f"  {key}  ({size:,} chars)")


def make_multi_evaluator(
    base_evaluator,
    skill_dir: Path,
):
    """
    Wraps a base evaluator so it works with dict candidates.

    GEPA calls evaluate(candidate, example) where candidate is dict[str,str].

    Phase 16.3 change: We always concatenate ALL components into the scored
    text (was: only score the primary "skill_md" or "claude_md" component).
    This makes multi-component Pareto meaningful — the score reflects the
    full candidate package, not just one file.

    For multi-objective Pareto tracking via `frontier_type="hybrid"`, we
    also score each component individually and expose the per-component
    scores via `side_info["scores"]`. GEPA reads this shape natively
    (gepa>=0.1.1) and maintains per-objective frontiers automatically.
    """

    def evaluate(candidate: dict[str, str], example) -> tuple[float, dict]:
        # Always concatenate ALL components for scoring (was: only the primary one)
        combined_text = "\n\n".join(f"# {k}\n{v}" for k, v in candidate.items())

        # Score the full candidate. The combined call IS the canonical score; if it
        # fails, we fall back to score=0.0 with the error in side_info so the optimizer
        # can see what happened. Per-component scoring below is still best-effort.
        try:
            score, side_info = base_evaluator(combined_text, example)
        except Exception as exc:
            score = 0.0
            side_info = {
                "error": f"base_evaluator failed on combined candidate: {exc}",
                "components": {k: len(v) for k, v in candidate.items()},
                "n_components": len(candidate),
            }

        # Add per-component metadata to ASI so GEPA can diagnose per-file issues
        side_info.setdefault("components", {k: len(v) for k, v in candidate.items()})
        side_info.setdefault("n_components", len(candidate))

        # Per-component scores for multi-objective Pareto (gepa reads side_info["scores"])
        # for frontier_type="hybrid". Each component is scored independently against
        # the same example; the side_info from each per-component call is discarded —
        # only the scalar score is retained. The loop is best-effort: a single
        # failure sets that component's score to 0.0 without breaking the eval.
        component_scores: dict[str, float] = {}
        for key, text in candidate.items():
            try:
                comp_score, _ = base_evaluator(text, example)
                component_scores[key] = float(comp_score)
            except Exception:
                # Per-component scoring is best-effort; a single failure shouldn't
                # break the whole evaluation
                component_scores[key] = 0.0
        side_info["scores"] = component_scores

        return score, side_info

    return evaluate


# ---------------------------------------------------------------------------
# Nested file-level optimization (multiple CLAUDE.md/AGENTS.md at different directory levels)
# ---------------------------------------------------------------------------

# File names to search for in nested optimization
_NESTED_FILE_PATTERNS = ["CLAUDE.md", "AGENTS.md", "SKILL.md"]

# Maximum bytes per nested file before truncating
_MAX_NESTED_FILE_BYTES = 60_000


def load_nested_files(
    root_dir: Path,
    file_patterns: list[str] | None = None,
    max_depth: int = 3,
) -> dict[str, str]:
    """
    Discover and load all CLAUDE.md/AGENTS.md/SKILL.md files in a directory tree.

    Each file becomes a named component with a key based on its relative path
    from root_dir. This enables GEPA to optimize each file independently while
    maintaining the directory structure.

    Args:
        root_dir:       Root directory to search from
        file_patterns:  List of file names to search for (default: CLAUDE.md, AGENTS.md, SKILL.md)
        max_depth:      Maximum directory depth to search (default: 3)

    Returns:
        dict[str, str]: Mapping from relative path key → file content.
                       Key format: "CLAUDE.md" for root, "src/CLAUDE.md" for nested.

    Example:
        >>> nested = load_nested_files(Path("."))
        >>> # Returns: {"CLAUDE.md": "...", "src/CLAUDE.md": "...", "test/CLAUDE.md": "..."}
    """
    if file_patterns is None:
        file_patterns = _NESTED_FILE_PATTERNS

    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    candidate: dict[str, str] = {}

    # Collect all directories once (root + subdirs) rather than rebuilding per pattern
    all_dirs = [root_dir] + _walk_subdirs(root_dir, max_depth)

    # Walk the directory tree up to max_depth
    for pattern in file_patterns:
        for subdir in all_dirs:
            fpath = subdir / pattern
            if fpath.exists() and fpath.is_file():
                # Compute relative path from root_dir
                rel_path = fpath.relative_to(root_dir)
                key = str(rel_path).replace("\\", "/")  # Normalize for cross-platform

                # Skip if already loaded (handles case where root has the file)
                if key in candidate:
                    continue

                content = fpath.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    candidate[key] = content[:_MAX_NESTED_FILE_BYTES]

    if not candidate:
        raise ValueError(
            f"No nested files ({'/'.join(file_patterns)}) found in {root_dir} "
            f"up to depth {max_depth}.\n"
            f"Hint: Use --target skill or --target claude for single-file optimization."
        )

    return candidate


def _walk_subdirs(root: Path, max_depth: int) -> list[Path]:
    """Recursively walk subdirectories up to max_depth."""
    results = []

    def _walk(current: Path, depth: int) -> None:
        if depth >= max_depth:
            return
        try:
            for entry in current.iterdir():
                if entry.is_dir() and not entry.name.startswith("."):
                    # Skip common non-code directories
                    if entry.name in (
                        "node_modules",
                        ".venv",
                        "venv",
                        "__pycache__",
                        ".git",
                        "dist",
                        "build",
                    ):
                        continue
                    results.append(entry)
                    _walk(entry, depth + 1)
        except PermissionError:
            pass

    _walk(root, 0)
    return results


def save_nested_candidate(
    candidate: dict[str, str],
    root_dir: Path,
    output_dir: Path,
) -> None:
    """
    Write each component of a nested file candidate back to disk.

    Restores the original directory structure under output_dir/.
    To deploy: copy output_dir/ content back over the original root_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, content in candidate.items():
        out_path = output_dir / key
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    print(f"[nested] Saved {len(candidate)} files to {output_dir}/")
    for key in sorted(candidate):
        size = len(candidate[key])
        print(f"  {key}  ({size:,} chars)")


def make_nested_evaluator(
    base_evaluator,
    root_key: str = "CLAUDE.md",
):
    """
    Wraps a base evaluator for nested file candidates.

    GEPA calls evaluate(candidate, example) where candidate is dict[str,str].
    Each key is a file path (e.g., "CLAUDE.md", "src/CLAUDE.md").

    Phase 16.3 change: We always concatenate ALL nested files into the scored
    text (was: only score the root_key file as the primary entry point).
    This makes nested-file Pareto meaningful — the score reflects the full
    nested candidate package, not just the root file.

    For multi-objective Pareto tracking via `frontier_type="hybrid"`, we
    also score each file individually and expose the per-file scores via
    `side_info["scores"]`. GEPA reads this shape natively (gepa>=0.1.1).
    """

    def evaluate(candidate: dict[str, str], example) -> tuple[float, dict]:
        # Always concatenate ALL files for scoring (was: only the root_key file)
        combined_text = "\n\n".join(f"# {k}\n{v}" for k, v in candidate.items())

        # Score the full candidate. The combined call IS the canonical score; if it
        # fails, we fall back to score=0.0 with the error in side_info so the optimizer
        # can see what happened. Per-file scoring below is still best-effort.
        try:
            score, side_info = base_evaluator(combined_text, example)
        except Exception as exc:
            score = 0.0
            side_info = {
                "error": f"base_evaluator failed on combined candidate: {exc}",
                "nested_files": {k: len(v) for k, v in candidate.items()},
                "n_nested_files": len(candidate),
                "nested_file_keys": list(candidate.keys()),
            }

        # Add per-file metadata to ASI so GEPA can diagnose per-file issues
        side_info.setdefault("nested_files", {k: len(v) for k, v in candidate.items()})
        side_info.setdefault("n_nested_files", len(candidate))
        side_info.setdefault("nested_file_keys", list(candidate.keys()))

        # Per-file scores for multi-objective Pareto (gepa reads side_info["scores"])
        # for frontier_type="hybrid". Each file is scored independently against
        # the same example; the side_info from each per-file call is discarded.
        component_scores: dict[str, float] = {}
        for key, text in candidate.items():
            try:
                comp_score, _ = base_evaluator(text, example)
                component_scores[key] = float(comp_score)
            except Exception:
                # Per-file scoring is best-effort
                component_scores[key] = 0.0
        side_info["scores"] = component_scores

        return score, side_info

    return evaluate


# ---------------------------------------------------------------------------
# Section-level within-file optimization
# ---------------------------------------------------------------------------


def load_sections_from_seed_file(
    seed_file: Path,
    max_depth: int = 2,
) -> dict[str, str]:
    """
    Load a markdown file and parse it into sections for GEPA optimization.

    Each ## heading becomes its own component with a normalized key.
    """
    sections = load_sections_from_file(seed_file, max_depth=max_depth)
    if not sections:
        raise ValueError(
            f"No sections found in {seed_file}. File may be empty or have no ## headings."
        )
    print(f"[sections] Loaded {len(sections)} sections from {seed_file}:")
    for k in sorted(sections):
        print(f"  {k}  ({len(sections[k]):,} chars)")
    return sections


def make_section_evaluator(
    base_evaluator,
    max_depth: int = 2,
):
    """
    Wraps a base evaluator for section-level candidates.

    GEPA calls evaluate(candidate, example) where candidate is dict[str,str].
    We reassemble the sections into a full document for scoring.

    The assembled document is what gets evaluated — sections are evolved
    independently but scored as a whole document.
    """

    def evaluate(candidate: dict[str, str], example) -> tuple[float, dict]:
        # Reassemble sections into the full document
        doc_text = merge_sections(candidate, max_depth=max_depth)

        score, side_info = base_evaluator(doc_text, example)

        # Add per-section metadata to ASI
        side_info["n_sections"] = len(candidate)
        side_info["section_keys"] = list(candidate.keys())
        side_info["section_lengths"] = {k: len(v) for k, v in candidate.items()}

        return score, side_info

    return evaluate


def save_sections_output(
    sections: dict[str, str],
    output_dir: Path,
    source_file: Path | None = None,
    max_depth: int = 2,
) -> None:
    """
    Write section components to output directory.

    Writes each section as <key>.md and the merged document as best_candidate.md.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write each section as its own file (for inspection)
    for key, content in sorted(sections.items()):
        safe_key = key.replace(".", "_").replace("/", "_")
        out_path = output_dir / f"{safe_key}.md"
        out_path.write_text(content, encoding="utf-8")

    # Write the merged document
    original = (
        source_file.read_text(encoding="utf-8", errors="replace")
        if source_file and source_file.exists()
        else ""
    )
    merged = merge_sections(sections, source_text=original, max_depth=max_depth)
    (output_dir / "best_candidate.md").write_text(merged, encoding="utf-8")

    print(f"[sections] Saved {len(sections)} sections + merged document to {output_dir}/")
    for k in sorted(sections):
        print(f"  {k}  ({len(sections[k]):,} chars)")


# ---------------------------------------------------------------------------
# GEPA optimize_anything runner
# ---------------------------------------------------------------------------


def make_length_constrained_proposer(max_chars: int = 2000):
    """Returns a custom GEPA proposer that injects a character-limit constraint
    into the reflection prompt. Prevents prompt bloat from accumulating across
    iterations — the single largest overfitting risk in long optimization runs.

    Reference: Decagon ablation study (March 2026) — 1,500-char constraint
    achieved 4× compression, -0.8% performance, +generalization.

    Args:
        max_chars: The character limit to enforce on the SKILL.md output. Default
            2000 matches the 2K-token SKILL.md target.

    Returns:
        A callable `proposer(current_candidate, reflective_dataset, components_to_update)`
        that appends a length constraint to each item's feedback in
        reflective_dataset, then returns None to delegate to GEPA's default proposer.
    """
    from gepa.optimize_anything import ProposalFn  # lazy import; not needed at module load

    def proposer(current_candidate, reflective_dataset, components_to_update):
        # The default proposer's output is string | dict[str, str]. We inject the
        # length constraint into the reflection context by appending to each
        # item's feedback before proposing. Returning None tells GEPA to use its
        # default proposer with the enriched data.
        constraint_note = (
            f"\n\nIMPORTANT: The skill MUST be under {max_chars} characters. "
            f"Prefer concise, specific bullets over verbose paragraphs. "
            f"If the current draft exceeds {max_chars} chars, aggressively prune "
            f"generic advice and keep only repo-specific, actionable guidance."
        )
        for item in reflective_dataset:
            item["feedback"] = item.get("feedback", "") + constraint_note
        # Returning None tells GEPA to use its default proposer with the (mutated) data
        _ = ProposalFn  # silence the unused-import lint; the type is for documentation
        return None

    return proposer


def run_gepa_optimize_anything(
    seed_candidate: str | dict[str, str],
    train_set: list[dict],
    val_set: list[dict],
    objective: str,
    background: str,
    max_metric_calls: int,
    task_lm: str,
    reflection_lm: str,
    output_dir: Path,
    use_llm_judge: bool,
    judge_lm: str,
    skill_dir: Path | None = None,
    max_depth: int = 2,
    is_sections: bool = False,
    sections_seed_file: Path | None = None,
    is_nested: bool = False,
    nested_root: Path | None = None,
    frontier_type: str = "instance",
    max_evals_override: int | None = None,
    target: str | None = None,
    proposer: str = "batch",
    max_skill_chars: int = 2000,
) -> str | dict[str, str]:
    """
    Run gepa.optimize_anything and return the best candidate.

    seed_candidate can be:
      str           → single SKILL.md (original behaviour)
      dict[str,str] → multi-component (--target multi), sections (--target sections),
                       or nested (--target nested);
                       each key is a file path (multi/nested) or section name (sections);
                       GEPA evolves all simultaneously.

    proposer: Reflection proposer strategy. 'loop' uses gskill's create_loop_proposer
              (one episode per reflection call); 'batch' (default) sends all evaluation
              results at once.
    """
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    if proposer == "loop":
        print(
            "[gepa] --proposer loop requested. Note: gskill's create_loop_proposer is "
            "only available when using gepa.gskill.gskill.train_optimize_anything wrapper. "
            "Falling back to standard gepa batch proposer."
        )

    is_multi = isinstance(seed_candidate, dict)

    base_evaluate = make_replay_evaluator(
        episodes=train_set,
        use_llm_judge=use_llm_judge,
        judge_lm=judge_lm,
        target=target,
    )

    if is_sections:
        evaluate = make_section_evaluator(base_evaluate, max_depth=max_depth)
    elif is_nested:
        evaluate = make_nested_evaluator(base_evaluate, root_key="CLAUDE.md")
    elif is_multi:
        evaluate = make_multi_evaluator(base_evaluate, skill_dir or Path("."))
    else:
        evaluate = base_evaluate

    mode_label = (
        f"sections ({len(seed_candidate)} sections)"
        if is_sections
        else (
            f"nested ({len(seed_candidate)} files)"
            if is_nested
            else (f"multi ({len(seed_candidate)} components)" if is_multi else "single")
        )
    )
    print(f"\n[gepa] Starting optimize_anything  mode={mode_label}")
    print(f"  train={len(train_set)} val={len(val_set)} max_metric_calls={max_metric_calls}")
    print(f"  task_lm={task_lm}  reflection_lm={reflection_lm}")
    if is_multi or is_sections:
        for k in sorted(seed_candidate):
            k_chars = (
                len(seed_candidate[k]) if isinstance(seed_candidate, dict) else len(seed_candidate)
            )
            print(f"    {k}  ({k_chars:,} chars)")
    print()

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluate,
        dataset=train_set,
        valset=val_set or train_set,
        objective=objective,
        background=background,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=max_evals_override
                if max_evals_override is not None
                else max_metric_calls,
                cache_evaluation=True,
                parallel=True,  # Replay evaluator is stateless → parallelise metric evaluation across episodes.
                max_workers=GEPA_NUM_THREADS_DEFAULT,
                frontier_type=cast(
                    Literal["instance", "objective", "hybrid", "cartesian"], frontier_type
                ),
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                # Replay path: 50+ episode corpus → 5 spans failure modes well;
                # smaller minibatch would miss rare-but-important failure patterns.
                reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_REPLAY,
                # Phase 18.1: length-constrained proposer to prevent prompt bloat
                custom_candidate_proposer=make_length_constrained_proposer(max_skill_chars),
            ),
        ),
    )

    best = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\n[gepa] Best score: {best_score:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_sections and isinstance(best, dict):
        save_sections_output(best, output_dir, source_file=sections_seed_file, max_depth=max_depth)
        (output_dir / "gepa_result.json").write_text(
            json.dumps(
                {
                    "best_score": best_score,
                    "n_evals": max_metric_calls,
                    "mode": "sections",
                    "max_depth": max_depth,
                    "n_sections": len(best),
                    "section_keys": list(best.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif is_multi and isinstance(best, dict):
        save_multi_candidate(best, skill_dir or Path("."), output_dir)
        (output_dir / "gepa_result.json").write_text(
            json.dumps(
                {
                    "best_score": best_score,
                    "n_evals": max_metric_calls,
                    "components": list(best.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif is_nested and isinstance(best, dict):
        save_nested_candidate(best, nested_root or Path("."), output_dir)
        (output_dir / "gepa_result.json").write_text(
            json.dumps(
                {
                    "best_score": best_score,
                    "n_evals": max_metric_calls,
                    "mode": "nested",
                    "nested_files": list(best.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        # str candidate — original behaviour
        text = best if isinstance(best, str) else str(best)
        (output_dir / "best_candidate.md").write_text(text, encoding="utf-8")
        (output_dir / "gepa_result.json").write_text(
            json.dumps({"best_score": best_score, "n_evals": max_metric_calls}, indent=2),
            encoding="utf-8",
        )
        print(f"[gepa] Saved to {output_dir}/best_candidate.md")

    return best


# ---------------------------------------------------------------------------
# DSPy GEPA runner
# ---------------------------------------------------------------------------


def run_dspy_gepa(
    seed_candidate: str | dict[str, str],
    train_set: list[dict],
    val_set: list[dict],
    objective: str,
    max_metric_calls: int,
    task_lm: str,
    reflection_lm: str,
    output_dir: Path,
) -> str:
    """
    Use dspy.MIPROv2 to optimize the skill as a DSPy Predict signature.
    Returns the optimized skill content as a string.

    NOTE — DSPy path output format:
      This DSPy path extracts DSPy's internal `signature.instructions` field
      after optimization, NOT a SKILL.md file. So the output format differs
      from GEPA's `optimize_anything` path which writes `best_candidate.md`.
      The file written by this function is `best_candidate_dspy.md`.
    """
    import dspy
    from dspy import MIPROv2

    # Normalize dict seed_candidate to str (DSPy only supports string-based skill content)
    if isinstance(seed_candidate, dict):
        seed_candidate = next(iter(seed_candidate.values()))

    # Guard: temperature is incompatible with extended thinking per llm_config.py.
    # The dspy.LM constructor below passes temperature=0.7 (task) and temperature=1.0
    # (reflection). If either model is configured with extended thinking, the API call
    # will fail at runtime. This guard fails fast with a clear error rather than
    # crashing at the first inference request.
    #
    # NOTE: This guard WILL fire for the current default models
    # (task_lm=DEFAULT_MODEL="minimax/minimax-m2.7-highspeed",
    #  reflection_lm=REFLECTION_MODEL="minimax/minimax-m3") because both are
    # thinking-enabled models. This is the intended safety behavior: the user must
    # explicitly acknowledge the conflict by either disabling thinking on those models
    # or removing the temperature kwarg from the dspy.LM calls below.
    if _model_uses_thinking(task_lm) or _model_uses_thinking(reflection_lm):
        raise ValueError(
            f"DSPy runner cannot combine temperature with extended thinking. "
            f"task_lm={task_lm!r} reflection_lm={reflection_lm!r} — at least one model "
            f"uses extended thinking. Either disable thinking on these models, or modify "
            f"run_dspy_gepa to construct dspy.LM objects WITHOUT the temperature kwarg "
            f"when thinking is enabled. See src/llm_config.py lines 99-105 for the conflict."
        )

    # Configure DSPy LMs
    task_lm_obj = dspy.LM(model=task_lm, temperature=0.7, max_tokens=4096)
    reflect_lm_obj = dspy.LM(model=reflection_lm, temperature=1.0, max_tokens=16000)

    dspy_train = [ep_to_example(ep) for ep in train_set if ep.get("task_prompt")]
    dspy_val = [ep_to_example(ep) for ep in val_set if ep.get("task_prompt")]

    # Metric: dspy 3.x GEPAFeedbackMetric signature — returns dspy.Prediction so the
    # reflection LM can iterate on failure modes via the feedback field. This is
    # the SAME contract as run_dspy_native_gepa, for consistency.
    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
    ) -> dspy.Prediction:
        # Find the matching episode for this gold example
        for ep in train_set + val_set:
            if ep.get("task_prompt", "")[:100] == gold.task_prompt[:100]:
                from evaluator import score_episode

                score, side_info = score_episode(ep)
                feedback = side_info.get("feedback", "")
                return dspy.Prediction(score=score, feedback=feedback)
        return dspy.Prediction(
            score=0.5,
            feedback="No matching episode found for gold example.",
        )

    program = SkillProgram(seed_candidate)
    # dspy 3.0: per-module LM injection (replaces legacy dspy.configure global config)
    program.set_lm(task_lm_obj)

    optimizer = MIPROv2(
        metric=metric,
        prompt_model=reflect_lm_obj,
        num_threads=4,
        auto="medium",
    )

    print("\n[dspy.MIPROv2] Compiling program")
    print(f"  train={len(dspy_train)} val={len(dspy_val)}\n")

    optimized = optimizer.compile(
        program,
        trainset=dspy_train,
        valset=dspy_val,
    )

    # Extract optimized skill from the compiled program
    # GEPA optimizes the predictor's instructions — retrieve them
    best_skill = seed_candidate  # fallback
    try:
        pred = optimized.predictor
        # dspy.GEPA puts the optimized instructions in the signature's field descriptions
        if hasattr(pred, "signature"):
            sig = pred.signature
            instructions = getattr(sig, "instructions", None)
            if instructions:
                best_skill = instructions
    except Exception:
        logger = logging.getLogger(__name__)
        logger.warning(
            "[run_dspy_gepa] Could not extract optimized instructions from DSPy MIPROv2 — using seed_candidate"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_candidate_dspy.md").write_text(best_skill, encoding="utf-8")
    try:
        optimized.save(str(output_dir / "dspy_program.json"))
    except Exception as exc:
        logger.warning("[run_dspy_gepa] DSPy stage-1 initialization failed: %s", exc)
    print(f"[dspy.MIPROv2] Saved to {output_dir}/best_candidate_dspy.md")
    return best_skill


def run_dspy_native_gepa(
    seed_candidate: str | dict[str, str],
    train_set: list[dict],
    val_set: list[dict],
    objective: str,
    max_metric_calls: int,
    task_lm: str,
    reflection_lm: str,
    output_dir: Path,
) -> str:
    """
    Use dspy.GEPA (the native multi-objective optimizer introduced in dspy 3.0)
    to optimize the skill as a DSPy Predict signature. Returns the optimized skill
    content as a string.
    """
    import dspy
    from dspy import GEPA

    # Normalize dict seed_candidate to str (DSPy only supports string-based skill content)
    if isinstance(seed_candidate, dict):
        seed_candidate = next(iter(seed_candidate.values()))

    # Guard: temperature is incompatible with extended thinking per llm_config.py.
    # The dspy.LM constructor below passes temperature=0.7 (task) and temperature=1.0
    # (reflection). If either model is configured with extended thinking, the API call
    # will fail at runtime. This guard fails fast with a clear error rather than
    # crashing at the first inference request.
    #
    # NOTE: This guard WILL fire for the current default models
    # (task_lm=DEFAULT_MODEL="minimax/minimax-m2.7-highspeed",
    #  reflection_lm=REFLECTION_MODEL="minimax/minimax-m3") because both are
    # thinking-enabled models. This is the intended safety behavior: the user must
    # explicitly acknowledge the conflict by either disabling thinking on those models
    # or removing the temperature kwarg from the dspy.LM calls below.
    if _model_uses_thinking(task_lm) or _model_uses_thinking(reflection_lm):
        raise ValueError(
            f"DSPy runner cannot combine temperature with extended thinking. "
            f"task_lm={task_lm!r} reflection_lm={reflection_lm!r} — at least one model "
            f"uses extended thinking. Either disable thinking on these models, or modify "
            f"run_dspy_native_gepa to construct dspy.LM objects WITHOUT the temperature "
            f"kwarg when thinking is enabled. See src/llm_config.py lines 99-105 for "
            f"the conflict."
        )

    # Configure DSPy LMs
    task_lm_obj = dspy.LM(model=task_lm, temperature=0.7, max_tokens=4096)
    reflect_lm_obj = dspy.LM(model=reflection_lm, temperature=1.0, max_tokens=16000)

    dspy_train = [ep_to_example(ep) for ep in train_set if ep.get("task_prompt")]
    dspy_val = [ep_to_example(ep) for ep in val_set if ep.get("task_prompt")]

    # Metric: dspy 3.x GEPAFeedbackMetric signature — returns dspy.Prediction so the
    # reflection LM can iterate on failure modes via the feedback field.
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
        for ep in train_set + val_set:
            if ep.get("task_prompt", "")[:100] == gold.task_prompt[:100]:
                from evaluator import score_episode

                score, side_info = score_episode(ep)
                feedback = side_info.get("feedback", "")
                return dspy.Prediction(score=score, feedback=feedback)
        return dspy.Prediction(score=0.5, feedback="No matching episode found for gold example.")

    program = SkillProgram(seed_candidate)
    # dspy 3.0: per-module LM injection (replaces legacy dspy.configure global config)
    program.set_lm(task_lm_obj)

    optimizer = GEPA(
        metric=metric,
        auto="medium",
        max_metric_calls=max_metric_calls,
        reflection_lm=reflect_lm_obj,
        num_threads=4,
        track_stats=True,
    )

    print("\n[dspy.GEPA] Compiling program (dspy 3.x native multi-objective)")
    print(f"  train={len(dspy_train)} val={len(dspy_val)} max_metric_calls={max_metric_calls}\n")

    optimized = optimizer.compile(program, trainset=dspy_train, valset=dspy_val)

    # Extract optimized skill from the compiled program
    # dspy.GEPA optimizes the predictor's instructions — retrieve them
    best_skill = seed_candidate  # fallback
    try:
        pred = optimized.predictor
        if hasattr(pred, "signature"):
            sig = pred.signature
            instructions = getattr(sig, "instructions", None)
            if instructions:
                best_skill = instructions
    except Exception:
        logger = logging.getLogger(__name__)
        logger.warning(
            "[run_dspy_native_gepa] Could not extract optimized instructions from "
            "dspy.GEPA — using seed_candidate"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_candidate_dspy.md").write_text(best_skill, encoding="utf-8")
    try:
        optimized.save(str(output_dir / "dspy_program.json"))
    except Exception as exc:
        logger.warning("[run_dspy_native_gepa] DSPy stage-1 initialization failed: %s", exc)
    print(f"[dspy.GEPA] Saved to {output_dir}/best_candidate_dspy.md")
    return best_skill


# ---------------------------------------------------------------------------
# No-sessions GEPA runner (synthetic dataset)
# ---------------------------------------------------------------------------


def run_gepa_synthetic(
    seed_candidate: str | dict[str, str],
    train_tasks: list[dict],
    val_tasks: list[dict],
    objective: str,
    background: str,
    max_metric_calls: int,
    reflection_lm: str,
    output_dir: Path,
    judge_lm: str,
    use_judge: bool,
    max_depth: int = 2,
    is_sections: bool = False,
    is_nested: bool = False,
    nested_root: Path | None = None,
    frontier_type: str = "instance",
    max_evals_override: int | None = None,
    max_skill_chars: int = 2000,
) -> str | dict[str, str]:
    """Run gepa.optimize_anything with synthetic task-based evaluation."""
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    base_evaluate = make_synthetic_evaluator(
        task_library=train_tasks,
        judge_lm=judge_lm,
        use_judge=use_judge,
    )

    if is_sections:
        evaluate = make_section_evaluator(base_evaluate, max_depth=max_depth)
    elif is_nested:
        evaluate = make_nested_evaluator(base_evaluate, root_key="CLAUDE.md")
    else:
        evaluate = base_evaluate

    is_dict = isinstance(seed_candidate, dict)
    mode = "LLM judge" if use_judge else "structural only (free)"
    mode_label = (
        f"sections ({len(seed_candidate)} sections)"
        if is_sections
        else (f"multi ({len(seed_candidate)} components)" if is_dict else f"single ({mode})")
    )
    print(f"\n[gepa-synthetic] Starting optimize_anything (evaluator: {mode_label})")
    print(f"  train_tasks={len(train_tasks)} val_tasks={len(val_tasks)}")
    print(f"  max_metric_calls={max_metric_calls}  reflection_lm={reflection_lm}\n")
    if is_dict:
        for k in sorted(seed_candidate):
            print(f"    {k}  ({len(seed_candidate[k]):,} chars)")

    result = optimize_anything(
        seed_candidate=seed_candidate
        if (isinstance(seed_candidate, str) and seed_candidate.strip())
        else seed_candidate,
        evaluator=evaluate,
        dataset=train_tasks,
        valset=val_tasks or train_tasks,
        objective=objective,
        background=background,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=max_evals_override
                if max_evals_override is not None
                else max_metric_calls,
                cache_evaluation=True,
                parallel=True,  # Synthetic evaluator is stateless → parallelise metric evaluation across tasks.
                max_workers=GEPA_NUM_THREADS_DEFAULT,
                frontier_type=cast(
                    Literal["instance", "objective", "hybrid", "cartesian"], frontier_type
                ),
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                # Synthetic path: 20-task library → 4 is enough without exhausting the library.
                reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_SYNTHETIC,
                # Phase 18.1: length-constrained proposer to prevent prompt bloat
                custom_candidate_proposer=make_length_constrained_proposer(max_skill_chars),
            ),
        ),
    )

    best = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\n[gepa-synthetic] Best score: {best_score:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_sections and isinstance(best, dict):
        save_sections_output(best, output_dir, source_file=None, max_depth=max_depth)
        (output_dir / "gepa_result.json").write_text(
            json.dumps(
                {
                    "best_score": best_score,
                    "n_evals": max_metric_calls,
                    "mode": "synthetic_sections",
                    "max_depth": max_depth,
                    "n_sections": len(best),
                    "n_train_tasks": len(train_tasks),
                    "section_keys": list(best.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif is_nested and isinstance(best, dict):
        save_nested_candidate(best, nested_root or Path("."), output_dir)
        (output_dir / "gepa_result.json").write_text(
            json.dumps(
                {
                    "best_score": best_score,
                    "n_evals": max_metric_calls,
                    "mode": "synthetic_nested",
                    "nested_files": list(best.keys()),
                    "n_train_tasks": len(train_tasks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        text = best if isinstance(best, str) else str(best)
        (output_dir / "best_candidate.md").write_text(text, encoding="utf-8")
        (output_dir / "gepa_result.json").write_text(
            json.dumps(
                {
                    "best_score": best_score,
                    "n_evals": max_metric_calls,
                    "mode": "synthetic",
                    "n_train_tasks": len(train_tasks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[gepa-synthetic] Saved to {output_dir}/best_candidate.md")
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    # Explicit opt-in: configure the MiniMax Anthropic-compatible endpoint for
    # all subsequent litellm calls in this process. Tests that import this
    # module without invoking main() will use the real Anthropic endpoint.
    from llm_config import configure

    try:
        configure()
    except OSError as exc:
        print(f"[optimize] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    ap = argparse.ArgumentParser(
        description="Optimize Claude Code SKILL.md / CLAUDE.md / AGENTS.md via GEPA + DSPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # What to optimize
    ap.add_argument(
        "--target",
        choices=["skill", "claude", "agent", "multi", "sections", "nested"],
        default="skill",
        help="'sections' optimizes within-file sections (each ## heading = separate component). "
        "'multi' optimizes all components in --skill-dir jointly. "
        "'nested' discovers and optimizes CLAUDE.md/AGENTS.md files at different directory levels.",
    )
    ap.add_argument(
        "--seed-file",
        default=None,
        help="Existing file to use as seed (single targets). Pass nothing for seedless.",
    )
    ap.add_argument(
        "--skill-dir",
        default=None,
        help="Skill folder to optimize jointly (--target multi). "
        "Contains SKILL.md, references/, scripts/.",
    )
    ap.add_argument(
        "--nested-root",
        default=None,
        help="Root directory for --target nested. Discovers all CLAUDE.md/AGENTS.md files "
        "in subdirectories up to depth 3. Defaults to current directory.",
    )
    ap.add_argument(
        "--nested-depth",
        type=int,
        default=3,
        metavar="N",
        help="Maximum directory depth for --target nested (default: 3).",
    )
    ap.add_argument(
        "--nested-patterns",
        default=None,
        help="Comma-separated file patterns for --target nested (default: CLAUDE.md,AGENTS.md,SKILL.md).",
    )
    ap.add_argument(
        "--section-depth",
        type=int,
        default=2,
        metavar="N",
        help="Maximum heading level to treat as optimizable sections (default: 2). "
        "2 = ## headings only, 3 = ## and ### headings, etc. "
        "Only applies to --target sections.",
    )

    # Data source — mutually exclusive modes
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--no-sessions",
        action="store_true",
        help="Use synthetic task-based evaluation (no Claude Code sessions needed)",
    )

    # Session-mode args
    ap.add_argument(
        "--project-filter",
        default=None,
        help="Substring to filter Claude Code project directories (session mode)",
    )
    ap.add_argument("--claude-dir", default=str(DEFAULT_CLAUDE_DIR))
    ap.add_argument("--min-tool-calls", type=int, default=2)
    ap.add_argument("--train-frac", type=float, default=0.80,
                    help="Fraction of episodes for training (default 0.80; GEPA FAQ recommends 80/20).")
    ap.add_argument("--val-frac", type=float, default=0.20,
                    help="Fraction of episodes for validation (default 0.20).")
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="Fraction of episodes for held-out test set (default 0.0; was 0.10 in legacy 70/20/10 split).")
    ap.add_argument(
        "--max-skill-chars",
        type=int,
        default=2000,
        help="Character limit for the optimized SKILL.md (default 2000; Decagon study found 1500-char constraint improves generalization)",
    )

    # Synthetic-mode args
    ap.add_argument(
        "--domain",
        default="banking",
        help="Domain name for built-in task library (no-sessions mode)",
    )
    ap.add_argument(
        "--domain-description",
        default=None,
        help="Free-text description of your codebase. If provided, LLM generates tasks.",
    )
    ap.add_argument(
        "--generate-tasks",
        type=int,
        default=0,
        help="Number of tasks to generate via LLM (0 = use built-in library)",
    )
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge, use structural scoring only (completely free, no API calls)",
    )

    # Optimization engine
    ap.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Maximum GEPA metric calls (default: 100 for phase 1, 60 for phase 2)",
    )
    ap.add_argument(
        "--task-lm", default=DEFAULT_MODEL, help="LM for task evaluation and task generation"
    )
    ap.add_argument(
        "--reflection-lm",
        default=REFLECTION_MODEL,
        help="LM for GEPA reflection / mutation proposals",
    )
    ap.add_argument(
        "--use-dspy",
        action="store_true",
        help="Use DSPy (BootstrapFewShot → GEPA) instead of gepa.optimize_anything",
    )
    ap.add_argument(
        "--dspy-backend",
        choices=["mipro", "native-gepa"],
        default="mipro",
        help=(
            "Backend for the DSPy optimizer in --use-dspy session-backed mode. "
            "'mipro' uses the legacy dspy.MIPROv2 path (default, backward compatible). "
            "'native-gepa' uses dspy.GEPA (dspy 3.0+ native multi-objective optimizer) "
            "with reflective feedback. Has no effect on --target synthetic mode "
            "(make_dspy_synthetic_pipeline always uses MIPROv2)."
        ),
    )
    ap.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM judge in session mode (ignored in --no-sessions, always uses judge)",
    )
    ap.add_argument("--judge-lm", default=DEFAULT_MODEL)
    ap.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Optimization phase: 1 (synthetic exploration, instance frontier, 100 evals) or 2 (session-backed refinement, instance frontier, 60 evals). Default: 1",
    )
    ap.add_argument(
        "--hybrid-frontier",
        action="store_true",
        default=False,
        help="Use gepa's frontier_type='hybrid' (multi-objective Pareto). Requires the new "
        "'scores' dict in side_info added in task 9.2; without those keys gepa 0.1.1 raises "
        "ValueError. Default: False (uses 'instance' frontier).",
    )
    ap.add_argument(
        "--time-split",
        action="store_true",
        default=False,
        help="Sort episodes chronologically by timestamp before train/val split "
        "(avoids data leakage per the Arize Prompt Learning paper). "
        "Default: False (random split).",
    )
    ap.add_argument(
        "--proposer",
        choices=["batch", "loop"],
        default="batch",
        help="Reflection proposer strategy. 'loop' (gskill-style) processes one episode "
        "per reflection call, producing more detailed skills at higher API cost. "
        "'batch' (default) sends all evaluation results at once.",
    )

    # Output
    ap.add_argument("--output-dir", default="outputs/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Phase-based GEPA configuration
    if args.hybrid_frontier:
        _gepa_frontier_type = "hybrid"
        # Hybrid requires side_info["scores"] from make_replay_evaluator.
        # task 9.2 added this to score_episode side_info.
        print(
            "[main] --hybrid-frontier enabled: requires side_info['scores'] from make_replay_evaluator. "
            "If you wrote a custom evaluator, ensure it returns a 'scores' dict."
        )
        _gepa_default_max_evals = 100
    elif args.phase == 1:
        _gepa_frontier_type = "instance"
        _gepa_default_max_evals = 100
    else:  # phase 2
        # Note: prior versions used "population" for phase 2, but gepa 0.1.1's
        # FrontierType is Literal["instance", "objective", "hybrid", "cartesian"]
        # (gepa/core/state.py:22). "population" is invalid.
        _gepa_frontier_type = "instance"
        _gepa_default_max_evals = 60

    # Honor user --max-evals override; fall back to phase default only if user did not pass --max-evals
    effective_max_evals = args.max_evals if args.max_evals is not None else _gepa_default_max_evals

    # ------------------------------------------------------------------ #
    # Load seed — multi-target uses a dict from --skill-dir,
    # single targets use a string from --seed-file
    # ------------------------------------------------------------------ #
    skill_dir: Path | None = None
    nested_root: Path | None = None
    seed_candidate: str | dict[str, str]

    if args.target == "multi":
        if not args.skill_dir:
            print("[main] ERROR: --target multi requires --skill-dir <path/to/skill/folder>")
            sys.exit(1)
        skill_dir = Path(args.skill_dir)
        try:
            seed_candidate = load_skill_dir(skill_dir)
            print(f"[main] Multi-component seed loaded from {skill_dir}:")
            for k, v in seed_candidate.items():
                print(f"  {k}  ({len(v):,} chars)")
        except (FileNotFoundError, ValueError) as e:
            print(f"[main] ERROR: {e}")
            sys.exit(1)
    elif args.target == "sections":
        # Section-level optimization: parse file into ## heading components
        if not args.seed_file:
            print("[main] ERROR: --target sections requires --seed-file <path/to/markdown/file>")
            sys.exit(1)
        seed_path = Path(args.seed_file)
        if not seed_path.exists():
            print(f"[main] ERROR: --seed-file '{args.seed_file}' not found")
            sys.exit(1)
        try:
            seed_candidate = load_sections_from_seed_file(seed_path, max_depth=args.section_depth)
        except ValueError as e:
            print(f"[main] ERROR: {e}")
            sys.exit(1)
    elif args.target == "nested":
        # Nested file optimization: discover CLAUDE.md/AGENTS.md files in directory tree
        nested_root = Path(args.nested_root) if args.nested_root else Path.cwd()
        patterns = args.nested_patterns.split(",") if args.nested_patterns else None
        try:
            seed_candidate = load_nested_files(
                nested_root,
                file_patterns=patterns,
                max_depth=args.nested_depth,
            )
            print(f"[main] Nested files seed loaded from {nested_root}:")
            for k, v in seed_candidate.items():
                print(f"  {k}  ({len(v):,} chars)")
        except (FileNotFoundError, ValueError) as e:
            print(f"[main] ERROR: {e}")
            sys.exit(1)
    elif args.seed_file and Path(args.seed_file).exists():
        seed_candidate = Path(args.seed_file).read_text(encoding="utf-8")
        print(f"[main] Seed loaded from {args.seed_file} ({len(seed_candidate)} chars)")
    else:
        seed_candidate = SEED_BY_TARGET.get(args.target, SEED_SKILL_MD)
        if args.seed_file:
            print(f"[main] Seed file '{args.seed_file}' not found — using built-in default")
        else:
            print(f"[main] No seed file — using built-in default for target={args.target}")

    output_dir = Path(args.output_dir) / args.target

    # ------------------------------------------------------------------ #
    # Objectives and background
    # ------------------------------------------------------------------ #
    objective = OBJECTIVE_BY_TARGET.get(args.target, OBJECTIVE_BY_TARGET["skill"])
    background = BACKGROUND_BY_TARGET.get(args.target, BACKGROUND_BY_TARGET["skill"])

    # -------------------------------------------------------------------------
    # M2.7-specific optimization guidance
    # -------------------------------------------------------------------------
    # MiniMax M2.7 weaknesses to mitigate:
    # - Tunnel-vision: tends to stick to first approach when debugging fails
    # - Verbosity: 4x more tokens than average
    # - "Read everything first": causes timeouts in time-sensitive workflows
    #
    # When using M2.7 for skill generation, apply these directives:
    #   "If the first approach fails, explicitly try a fundamentally different strategy."
    #   "Do not repeat the same approach with minor variations."
    #   "Be concise — target 50% fewer tokens than typical output."
    #   "Read only the most relevant files first; defer deep exploration unless needed."

    # ------------------------------------------------------------------ #
    # Branch: no-sessions vs session mode
    # ------------------------------------------------------------------ #
    if args.no_sessions:
        # -------- SYNTHETIC / NO-SESSIONS PATH -------- #
        print("\n[main] Mode: SYNTHETIC (no Claude Code sessions required)")

        # Build task library
        if args.generate_tasks > 0 or args.domain_description:
            n_tasks = args.generate_tasks or 12
            desc = args.domain_description or f"A software project in the {args.domain} domain."
            task_library = generate_tasks_for_domain(
                domain=args.domain,
                domain_description=desc,
                judge_lm=args.task_lm,
                n=n_tasks,
            )
        else:
            task_library = load_task_library(args.domain)

        # Normalize to list[dict] — generate_tasks_for_domain may return dict | list[dict]
        task_library = list(task_library) if isinstance(task_library, dict) else task_library

        if not task_library:
            print("[main] ERROR: No tasks available. Use --domain or --domain-description.")
            sys.exit(1)

        # Split tasks
        rng = random.Random(args.seed)
        shuffled = task_library[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        t = max(3, int(n * 0.75))
        train_tasks = shuffled[:t]
        val_tasks = shuffled[t:] or shuffled[:3]  # fallback for tiny libraries
        print(f"[main] Task split: train={len(train_tasks)} val={len(val_tasks)}")

        use_judge = not args.no_judge

        if args.use_dspy and args.target == "sections":
            print("[main] ERROR: DSPy mode (--use-dspy) does not yet support --target sections.")
            print("  Sections mode requires gepa.optimize_anything (dict candidate) mode.")
            print("  Use --target skill or --target claude with --use-dspy instead.")
            sys.exit(1)

        if args.use_dspy:
            # DSPy two-stage: BootstrapFewShot → GEPA
            best = make_dspy_synthetic_pipeline(
                task_library=task_library,
                seed_candidate=seed_candidate,
                task_lm=args.task_lm,
                reflection_lm=args.reflection_lm,
                max_bootstrap_evals=max(10, effective_max_evals // 4),
                max_gepa_evals=effective_max_evals,
                output_dir=str(output_dir),
            )
        else:
            best = run_gepa_synthetic(
                seed_candidate=seed_candidate,
                train_tasks=train_tasks,
                val_tasks=val_tasks,
                objective=objective,
                background=background,
                max_metric_calls=effective_max_evals,
                reflection_lm=args.reflection_lm,
                output_dir=output_dir,
                judge_lm=args.judge_lm,
                use_judge=use_judge,
                max_depth=args.section_depth,
                is_sections=(args.target == "sections"),
                is_nested=(args.target == "nested"),
                nested_root=nested_root,
                frontier_type=_gepa_frontier_type,
                max_evals_override=effective_max_evals,
                max_skill_chars=args.max_skill_chars,
            )

    else:
        # -------- SESSION-BACKED PATH -------- #
        print(f"\n[main] Mode: SESSION-BACKED (reading from {args.claude_dir})")

        episodes = build_corpus(
            Path(args.claude_dir),
            project_filter=args.project_filter,
            min_tool_calls=args.min_tool_calls,
            sort_by_time=args.time_split,
        )
        if not episodes:
            print(
                "[main] ERROR: No episodes found.\n"
                "  Check --claude-dir and --project-filter.\n"
                "  To run without sessions: add --no-sessions"
            )
            sys.exit(1)

        train_set, val_set, test_set = split_corpus(
            episodes, args.train_frac, args.val_frac, args.test_frac, args.seed
        )

        # With very few episodes, the percentage-based split produces unusable
        # sets (e.g. 1 train / 0 val).  Override: use all episodes for both
        # train and val when corpus is tiny, and warn the user.
        if len(episodes) < 10:
            print(
                f"[main] WARNING: Only {len(episodes)} episodes found. "
                f"This is a very thin corpus — results will be limited.\n"
                f"  For better results: run Claude Code on more tasks in this project,\n"
                f"  or drop --project-filter to use all available sessions,\n"
                f"  or use --no-sessions with the built-in task library.\n"
                f"  Continuing with all {len(episodes)} episodes as both train and val."
            )
            train_set = episodes
            val_set = episodes
            test_set = []

        print(
            f"[main] Corpus split: train={len(train_set)} val={len(val_set)} test={len(test_set)}"
        )

        if args.use_dspy:
            # --dspy-backend selects between MIPROv2 (legacy) and dspy.GEPA (dspy 3.0+ native)
            if args.dspy_backend == "native-gepa":
                best = run_dspy_native_gepa(
                    seed_candidate=seed_candidate,
                    train_set=train_set,
                    val_set=val_set,
                    objective=objective,
                    max_metric_calls=effective_max_evals,
                    task_lm=args.task_lm,
                    reflection_lm=args.reflection_lm,
                    output_dir=output_dir,
                )
            else:  # "mipro" (default)
                best = run_dspy_gepa(
                    seed_candidate=seed_candidate,
                    train_set=train_set,
                    val_set=val_set,
                    objective=objective,
                    max_metric_calls=effective_max_evals,
                    task_lm=args.task_lm,
                    reflection_lm=args.reflection_lm,
                    output_dir=output_dir,
                )
        else:
            best = run_gepa_optimize_anything(
                seed_candidate=seed_candidate,
                train_set=train_set,
                val_set=val_set,
                objective=objective,
                background=background,
                max_metric_calls=effective_max_evals,
                task_lm=args.task_lm,
                reflection_lm=args.reflection_lm,
                output_dir=output_dir,
                use_llm_judge=args.use_llm_judge,
                judge_lm=args.judge_lm,
                skill_dir=skill_dir,
                max_depth=args.section_depth,
                is_sections=(args.target == "sections"),
                sections_seed_file=Path(args.seed_file) if args.seed_file else None,
                is_nested=(args.target == "nested"),
                nested_root=Path(args.nested_root) if args.nested_root else None,
                frontier_type=_gepa_frontier_type,
                max_evals_override=effective_max_evals,
                target=args.target,
                proposer=args.proposer,
                max_skill_chars=args.max_skill_chars,
            )

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    if isinstance(best, dict):
        if args.target == "sections":
            print(f"Section-level result ({len(best)} sections) in {output_dir}/")
            for k in sorted(best):
                print(f"  {k}  ({len(best[k]):,} chars)")
            print(f"\nMerged document: {output_dir}/best_candidate.md")
        elif args.target == "nested":
            print(f"Nested files result ({len(best)} files) in {output_dir}/")
            for k in sorted(best):
                print(f"  {k}  ({len(best[k]):,} chars)")
        else:
            print(f"Multi-component result ({len(best)} files) in {output_dir}/")
            for k in sorted(best):
                print(f"  {k}  ({len(best[k]):,} chars)")
    else:
        print(f"Result (first 800 chars):\n{best[:800]}")
        print(f"\nFull file: {output_dir}/best_candidate.md")
    print()


if __name__ == "__main__":
    main()
