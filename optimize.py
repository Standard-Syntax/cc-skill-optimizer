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

# Ensure WARNING level logging is enabled for the module
logging.basicConfig(level=logging.WARNING)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import llm_config  # noqa: F401 — side-effect: sets env vars for all LiteLLM calls
from evaluator import make_replay_evaluator
from llm_config import (
    DEFAULT_MODEL,
    REFLECTION_MODEL,
)
from parse_session import DEFAULT_CLAUDE_DIR, build_corpus
from section_parser import (
    load_sections_from_file,
    merge_sections,
)
from synthetic_evaluator import (
    generate_tasks_for_domain,
    load_task_library,
    make_dspy_synthetic_pipeline,
    make_synthetic_evaluator,
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def split_corpus(
    episodes: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_frac)
    v = int(n * val_frac)
    return shuffled[:t], shuffled[t : t + v], shuffled[t + v :]


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
    We concatenate all components into a single skill_md string, then score
    using the base evaluator which expects a plain string (the SKILL.md content).

    For replay mode (no live Claude Code), we just score the skill_md component.
    """

    def evaluate(candidate: dict[str, str], example) -> tuple[float, dict]:
        # Extract the primary skill text for heuristic scoring
        skill_text = candidate.get("skill_md") or candidate.get("claude_md") or ""
        if not skill_text:
            # Fall back to concatenating all components
            skill_text = "\n\n".join(f"# {k}\n{v}" for k, v in candidate.items())

        score, side_info = base_evaluator(skill_text, example)

        # Add per-component metadata to ASI so GEPA can diagnose per-file issues
        side_info["components"] = {k: len(v) for k, v in candidate.items()}
        side_info["n_components"] = len(candidate)
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

    The evaluator scores based on the root-level file (root_key) as the primary
    entry point, but includes metadata about all files in the ASI so GEPA can
    diagnose issues at any level.
    """

    def evaluate(candidate: dict[str, str], example) -> tuple[float, dict]:
        # Extract the primary file content for scoring
        primary_text = candidate.get(root_key) or ""
        if not primary_text and candidate:
            # Fall back to the shortest file (likely the most focused)
            fallback_key = min(candidate, key=lambda k: len(candidate[k]))
            primary_text = candidate[fallback_key]

        score, side_info = base_evaluator(primary_text, example)

        # Add per-file metadata to ASI so GEPA can diagnose per-file issues
        side_info["nested_files"] = {k: len(v) for k, v in candidate.items()}
        side_info["n_nested_files"] = len(candidate)
        side_info["nested_file_keys"] = list(candidate.keys())

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
) -> str | dict[str, str]:
    """
    Run gepa.optimize_anything and return the best candidate.

    seed_candidate can be:
      str           → single SKILL.md (original behaviour)
      dict[str,str] → multi-component (--target multi), sections (--target sections),
                       or nested (--target nested);
                       each key is a file path (multi/nested) or section name (sections);
                       GEPA evolves all simultaneously.
    """
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    is_multi = isinstance(seed_candidate, dict)

    base_evaluate = make_replay_evaluator(
        episodes=train_set,
        use_llm_judge=use_llm_judge,
        judge_lm=judge_lm,
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
            print(f"    {k}  ({len(seed_candidate[k]):,} chars)")
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
                max_metric_calls=max_metric_calls,
                cache_evaluation=True,
                parallel=False,
                frontier_type="instance",
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=3,
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
    seed_candidate: str,
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
    This is more powerful but requires the DSPy program abstraction.
    """
    import dspy
    from dspy.teleprompt import MIPROv2

    # Configure DSPy LMs
    task_lm_obj = dspy.LM(model=task_lm, temperature=0.7, max_tokens=4096)
    reflect_lm_obj = dspy.LM(model=reflection_lm, temperature=1.0, max_tokens=16000)
    dspy.configure(lm=task_lm_obj)

    # DSPy Signature: given task + context, produce high-quality response
    class SkillGuidedTask(dspy.Signature):
        """Apply repository skills to complete a software engineering task."""

        skill_instructions: str = dspy.InputField(desc="SKILL.md content guiding the agent")
        task_prompt: str = dspy.InputField(desc="The software engineering task to complete")
        error_context: str = dspy.InputField(
            desc="Prior errors and context from the session", default=""
        )
        completion: str = dspy.OutputField(
            desc="How the agent should approach and complete this task"
        )

    class SkillProgram(dspy.Module):
        def __init__(self, skill_content: str):
            self.skill_content = skill_content
            self.predictor = dspy.Predict(SkillGuidedTask)

        def forward(self, task_prompt: str, error_context: str = "") -> dspy.Prediction:
            return self.predictor(
                skill_instructions=self.skill_content,
                task_prompt=task_prompt,
                error_context=error_context,
            )

    # Convert episodes to DSPy Examples
    def ep_to_example(ep: dict) -> dspy.Example:
        errors = "; ".join(ep.get("error_messages", [])[:2])
        return dspy.Example(
            task_prompt=ep.get("task_prompt", ""),
            error_context=errors,
            # Gold: describe the ideal outcome based on actual session data
            completion=_ideal_completion_from_episode(ep),
        ).with_inputs("task_prompt", "error_context")

    def _ideal_completion_from_episode(ep: dict) -> str:
        """Construct a gold-standard completion hint from the real session."""
        parts = []
        outcome = ep.get("outcome", "unknown")
        if outcome == "success":
            parts.append("Successfully completed the task with minimal tool calls.")
        elif outcome == "error":
            parts.append(
                "Task encountered errors. The key issues were: "
                + "; ".join(ep.get("error_messages", ["unknown"])[:2])
            )
        cmds = ep.get("bash_commands", [])[:3]
        if cmds:
            parts.append("Key commands: " + "; ".join(cmds))
        return " ".join(parts) or "Task completed."

    dspy_train = [ep_to_example(ep) for ep in train_set if ep.get("task_prompt")]
    dspy_val = [ep_to_example(ep) for ep in val_set if ep.get("task_prompt")]

    # Metric: score based on episode quality signals
    def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        # Find the matching episode for this gold example
        for ep in train_set + val_set:
            if ep.get("task_prompt", "")[:100] == gold.task_prompt[:100]:
                from evaluator import score_episode

                score, _ = score_episode(ep)
                return score
        return 0.5

    program = SkillProgram(seed_candidate)

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
                max_metric_calls=max_metric_calls,
                cache_evaluation=True,
                frontier_type="instance",
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=3,
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
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.20)

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
    ap.add_argument("--max-evals", type=int, default=100, help="Maximum GEPA metric calls")
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
        "--use-llm-judge",
        action="store_true",
        help="Use LLM judge in session mode (ignored in --no-sessions, always uses judge)",
    )
    ap.add_argument("--judge-lm", default=DEFAULT_MODEL)

    # Output
    ap.add_argument("--output-dir", default="outputs/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Load seed — multi-target uses a dict from --skill-dir,
    # single targets use a string from --seed-file
    # ------------------------------------------------------------------ #
    skill_dir: Path | None = None
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
                max_bootstrap_evals=max(10, args.max_evals // 4),
                max_gepa_evals=args.max_evals,
                output_dir=str(output_dir),
            )
        else:
            best = run_gepa_synthetic(
                seed_candidate=seed_candidate,
                train_tasks=train_tasks,
                val_tasks=val_tasks,
                objective=objective,
                background=background,
                max_metric_calls=args.max_evals,
                reflection_lm=args.reflection_lm,
                output_dir=output_dir,
                judge_lm=args.judge_lm,
                use_judge=use_judge,
                max_depth=args.section_depth,
                is_sections=(args.target == "sections"),
                is_nested=(args.target == "nested"),
                nested_root=nested_root,
            )

    else:
        # -------- SESSION-BACKED PATH -------- #
        print(f"\n[main] Mode: SESSION-BACKED (reading from {args.claude_dir})")

        episodes = build_corpus(
            Path(args.claude_dir),
            project_filter=args.project_filter,
            min_tool_calls=args.min_tool_calls,
        )
        if not episodes:
            print(
                "[main] ERROR: No episodes found.\n"
                "  Check --claude-dir and --project-filter.\n"
                "  To run without sessions: add --no-sessions"
            )
            sys.exit(1)

        train_set, val_set, test_set = split_corpus(
            episodes, args.train_frac, args.val_frac, args.seed
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
            best = run_dspy_gepa(
                seed_candidate=seed_candidate,
                train_set=train_set,
                val_set=val_set,
                objective=objective,
                max_metric_calls=args.max_evals,
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
                max_metric_calls=args.max_evals,
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
