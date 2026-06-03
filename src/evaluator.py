"""
evaluator.py
============
GEPA-compatible evaluator for Claude Code skill / CLAUDE.md optimization.

Two modes:

  1. REPLAY mode  (default, no Claude Code installation needed)
     Feed previously-parsed episodes and score a candidate skill against
     them using an LLM judge.  Fast & cheap — ideal for GEPA's iterative
     reflection loop.

  2. LIVE mode  (requires Claude Code + the target repo)
     Write the candidate skill to disk, invoke `claude --print` on a task,
     parse the resulting session JSONL, and score it.  Expensive but ground-
     truth accurate.

GEPA picks up the returned (score, side_info) tuple; side_info drives ASI.

Anthropic endpoint + thinking configuration
-------------------------------------------
Judge calls use claude-haiku-4-5 with thinking DISABLED (fast, cheap, JSON).
Thinking blocks are NOT needed here — judge calls are single-turn with no
tool use.  See llm_config.py for the full research notes on when thinking
blocks must be passed back (tool use loops only).

temperature and top_k are INCOMPATIBLE with Anthropic extended thinking —
they are intentionally omitted from INFERENCE_PARAMS.

judge_weight default
--------------------
Both make_replay_evaluator and make_synthetic_evaluator default judge_weight to
0.65 (65% LLM judge / 35% heuristic).  The LLM judge carries the primary
semantic signal; the heuristic is compressed into the [0.35, 0.85] band so it
never dominates the blend.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

# PyYAML >= 6.0 for YAML validation gate (--target agent)
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False

# Import llm_config constants (also sets env vars as side-effect)
try:
    from llm_config import (
        DEFAULT_MODEL,
        DIRECT_OUTPUT_PREFIX,
        EVAL_MAX_TOKENS,
        EXTRA_BODY,
        INFERENCE_PARAMS,
        THINKING_CONFIG_EVAL,
    )
except ImportError:
    # Fallback when running outside the src/ path (e.g. direct test invocation)
    DEFAULT_MODEL = "minimax/minimax-m2.7-highspeed"
    INFERENCE_PARAMS = {}
    EXTRA_BODY = {}
    EVAL_MAX_TOKENS = 512
    THINKING_CONFIG_EVAL = {"type": "disabled"}
    DIRECT_OUTPUT_PREFIX = "Respond directly and concisely. Output only what is requested."

# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

from utils import _parse_llm_json

# Default thresholds for _efficiency_bonus (configurable via make_replay_evaluator)
DEFAULT_TOOL_CALL_THRESHOLDS = {
    "low_max": 5,
    "medium_max": 12,
    "high_min": 25,
    "extreme_min": 40,
    "low_bonus": 0.08,
    "medium_bonus": 0.04,
    "high_penalty": 0.05,
    "extreme_penalty": 0.10,
}


def _build_feedback(episode: dict, side_info: dict) -> str:
    """Synthesize a 1-3 sentence feedback string for the GEPA reflection LM."""
    parts: list[str] = []
    outcome = episode.get("outcome", "unknown")
    errs = side_info.get("error_messages", [])
    n_tools = side_info.get("n_tool_calls", 0)
    n_errors = side_info.get("n_errors", 0)
    compaction = side_info.get("compaction", False)

    if outcome == "error" and errs:
        parts.append(f"Agent hit errors: {'; '.join(errs[:2])}")
    if n_tools >= 25:
        parts.append(
            f"Excessive tool calls ({n_tools}); skill lacks shortcuts or pointer for direct path."
        )
    if compaction:
        parts.append(
            "Context compaction hit during the session — skill may be too verbose or omit key pointers."
        )
    if outcome == "interrupted":
        parts.append(
            "Session ended before agent completed; review whether the skill guided the agent to a clear stopping point."
        )

    # Bash command sequence (first 12 commands)
    bash_cmds = episode.get("bash_commands") or []
    if bash_cmds:
        parts.append(f"Commands run: {'; '.join(bash_cmds[:12])}")

    # Files touched (first 8 files)
    files = episode.get("files_written") or []
    if files:
        parts.append(f"Files touched: {', '.join(files[:8])}")

    # Distinct error count (when n_errors > 1)
    if n_errors > 1:
        parts.append(f"{n_errors} distinct errors encountered.")

    if not parts:
        parts.append(f"Outcome: {outcome}. No specific failure detected.")
    return " ".join(parts)


def _outcome_score(episode: dict) -> float:
    """Convert episode outcome to a base score [0, 1]."""
    mapping = {
        "success": 1.0,
        "unknown": 0.5,
        "interrupted": 0.3,
        "error": 0.0,
    }
    return mapping.get(episode.get("outcome", "unknown"), 0.5)


def _efficiency_bonus(episode: dict, thresholds: dict | None = None) -> float:
    """
    Reward fewer tool calls and shorter duration. Thresholds are configurable.
    Returns a delta in [-0.15, +0.15].
    """
    t = thresholds or DEFAULT_TOOL_CALL_THRESHOLDS
    bonus = 0.0
    n_tools = len(episode.get("tool_calls", []))
    if n_tools <= t["low_max"]:
        bonus += t["low_bonus"]
    elif n_tools <= t["medium_max"]:
        bonus += t["medium_bonus"]
    elif n_tools >= t["extreme_min"]:
        bonus -= t["extreme_penalty"]
    elif n_tools >= t["high_min"]:
        bonus -= t["high_penalty"]

    dur = episode.get("duration_s")
    if dur is not None:
        if dur < 60:
            bonus += 0.07
        elif dur < 180:
            bonus += 0.03
        elif dur > 600:
            bonus -= 0.08

    return max(-0.15, min(0.15, bonus))


def _compute_cache_ratio(episode: dict) -> float:
    """Return raw cache hit ratio (cache_read / total_input), clamped to [0.0, 1.0]."""
    tok = episode.get("token_stats", {})
    cache_read = tok.get("cache_read", 0)
    total_input = tok.get("input", 0) + cache_read
    if total_input == 0:
        return 0.0
    return max(0.0, min(1.0, cache_read / total_input))


def score_episode(episode: dict, thresholds: dict | None = None) -> tuple[float, dict]:
    """
    Score a single episode without an LLM judge.
    Returns (score ∈ [0,1], side_info dict).
    """
    base = _outcome_score(episode)
    eff = _efficiency_bonus(episode, thresholds)
    # cache_ratio is exposed in side_info only, NOT in the score formula.
    # Cache hit ratio reflects session input structure, not skill quality —
    # a skill identical to the previous iteration inherits its predecessor's
    # cache state and would receive a bonus regardless of quality.
    # The reflection LM can use cache_ratio qualitatively as a verbosity signal.
    score = max(0.0, min(1.0, base + eff))

    # scores dict for multi-objective Pareto tracking (gepa 0.1.1 reads side_info["scores"])
    scores = {
        "outcome": base,
        "efficiency": (eff + 0.15) / 0.30,
        "cache_efficiency": _compute_cache_ratio(episode),
        "low_error_rate": 1.0 if not episode.get("error_messages") else 0.0,
    }

    # Truncation limits increased from prior values to give the reflection LM
    # more context per the gskill SI design pattern. The gskill paper reports
    # the reflection LM benefits most from full tool_calls and complete test
    # output, not just the first few. These larger limits are appropriate for
    # the small corpora (20-50 episodes) the project targets.
    side_info = {
        "score": score,
        "outcome": episode.get("outcome"),
        "n_tool_calls": len(episode.get("tool_calls", [])),
        "n_errors": len(episode.get("error_messages", [])),
        "duration_s": episode.get("duration_s"),
        "error_messages": episode.get("error_messages", [])[:10],
        "bash_commands": episode.get("bash_commands", [])[:20],
        "files_written": episode.get("files_written", [])[:20],
        "compaction": episode.get("compaction_summary") is not None,
        "token_stats": episode.get("token_stats", {}),
        "task_prompt": episode.get("task_prompt", "")[:200],
        "final_assistant_msg": (episode.get("assistant_text") or [""])[-1][:2000],
        "cache_ratio": _compute_cache_ratio(episode),
        "scores": scores,
    }
    side_info["feedback"] = _build_feedback(episode, side_info)
    return score, side_info


# ---------------------------------------------------------------------------
# LLM judge for replay mode
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    f"{DIRECT_OUTPUT_PREFIX}\n\n"
    "You are a Claude Code skill evaluator. You will receive:\n"
    "- A candidate SKILL.md\n"
    "- A session episode (tool calls, errors, outcome)\n\n"
    "Score 0.0–1.0: how much would this skill have helped the agent?\n"
    "Consider: error prevention, tool efficiency, context window usage.\n\n"
    "Output ONLY valid JSON on one line — no preamble, no markdown:\n"
    '{"score": <float>, "reasoning": "<one sentence>"}'
)


def llm_judge_score(
    candidate_skill: str,
    episode: dict,
    judge_lm: str = DEFAULT_MODEL,
) -> tuple[float, str]:
    """
    Call an LLM to score how helpful the candidate_skill would have been
    for this episode.  Returns (score, reasoning).

    Uses Haiku with thinking disabled — single-turn JSON scoring call,
    no tool use, no multi-turn history.  Thinking blocks not needed here.
    temperature and top_k intentionally omitted (incompatible with thinking).
    """
    import litellm  # type: ignore

    from parse_session import episode_to_asi

    asi = episode_to_asi(episode)

    # Truncate to 8000 chars to expose the full ~2K-token SKILL.md to the judge.
    # Future work: implement two-pass scoring (split candidate at midpoint, judge
    # each half, average the scores) to handle skills > 8000 chars without bias.
    user_msg = (
        f"SKILL.md:\n{candidate_skill[:8000]}\n\n"
        f"Episode:\n{asi[:2000]}\n\n"
        'Output: {"score": <0-1>, "reasoning": "<one sentence>"}'
    )

    try:
        resp = litellm.completion(
            model=judge_lm,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=EVAL_MAX_TOKENS,
            **INFERENCE_PARAMS,
        )
        raw = resp.choices[0].message.content or "{}"
        data = _parse_llm_json(raw, {})
        return float(data.get("score", 0.5)), data.get("reasoning", "")
    except Exception as exc:
        return 0.5, f"judge_error: {exc}"


# ---------------------------------------------------------------------------
# Replay evaluator factory
# ---------------------------------------------------------------------------


def _check_yaml_validity(candidate: str) -> tuple[bool, str | None]:
    """
    Check that any YAML code blocks in the candidate are parseable.
    Returns (is_valid, error_message). When is_valid is False, error_message is set.
    Markdown without YAML blocks returns (True, None) — no YAML means no validation needed.
    """
    if not _YAML_AVAILABLE:
        return True, None  # Skip validation if PyYAML is not installed
    # Match ```yaml ... ``` and ```yml ... ``` blocks
    pattern = re.compile(r"^```(?:yaml|yml)\s*\n(.*?)\n```", re.MULTILINE | re.DOTALL)
    blocks = pattern.findall(candidate)
    if not blocks:
        return True, None  # No YAML blocks; valid by default
    for i, block in enumerate(blocks):
        try:
            yaml.safe_load(block)
        except yaml.YAMLError as e:
            return False, f"YAML block {i+1} failed to parse: {e}"
    return True, None


def make_replay_evaluator(
    episodes: list[dict],
    use_llm_judge: bool = False,
    judge_lm: str = "anthropic/claude-haiku-4-5-20251001",
    judge_weight: float = 0.65,  # LLM judge is the primary semantic signal; heuristic score is compressed in [0.35, 0.85]
    tool_call_thresholds: dict | None = None,
    target: str | None = None,
):
    """
    Returns a GEPA-compatible evaluate(candidate, example) function.

    candidate: str  — the skill/CLAUDE.md content being optimized
    example:   dict — one episode from the corpus (from dataset= arg to optimize_anything)

    side_info: dict with keys 'score', 'outcome', 'scores' (multi-objective Pareto dict for
        gepa 0.1.1), and other diagnostic fields. The 'scores' key contains 4 normalized [0,1]
        values: 'outcome', 'efficiency', 'cache_efficiency', 'low_error_rate'. Required for
        frontier_type='hybrid' or 'objective' (gepa 0.1.1 reads side_info['scores']).
    """
    thresholds = tool_call_thresholds or DEFAULT_TOOL_CALL_THRESHOLDS

    def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
        # YAML validation gate (--target agent only)
        _yaml_valid = None
        _yaml_error = None
        if target == "agent":
            yaml_valid, yaml_error = _check_yaml_validity(candidate)
            _yaml_valid = yaml_valid
            _yaml_error = yaml_error
            if not yaml_valid:
                return 0.0, {
                    "yaml_valid": False,
                    "yaml_error": yaml_error,
                    "score": 0.0,
                    "feedback": f"YAML validation failed: {yaml_error}",
                }

        # Heuristic score (fast, free)
        heuristic_score, side_info = score_episode(example, thresholds)

        # Mark YAML as valid (gate passed for agent target)
        if target == "agent":
            side_info["yaml_valid"] = True

        if use_llm_judge:
            judge_score, reasoning = llm_judge_score(candidate, example, judge_lm)
            final_score = (1 - judge_weight) * heuristic_score + judge_weight * judge_score
            side_info["judge_score"] = judge_score
            side_info["judge_reasoning"] = reasoning
            # Enrich feedback with judge reasoning
            judge_reasoning = side_info.get("judge_reasoning", "")
            if judge_reasoning:
                existing = side_info.get("feedback", "")
                side_info["feedback"] = (
                    f"{existing} Judge: {judge_reasoning[:200]}"
                    if existing
                    else f"Judge: {judge_reasoning[:200]}"
                )
        else:
            final_score = heuristic_score

        side_info["candidate_length"] = len(candidate)
        side_info["candidate_preview"] = candidate[:300]

        return final_score, side_info

    return evaluate


# ---------------------------------------------------------------------------
# Live evaluator (requires Claude Code CLI)
# ---------------------------------------------------------------------------


def make_live_evaluator(
    repo_path: Path,
    skill_slot_path: Path,
    timeout: int = 300,
    test_commands: list[str] | None = None,
    tool_call_thresholds: dict | None = None,
):
    """
    Write candidate skill to disk, run claude on a task, parse the resulting
    session JSONL, and score it.

    Args:
        repo_path:       Working directory for claude invocations.
        skill_slot_path: Where to write the candidate (e.g. .claude/skills/myrepo/SKILL.md).
        timeout:         Max seconds per claude invocation.
        test_commands:   Shell commands to run after claude for pass/fail signal.
                         e.g. ["pytest tests/ -q --tb=no"]
        tool_call_thresholds: Optional dict of thresholds for _efficiency_bonus.
    """
    from parse_session import parse_session

    thresholds = tool_call_thresholds or DEFAULT_TOOL_CALL_THRESHOLDS

    def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
        task_prompt = example.get("task_prompt", "")
        if not task_prompt:
            return 0.0, {"error": "no task_prompt in example"}

        # Write candidate skill
        skill_slot_path.parent.mkdir(parents=True, exist_ok=True)
        skill_slot_path.write_text(candidate, encoding="utf-8")

        # Find latest session file before invocation
        projects_dir = Path.home() / ".claude" / "projects"
        pre_sessions: set[Path] = (
            set(projects_dir.rglob("*.jsonl")) if projects_dir.exists() else set()
        )

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                ["claude", "--print", "--dangerously-skip-permissions", task_prompt],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            return 0.0, {"error": "timeout", "timeout_s": timeout}
        except FileNotFoundError:
            return 0.0, {
                "error": "claude CLI not found — install with: npm i -g @anthropic-ai/claude-code"
            }

        duration = time.monotonic() - t0

        # Find new session file
        post_sessions = set(projects_dir.rglob("*.jsonl")) if projects_dir.exists() else set()
        new_sessions = post_sessions - pre_sessions
        episode: dict = {}
        if new_sessions:
            newest = max(new_sessions, key=lambda p: p.stat().st_mtime)
            episode = parse_session(newest)

        # Run test commands for ground-truth pass/fail
        test_pass: bool | None = None
        test_output = ""
        if test_commands:
            try:
                tres = subprocess.run(
                    test_commands[0],
                    shell=True,
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                test_pass = tres.returncode == 0
                test_output = (tres.stdout + tres.stderr)[:500]
            except Exception as e:
                test_output = str(e)

        # Score
        if test_pass is True:
            base_score = 1.0
        elif test_pass is False:
            base_score = 0.0
        elif episode:
            base_score, _ = score_episode(episode, thresholds)
        else:
            base_score = 0.5 if returncode == 0 else 0.1

        eff = _efficiency_bonus(episode, thresholds) if episode else 0.0
        score = max(0.0, min(1.0, base_score + eff))

        side_info = {
            "score": score,
            "returncode": returncode,
            "duration_s": duration,
            "stdout": stdout[:500],
            "stderr": stderr[:300],
            "test_pass": test_pass,
            "test_output": test_output,
            "n_tool_calls": len(episode.get("tool_calls", [])) if episode else None,
            "outcome": episode.get("outcome") if episode else "no_session",
            "error_messages": episode.get("error_messages", [])[:3] if episode else [],
        }
        return score, side_info

    return evaluate
