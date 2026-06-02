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
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

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
    DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
    INFERENCE_PARAMS = {}
    EXTRA_BODY = {}
    EVAL_MAX_TOKENS = 512
    THINKING_CONFIG_EVAL = {"type": "disabled"}
    DIRECT_OUTPUT_PREFIX = "Respond directly and concisely. Output only what is requested."

# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def _parse_llm_json(raw: str, default: dict | list) -> dict | list:
    """Strip markdown code fences and parse LLM JSON response."""
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _outcome_score(episode: dict) -> float:
    """Convert episode outcome to a base score [0, 1]."""
    mapping = {
        "success": 1.0,
        "unknown": 0.5,
        "interrupted": 0.3,
        "error": 0.0,
    }
    return mapping.get(episode.get("outcome", "unknown"), 0.5)


def _efficiency_bonus(episode: dict) -> float:
    """
    Reward fewer tool calls and shorter duration.
    Returns a delta in [-0.15, +0.15].
    """
    bonus = 0.0
    n_tools = len(episode.get("tool_calls", []))
    if n_tools <= 5:
        bonus += 0.08
    elif n_tools <= 12:
        bonus += 0.04
    elif n_tools >= 40:
        bonus -= 0.10
    elif n_tools >= 25:
        bonus -= 0.05

    dur = episode.get("duration_s")
    if dur is not None:
        if dur < 60:
            bonus += 0.07
        elif dur < 180:
            bonus += 0.03
        elif dur > 600:
            bonus -= 0.08

    return max(-0.15, min(0.15, bonus))


def _cache_bonus(episode: dict) -> float:
    """Reward high cache hit ratio (SKILL.md keeps tokens in prompt-cache)."""
    tok = episode.get("token_stats", {})
    cache_read = tok.get("cache_read", 0)
    total_input = tok.get("input", 0) + cache_read
    if total_input == 0:
        return 0.0
    ratio = cache_read / total_input
    return min(0.05, ratio * 0.10)


def score_episode(episode: dict) -> tuple[float, dict]:
    """
    Score a single episode without an LLM judge.
    Returns (score ∈ [0,1], side_info dict).
    """
    base = _outcome_score(episode)
    eff = _efficiency_bonus(episode)
    cache = _cache_bonus(episode)
    score = max(0.0, min(1.0, base + eff + cache))

    side_info = {
        "score": score,
        "outcome": episode.get("outcome"),
        "n_tool_calls": len(episode.get("tool_calls", [])),
        "n_errors": len(episode.get("error_messages", [])),
        "duration_s": episode.get("duration_s"),
        "error_messages": episode.get("error_messages", [])[:3],
        "bash_commands": episode.get("bash_commands", [])[:8],
        "files_written": episode.get("files_written", [])[:6],
        "compaction": episode.get("compaction_summary") is not None,
        "token_stats": episode.get("token_stats", {}),
        "task_prompt": episode.get("task_prompt", "")[:200],
        "final_assistant_msg": (episode.get("assistant_text") or [""])[-1][:400],
    }
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

    user_msg = (
        f"SKILL.md:\n{candidate_skill[:3000]}\n\n"
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


def make_replay_evaluator(
    episodes: list[dict],
    use_llm_judge: bool = False,
    judge_lm: str = "anthropic/claude-haiku-4-5-20251001",
    judge_weight: float = 0.4,
):
    """
    Returns a GEPA-compatible evaluate(candidate, example) function.

    candidate: str  — the skill/CLAUDE.md content being optimized
    example:   dict — one episode from the corpus (from dataset= arg to optimize_anything)
    """

    def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
        # Heuristic score (fast, free)
        heuristic_score, side_info = score_episode(example)

        if use_llm_judge:
            judge_score, reasoning = llm_judge_score(candidate, example, judge_lm)
            final_score = (1 - judge_weight) * heuristic_score + judge_weight * judge_score
            side_info["judge_score"] = judge_score
            side_info["judge_reasoning"] = reasoning
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
    """
    from parse_session import parse_session

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
            base_score, _ = score_episode(episode)
        else:
            base_score = 0.5 if returncode == 0 else 0.1

        eff = _efficiency_bonus(episode) if episode else 0.0
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
