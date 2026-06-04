"""
Tests for Task 15.1: wire neutral_closing into _outcome_score().

Regression guard for the silent accuracy regression where parse_session.py
sets episode["neutral_closing"]=True on unknown-outcome episodes with
files_written > 0 and no errors, but the evaluator ignored the field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluator import _outcome_score


class TestOutcomeScoreNeutralClosing:
    """_outcome_score() respects neutral_closing on unknown outcomes."""

    def test_neutral_closing_true_returns_0_7(self):
        score = _outcome_score({"outcome": "unknown", "neutral_closing": True})
        assert score == 0.7, f"Expected 0.7 for unknown+neutral_closing, got {score}"

    def test_neutral_closing_false_returns_0_5(self):
        score = _outcome_score({"outcome": "unknown", "neutral_closing": False})
        assert score == 0.5, f"Expected 0.5 for unknown+neutral_closing=False, got {score}"

    def test_neutral_closing_key_missing_returns_0_5(self):
        # No neutral_closing key at all — should fall back to plain unknown mapping
        score = _outcome_score({"outcome": "unknown"})
        assert score == 0.5, f"Expected 0.5 for unknown (no neutral_closing key), got {score}"

    def test_neutral_closing_ignored_when_outcome_is_success(self):
        # neutral_closing is a SECONDARY heuristic — only meaningful when outcome is ambiguous
        score = _outcome_score({"outcome": "success", "neutral_closing": True})
        assert score == 1.0, f"Expected 1.0 (success mapping), got {score}"

    def test_neutral_closing_ignored_when_outcome_is_error(self):
        score = _outcome_score({"outcome": "error", "neutral_closing": True})
        assert score == 0.0, f"Expected 0.0 (error mapping), got {score}"

    def test_neutral_closing_ignored_when_outcome_is_interrupted(self):
        score = _outcome_score({"outcome": "interrupted", "neutral_closing": True})
        assert score == 0.3, f"Expected 0.3 (interrupted mapping), got {score}"


class TestOutcomeScoreBaseMappings:
    """_outcome_score() preserves the four explicit outcome mappings."""

    def test_success_returns_1_0(self):
        assert _outcome_score({"outcome": "success"}) == 1.0

    def test_interrupted_returns_0_3(self):
        assert _outcome_score({"outcome": "interrupted"}) == 0.3

    def test_error_returns_0_0(self):
        assert _outcome_score({"outcome": "error"}) == 0.0

    def test_unknown_returns_0_5(self):
        # Plain unknown (no neutral_closing) — pre-change behavior must be preserved
        assert _outcome_score({"outcome": "unknown"}) == 0.5

    def test_unknown_outcome_falls_back_to_0_5(self):
        # Unrecognized outcome string falls back to the unknown mapping (0.5)
        assert _outcome_score({"outcome": "mystery_state"}) == 0.5

    def test_missing_outcome_key_defaults_to_0_5(self):
        # No "outcome" key at all — should use the .get(..., "unknown") default
        assert _outcome_score({}) == 0.5


class TestOutcomeScoreIntegrationWithScoreEpisode:
    """Verify the score_episode() aggregate function still returns valid (float, dict) tuples."""

    def test_score_episode_returns_valid_tuple_for_neutral_closing_episode(self):
        from evaluator import score_episode

        ep = {
            "outcome": "unknown",
            "neutral_closing": True,
            "tool_calls": [{"tool": "read"}, {"tool": "write"}],
            "error_messages": [],
            "duration_s": 45.0,
            "token_stats": {"input": 100, "output": 50, "cache_read": 0, "cache_create": 0},
            "bash_commands": [],
            "files_written": ["foo.py"],
            "files_read": [],
            "assistant_text": ["done"],
            "task_prompt": "write a function",
            "compaction_summary": None,
            "thinking_blocks": [],
        }
        score, side_info = score_episode(ep)
        # Score must be in [0, 1]
        assert 0.0 <= score <= 1.0
        # Side info must be a dict
        assert isinstance(side_info, dict)
        # The outcome score component should be 0.7 (the new value), not 0.5
        assert side_info.get("scores", {}).get("outcome") == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
