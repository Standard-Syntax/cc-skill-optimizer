"""
Test for cache_bonus removal in src/evaluator.py (Task 5.1)

Verifies:
- _cache_bonus is removed (raises ImportError)
- _compute_cache_ratio helper added
- cache_ratio surfaced in side_info
- Score formula no longer includes cache bonus
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Ensure src/ is on the path
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

import pytest

# Import after path setup
import evaluator as ev


class TestCacheBonusRemoved:
    """Test that _cache_bonus no longer exists."""

    def test_cache_bonus_not_importable(self):
        """Attempt import of _cache_bonus should raise ImportError."""
        with pytest.raises(ImportError):
            from evaluator import _cache_bonus  # noqa: F401


class TestComputeCacheRatio:
    """Test the new _compute_cache_ratio helper function."""

    def test_empty_token_stats_returns_zero(self):
        """Episode with empty token_stats {} returns 0.0."""
        episode = {"token_stats": {}}
        result = ev._compute_cache_ratio(episode)
        assert result == 0.0

    def test_missing_token_stats_returns_zero(self):
        """Episode without token_stats key returns 0.0."""
        episode = {}
        result = ev._compute_cache_ratio(episode)
        assert result == 0.0

    def test_typical_episode_returns_correct_ratio(self):
        """cache_read=30, input=10 → ratio = 30/(30+10) = 0.75."""
        episode = {"token_stats": {"cache_read": 30, "input": 10}}
        result = ev._compute_cache_ratio(episode)
        assert result == 0.75

    def test_zero_total_input_returns_zero(self):
        """cache_read=0, input=0 → ratio = 0.0 (avoids ZeroDivisionError)."""
        episode = {"token_stats": {"cache_read": 0, "input": 0}}
        result = ev._compute_cache_ratio(episode)
        assert result == 0.0

    def test_all_cached_returns_one(self):
        """cache_read=100, input=0 → ratio = 1.0 (clamped)."""
        episode = {"token_stats": {"cache_read": 100, "input": 0}}
        result = ev._compute_cache_ratio(episode)
        assert result == 1.0

    def test_clamped_to_zero_for_negative_input(self):
        """Negative input should clamp to 0."""
        # Even with negative, result should be clamped to [0, 1]
        episode = {"token_stats": {"cache_read": 50, "input": -10}}
        result = ev._compute_cache_ratio(episode)
        # 50 / (50 + -10) = 50/40 = 1.25, clamped to 1.0
        assert result == 1.0


class TestScoreEpisodeNoCacheInScore:
    """Test that score_episode formula doesn't include cache."""

    def test_high_cache_does_not_boost_score(self):
        """High cache ratio + good outcome should equal outcome + efficiency only."""
        # Episode: high cache (100, 0), success outcome, no tool calls (best efficiency)
        episode = {
            "outcome": "success",
            "tool_calls": [],  # ≤5 means max efficiency bonus
            "duration_s": 30,  # <60s means +0.07 duration bonus
            "token_stats": {"cache_read": 100, "input": 0},
        }

        # Compute expected: outcome + efficiency (no cache in formula!)
        expected_outcome = ev._outcome_score(episode)
        expected_eff = ev._efficiency_bonus(episode, None)
        expected_score = expected_outcome + expected_eff
        expected_score = max(0.0, min(1.0, expected_score))

        # Get actual from score_episode
        actual_score, side_info = ev.score_episode(episode)

        # Verify score equals outcome + efficiency (NOT including cache bonus)
        # Max possible: 1.0 (success) + 0.15 (efficiency) = 1.15, clamped to 1.0
        # But with duration < 60: bonus = 0.07, so 1.0 + 0.15 + 0.07 = 1.22 → clamped to 1.0
        assert actual_score == expected_score

    def test_cache_ratio_is_exposed_in_side_info(self):
        """score_episode should include cache_ratio in side_info."""
        episode = {
            "outcome": "success",
            "tool_calls": [],
            "token_stats": {"cache_read": 30, "input": 10},
        }

        _, side_info = ev.score_episode(episode)
        expected_ratio = ev._compute_cache_ratio(episode)

        assert "cache_ratio" in side_info
        assert side_info["cache_ratio"] == expected_ratio


class TestSideInfoPreservesKeys:
    """Test that side_info retains all existing keys."""

    def test_all_expected_keys_present(self):
        """Verify all keys: score, outcome, n_tool_calls, n_errors, duration_s, etc."""
        episode = {
            "outcome": "success",
            "tool_calls": [{"name": "read", "params": {}}],
            "error_messages": ["some error"],
            "duration_s": 100,
            "bash_commands": ["ls"],
            "files_written": ["a.txt"],
            "compaction_summary": True,
            "token_stats": {},
            "task_prompt": "test task",
            "assistant_text": ["response"],
        }

        _, side_info = ev.score_episode(episode)

        expected_keys = [
            "score",
            "outcome",
            "n_tool_calls",
            "n_errors",
            "duration_s",
            "error_messages",
            "bash_commands",
            "files_written",
            "compaction",
            "token_stats",
            "task_prompt",
            "final_assistant_msg",
            "cache_ratio",
            "feedback",
        ]

        for key in expected_keys:
            assert key in side_info, f"Missing key: {key}"


class TestScoreRangeRegression:
    """Test that scores remain bounded to [0, 1]."""

    def test_scores_clamped_to_zero_one_range(self):
        """Scores should be clamped to [0.0, 1.0]."""
        # Worst case: error outcome, extreme tool calls, long duration
        worst_episode = {
            "outcome": "error",
            "tool_calls": [{"name": "x", "params": {}}] * 50,  # >40 tool calls
            "duration_s": 700,  # >600s
        }

        score, _ = ev.score_episode(worst_episode)
        assert 0.0 <= score <= 1.0

        # Best case: success, minimal tool calls, fast
        best_episode = {
            "outcome": "success",
            "tool_calls": [],
            "duration_s": 30,
        }

        score, _ = ev.score_episode(best_episode)
        assert 0.0 <= score <= 1.0


class TestMakeReplayEvaluator:
    """Test that make_replay_evaluator factory still works."""

    def test_factory_constructs_without_error(self):
        """make_replay_evaluator should construct without referencing _cache_bonus."""
        episodes = [{"outcome": "success", "tool_calls": []}]

        # Should not raise - factory should not reference _cache_bonus
        evaluator = ev.make_replay_evaluator(episodes, use_llm_judge=False)

        # Should be able to call evaluate
        score, side_info = evaluator("test skill content", episodes[0])

        assert isinstance(score, float)
        assert isinstance(side_info, dict)
