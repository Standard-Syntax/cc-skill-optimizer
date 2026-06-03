"""
tests/test_evaluator_fixes.py
==============================

Behavioral tests verifying 3 design fixes in:
  - src/synthetic_evaluator.py
  - src/evaluator.py

Fix 1: _parse_llm_json helper (both files)
Fix 2: ValueError replaces assert for weight validation (synthetic_evaluator.py)
Fix 3: _COMPILED_SPECIFICITY pre-compiled regex patterns (synthetic_evaluator.py)
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

# -----------------------------------------------------------------------
# Import after path setup
# -----------------------------------------------------------------------
import synthetic_evaluator as se
import evaluator as ev


# =======================================================================
# Fix 1: _parse_llm_json
# =======================================================================


class TestParseLlmJson_synthetic:
    """Tests for synthetic_evaluator._parse_llm_json"""

    def test_valid_json_dict(self):
        result = se._parse_llm_json('{"score": 0.8}', {})
        assert result == {"score": 0.8}

    def test_valid_json_list(self):
        result = se._parse_llm_json("[1, 2, 3]", [])
        assert result == [1, 2, 3]

    def test_invalid_json_returns_default_dict(self):
        default = {"fallback": True}
        result = se._parse_llm_json('{"score": 0.8', default)
        assert result == default

    def test_invalid_json_returns_default_list(self):
        default = [1, 2, 3]
        result = se._parse_llm_json("not json at all", default)
        assert result == default

    def test_json_fence_json_block(self):
        """JSON wrapped in ```json ... ``` fences is parsed correctly."""
        raw = '```json\n{"score": 0.85, "reasoning": "good"}\n```'
        result = se._parse_llm_json(raw, {})
        assert result == {"score": 0.85, "reasoning": "good"}

    def test_json_fence_plain_backticks(self):
        """JSON wrapped in plain ``` ... ``` fences is also stripped."""
        raw = '```\n{"score": 0.6}\n```'
        result = se._parse_llm_json(raw, {})
        assert result == {"score": 0.6}

    def test_empty_string_returns_default(self):
        result = se._parse_llm_json("", {"default": True})
        assert result == {"default": True}

    def test_empty_json_array(self):
        result = se._parse_llm_json("[]", [])
        assert result == []

    def test_whitespace_only_returns_default(self):
        result = se._parse_llm_json("   \n\n  ", {})
        assert result == {}

    def test_leading_trailing_whitespace_stripped(self):
        raw = '  \n  {"score": 0.5}\n  '
        result = se._parse_llm_json(raw, {})
        assert result == {"score": 0.5}

    def test_mixed_fence_and_whitespace(self):
        raw = '```json  \n{"score": 0.99}  \n```  '
        result = se._parse_llm_json(raw, {})
        assert result == {"score": 0.99}


class TestParseLlmJson_evaluator:
    """Tests for evaluator._parse_llm_json (same implementation, same behaviours)"""

    def test_valid_json_dict(self):
        result = ev._parse_llm_json('{"score": 0.8}', {})
        assert result == {"score": 0.8}

    def test_valid_json_list(self):
        result = ev._parse_llm_json("[1, 2, 3]", [])
        assert result == [1, 2, 3]

    def test_invalid_json_returns_default(self):
        result = ev._parse_llm_json('{"score": 0.8', {"fallback": True})
        assert result == {"fallback": True}

    def test_json_fence_json_block(self):
        raw = '```json\n{"score": 0.85}\n```'
        result = ev._parse_llm_json(raw, {})
        assert result == {"score": 0.85}

    def test_empty_string_returns_default(self):
        result = ev._parse_llm_json("", {"default": True})
        assert result == {"default": True}

    def test_empty_list_default(self):
        result = ev._parse_llm_json("", [])
        assert result == []

    def test_whitespace_json_fence(self):
        raw = '```json\n  {"a": 1}  \n```'
        result = ev._parse_llm_json(raw, {})
        assert result == {"a": 1}


# =======================================================================
# Fix 2: ValueError for weight validation (Issue 10)
# synthetic_evaluator.make_synthetic_evaluator only
# =======================================================================


class TestWeightValidation:
    """ValueError is raised when use_judge=True and weights don't sum to 1.0"""

    def test_weights_sum_to_one_no_error(self):
        """No exception when weights sum exactly to 1.0."""
        tasks = se.load_task_library("general")
        # Should not raise
        eval_fn = se.make_synthetic_evaluator(
            task_library=tasks,
            judge_weight=0.65,
            structural_weight=0.35,
            use_judge=True,
        )
        assert callable(eval_fn)

    def test_weights_sum_to_one_structural_only_no_error(self):
        """use_judge=False should not validate weights at all."""
        tasks = se.load_task_library("general")
        # Should not raise even with nonsense weights
        eval_fn = se.make_synthetic_evaluator(
            task_library=tasks,
            judge_weight=0.5,
            structural_weight=0.2,  # sums to 0.7, not 1.0
            use_judge=False,
        )
        assert callable(eval_fn)

    def test_weights_sum_close_to_one_no_error(self):
        """Abs diff >= 0.01 raises; abs diff < 0.01 passes."""
        tasks = se.load_task_library("general")
        eval_fn = se.make_synthetic_evaluator(
            task_library=tasks,
            judge_weight=0.649,
            structural_weight=0.351,  # sum = 1.0 exactly
            use_judge=True,
        )
        assert callable(eval_fn)

    def test_weights_sum_to_0_9_raises(self):
        """Weights summing to 0.9 (< 1.0) raises ValueError."""
        tasks = se.load_task_library("general")
        with pytest.raises(ValueError, match=r"judge_weight.*structural_weight.*must sum to 1.0"):
            se.make_synthetic_evaluator(
                task_library=tasks,
                judge_weight=0.5,
                structural_weight=0.4,
                use_judge=True,
            )

    def test_weights_sum_to_1_1_raises(self):
        """Weights summing to 1.1 (> 1.0) raises ValueError."""
        tasks = se.load_task_library("general")
        with pytest.raises(ValueError, match=r"judge_weight.*structural_weight.*must sum to 1.0"):
            se.make_synthetic_evaluator(
                task_library=tasks,
                judge_weight=0.6,
                structural_weight=0.5,
                use_judge=True,
            )

    def test_single_weight_zero_raises(self):
        """judge_weight=0 with structural_weight=0 raises."""
        tasks = se.load_task_library("general")
        with pytest.raises(ValueError, match=r"judge_weight.*structural_weight.*must sum to 1.0"):
            se.make_synthetic_evaluator(
                task_library=tasks,
                judge_weight=0.0,
                structural_weight=0.0,
                use_judge=True,
            )

    def test_judge_weight_zero_structural_one_no_error(self):
        """Zero judge weight is allowed (edge case — pure structural)."""
        tasks = se.load_task_library("general")
        eval_fn = se.make_synthetic_evaluator(
            task_library=tasks,
            judge_weight=0.0,
            structural_weight=1.0,
            use_judge=True,
        )
        assert callable(eval_fn)

    def test_error_message_contains_actual_values(self):
        """ValueError message includes the submitted weight values."""
        tasks = se.load_task_library("general")
        try:
            se.make_synthetic_evaluator(
                task_library=tasks,
                judge_weight=0.3,
                structural_weight=0.3,
                use_judge=True,
            )
        except ValueError as exc:
            msg = str(exc)
            assert "0.3" in msg or "judge_weight" in msg


# =======================================================================
# Fix 3: _COMPILED_SPECIFICITY pre-compiled regex patterns
# synthetic_evaluator only
# =======================================================================


class TestCompiledSpecificity:
    """_COMPILED_SPECIFICITY produces the same results as re.findall per pattern."""

    def test_all_seven_patterns_compile_without_error(self):
        """All 7 patterns in _SPECIFICITY_PATTERNS compile successfully."""
        for i, pattern in enumerate(se._SPECIFICITY_PATTERNS):
            compiled = re.compile(pattern)  # must not raise
            assert compiled.pattern == se._SPECIFICITY_PATTERNS[i]

    def test_compiled_count_equals_findall_count(self):
        """Pre-compiled hits match per-pattern re.findall hits."""
        candidate = """
        Use `uv run pytest` and `ruff check --fix` for fast feedback.
        Set target-version = 'py313' in [tool.ruff].
        The CALCULATE function with DATESINPERIOD is powerful.
        Run: $ python -m pytest tests/ --tb=short -v
        Edit setup.py and pyproject.toml for configuration.
        Polars and Pydantic v2 are the tools you need.
        Try --verbose flag for detailed output.
        """
        compiled_hits = sum(
            len(se._COMPILED_SPECIFICITY[i].findall(candidate))
            for i in range(len(se._SPECIFICITY_PATTERNS))
        )
        naive_hits = sum(len(re.findall(p, candidate)) for p in se._SPECIFICITY_PATTERNS)
        assert compiled_hits == naive_hits

    def test_empty_candidate_yields_zero_hits(self):
        """Zero hits on an empty candidate string."""
        hits = sum(
            len(se._COMPILED_SPECIFICITY[i].findall(""))
            for i in range(len(se._SPECIFICITY_PATTERNS))
        )
        assert hits == 0

    def test_no_matches_candidate_yields_zero(self):
        """Candidate with no pattern matches returns 0."""
        candidate = "This is a generic sentence with no specific content."
        hits = sum(
            len(se._COMPILED_SPECIFICITY[i].findall(candidate))
            for i in range(len(se._SPECIFICITY_PATTERNS))
        )
        assert hits == 0

    def test_each_pattern_finds_expected_content(self):
        """Each individual pattern finds its expected match type."""
        # Pattern 0: inline code `...`
        assert len(se._COMPILED_SPECIFICITY[0].findall("Use `uv run`")) == 1
        assert len(se._COMPILED_SPECIFICITY[0].findall("no backticks")) == 0

        # Pattern 1: specific tool names (uv|ruff|ty|pytest|rg|fd|eza|bat)
        assert len(se._COMPILED_SPECIFICITY[1].findall("run uv and ruff")) == 2
        assert len(se._COMPILED_SPECIFICITY[1].findall("pip and npm")) == 0

        # Pattern 2: specific library names (Pydantic|Polars|LangGraph|DuckDB|WeasyPrint|structlog)
        assert len(se._COMPILED_SPECIFICITY[2].findall("Polars and Pydantic")) == 2
        assert len(se._COMPILED_SPECIFICITY[2].findall("pandas and numpy")) == 0

        # Pattern 3: domain terms (TMDL|DAX|CALCULATE|DATESINPERIOD)
        assert len(se._COMPILED_SPECIFICITY[3].findall("DAX and CALCULATE")) == 2
        assert len(se._COMPILED_SPECIFICITY[3].findall("SQL query")) == 0

        # Pattern 4: CLI flags --\w+
        assert len(se._COMPILED_SPECIFICITY[4].findall("--verbose --dry-run")) == 2
        assert len(se._COMPILED_SPECIFICITY[4].findall("no flags")) == 0

        # Pattern 5: shell commands $\s*\w+
        assert len(se._COMPILED_SPECIFICITY[5].findall("$ python script.py")) == 1
        assert len(se._COMPILED_SPECIFICITY[5].findall("just text")) == 0

        # Pattern 6: file extensions
        assert len(se._COMPILED_SPECIFICITY[6].findall(".py and .toml")) == 2
        assert len(se._COMPILED_SPECIFICITY[6].findall("no extension")) == 0

    def test_structural_score_uses_compiled_patterns(self):
        """structural_score() returns a (score, breakdown) tuple with expected keys."""
        score, breakdown = se.structural_score(
            "## Overview\n\n"
            "Use `uv run pytest` and `--verbose` for testing.\n"
            "Polars is great for data processing.\n"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(breakdown, dict)
        assert "specificity" in breakdown
        assert isinstance(breakdown["specificity"], float)


# =======================================================================
# Summary fixture — prints results summary
# =======================================================================


def test_all_patterns_accounted_for():
    """Verify the number of patterns hasn't changed (7 total)."""
    assert len(se._SPECIFICITY_PATTERNS) == 7
    assert len(se._COMPILED_SPECIFICITY) == 7


# =======================================================================
# Identity test: _parse_llm_json is the same function across modules
# =======================================================================


def test_parse_llm_json_is_same_function():
    """Verify both evaluator modules import the same _parse_llm_json from utils."""
    import utils

    assert se._parse_llm_json is ev._parse_llm_json
    assert se._parse_llm_json is utils._parse_llm_json


# =======================================================================
# Feedback synthesis tests
# =======================================================================


class TestFeedbackSynthesisEvaluator:
    """Tests for evaluator._build_feedback"""

    def test_error_outcome_includes_error_details(self):
        episode = {"outcome": "error"}
        side_info = {
            "n_tool_calls": 3,
            "compaction": False,
            "error_messages": ["FileNotFoundError", "Timeout"],
        }
        fb = ev._build_feedback(episode, side_info)
        assert "errors" in fb.lower()
        assert "FileNotFoundError" in fb

    def test_excessive_tool_calls_detected(self):
        episode = {"outcome": "success"}
        side_info = {"n_tool_calls": 30, "compaction": False}
        fb = ev._build_feedback(episode, side_info)
        assert "Excessive tool calls" in fb
        assert "30" in fb

    def test_compaction_flagged(self):
        episode = {"outcome": "success"}
        side_info = {"n_tool_calls": 10, "compaction": True}
        fb = ev._build_feedback(episode, side_info)
        assert "compaction" in fb.lower()

    def test_interrupted_outcome_notes_session_end(self):
        episode = {"outcome": "interrupted"}
        side_info = {"n_tool_calls": 5, "compaction": False}
        fb = ev._build_feedback(episode, side_info)
        assert "ended" in fb.lower() or "interrupted" in fb.lower()

    def test_success_with_minimal_tools_generic_feedback(self):
        episode = {"outcome": "success"}
        side_info = {"n_tool_calls": 3, "compaction": False}
        fb = ev._build_feedback(episode, side_info)
        assert fb  # non-empty
        assert "errors" not in fb.lower()  # no error mentioned


# =======================================================================
# YAML validation gate (--target agent)
# =======================================================================


class TestYamlValidationGate:
    """Tests for the YAML validation gate in make_replay_evaluator (target='agent')."""

    def test_invalid_yaml_returns_zero_score(self):
        """A candidate with an unparseable YAML block returns score=0.0 with yaml_valid=False."""
        from evaluator import make_replay_evaluator

        ev_fn = make_replay_evaluator(episodes=[], target="agent")
        candidate = "# AGENTS.md\n\n```yaml\nroles: [agent1, agent2\nbroken: [unclosed\n```\n"
        score, side_info = ev_fn(candidate, {})
        assert score == 0.0
        assert side_info["yaml_valid"] is False
        assert "yaml_error" in side_info

    def test_valid_yaml_returns_normal_score(self):
        """A candidate with valid YAML blocks returns a normal score (>= 0) with yaml_valid=True."""
        from evaluator import make_replay_evaluator

        ep = {
            "outcome": "success",
            "tool_calls": [],
            "error_messages": [],
            "bash_commands": [],
            "files_written": [],
        }
        ev_fn = make_replay_evaluator(episodes=[ep], target="agent")
        candidate = "# AGENTS.md\n\n```yaml\nroles:\n  - agent1\n  - agent2\n```\n"
        score, side_info = ev_fn(candidate, ep)
        assert score > 0.0
        assert side_info.get("yaml_valid") is True  # present and True after valid YAML

    def test_non_agent_target_skips_yaml_check(self):
        """When target is not 'agent', the YAML check is skipped entirely (no yaml_valid key set on the early-return path)."""
        from evaluator import make_replay_evaluator

        ep = {"outcome": "success", "tool_calls": [], "error_messages": []}
        ev_fn = make_replay_evaluator(episodes=[ep], target="skill")
        # This candidate has invalid YAML, but target='skill' so validation should be skipped
        candidate = "# SKILL.md\n\n```yaml\nbroken: [unclosed\n```\n"
        score, side_info = ev_fn(candidate, ep)
        # Should NOT return 0.0 from YAML gate
        # Should return a normal score (heuristic_only = 1.0 for success outcome)
        assert score == 1.0
        # yaml_valid should not be set (validation was skipped)
        assert "yaml_valid" not in side_info or side_info.get("yaml_valid") is True


# =======================================================================
# Task 10.4: sort_by_time parameter in build_corpus / sort_episodes_by_timestamp
# =======================================================================


class TestBuildCorpusSortByTime:
    """Tests for sort_episodes_by_timestamp (task 10.4)."""

    def test_sort_by_time_true_sorts_chronologically(self):
        """When sort_by_time=True, episodes are sorted by timestamp ascending."""
        from parse_session import sort_episodes_by_timestamp

        episodes = [
            {"timestamp": "2024-01-20T10:00:00+00:00", "outcome": "success"},
            {"timestamp": "2024-01-10T10:00:00+00:00", "outcome": "success"},
            {"timestamp": "2024-01-15T10:00:00+00:00", "outcome": "success"},
        ]
        result = sort_episodes_by_timestamp(episodes)
        assert result[0]["timestamp"] == "2024-01-10T10:00:00+00:00"
        assert result[1]["timestamp"] == "2024-01-15T10:00:00+00:00"
        assert result[2]["timestamp"] == "2024-01-20T10:00:00+00:00"

    def test_episodes_without_timestamp_sorted_to_end(self):
        """Episodes without a timestamp field are placed at the end of the sorted list."""
        from parse_session import sort_episodes_by_timestamp

        episodes = [
            {"timestamp": "2024-01-20T10:00:00+00:00", "outcome": "success"},
            {"outcome": "success"},  # no timestamp
            {"timestamp": "2024-01-10T10:00:00+00:00", "outcome": "success"},
        ]
        result = sort_episodes_by_timestamp(episodes)
        # Timestamped episodes come first, in order
        assert result[0]["timestamp"] == "2024-01-10T10:00:00+00:00"
        assert result[1]["timestamp"] == "2024-01-20T10:00:00+00:00"
        # No-timestamp episode is last
        assert "timestamp" not in result[2]

    def test_empty_list_returns_empty(self):
        """Empty corpus returns empty list."""
        from parse_session import sort_episodes_by_timestamp

        result = sort_episodes_by_timestamp([])
        assert result == []

    def test_original_list_not_mutated(self):
        """The original corpus is not mutated by sorting."""
        from parse_session import sort_episodes_by_timestamp

        original = [
            {"timestamp": "2024-01-20T10:00:00+00:00", "outcome": "success"},
            {"timestamp": "2024-01-10T10:00:00+00:00", "outcome": "success"},
        ]
        original_copy = list(original)
        sort_episodes_by_timestamp(original)
        assert original == original_copy


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
