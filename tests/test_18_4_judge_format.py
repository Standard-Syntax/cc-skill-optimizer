"""
Tests for Task 18.4: enrich _JUDGE_SYSTEM with SKILL.md format constraints.

After Phase 18.4, the LLM judge in src/evaluator.py scores skills with a
format-rubric: well-formed skills are under 2000 tokens, use markdown
headers and numbered lists, and contain repo-specific commands. This
aligns the judge with the structural scorer (Phase 5) which already
rewards sections, lists, and penalizes generic phrases.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path so evaluator.py's `from utils import` resolves correctly
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))


class TestJudgeSystemFormatRubric:
    """_JUDGE_SYSTEM contains the SKILL.md format-rubric sentence."""

    def test_format_rubric_sentence_is_present(self):
        """The 'Format quality also matters' sentence must appear in _JUDGE_SYSTEM."""
        from src.evaluator import _JUDGE_SYSTEM
        assert "Format quality also matters" in _JUDGE_SYSTEM, (
            "Expected 'Format quality also matters' sentence in _JUDGE_SYSTEM"
        )

    def test_rubric_mentions_2000_token_target(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert "under 2000 tokens" in _JUDGE_SYSTEM, (
            "Expected rubric to mention the 2000-token SKILL.md target"
        )

    def test_rubric_mentions_markdown_headers(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert "markdown headers" in _JUDGE_SYSTEM, (
            "Expected rubric to require markdown headers"
        )

    def test_rubric_mentions_numbered_lists(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert "numbered lists" in _JUDGE_SYSTEM, (
            "Expected rubric to require numbered lists"
        )

    def test_rubric_mentions_repo_specific_commands(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert "repo-specific commands" in _JUDGE_SYSTEM, (
            "Expected rubric to require repo-specific commands"
        )

    def test_rubric_penalises_generic_skills(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert "generic advice" in _JUDGE_SYSTEM, (
            "Expected rubric to penalize generic advice"
        )

    def test_rubric_penalises_verbose_skills(self):
        from src.evaluator import _JUDGE_SYSTEM
        # Either "overly long" or "vague" should be present
        assert "overly long" in _JUDGE_SYSTEM or "vague" in _JUDGE_SYSTEM, (
            "Expected rubric to penalize verbose/vague skills"
        )

    def test_judge_system_still_ends_with_json_contract(self):
        """The JSON output contract must remain at the end of _JUDGE_SYSTEM."""
        from src.evaluator import _JUDGE_SYSTEM
        # The JSON contract line is the LAST line of the constant
        assert _JUDGE_SYSTEM.rstrip().endswith(
            '{"score": <float>, "reasoning": "<one sentence>"}'
        ), (
            "JSON output contract should be the last line of _JUDGE_SYSTEM"
        )


class TestJudgeSystemPreservesExistingCriteria:
    """The existing 'Consider: error prevention, tool efficiency, context window usage.'
    criteria are still present (the new format-rubric is ADDITIVE, not a replacement)."""

    def test_existing_consider_line_still_present(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert (
            "Consider: error prevention, tool efficiency, context window usage."
            in _JUDGE_SYSTEM
        ), (
            "Existing 'Consider:' criteria should be preserved"
        )

    def test_existing_role_description_still_present(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert "You are a Claude Code skill evaluator" in _JUDGE_SYSTEM, (
            "Role description should be preserved"
        )

    def test_existing_input_description_still_present(self):
        from src.evaluator import _JUDGE_SYSTEM
        assert "- A candidate SKILL.md" in _JUDGE_SYSTEM, (
            "Input description should be preserved"
        )


class TestM27JudgeNotModified:
    """The M2.7-specific judge (_JUDGE_SYSTEM_M2_7) in src/synthetic_evaluator.py
    is NOT changed by Phase 18.4 (it already has its own format guidance)."""

    def test_m27_judge_system_unchanged(self):
        """Verify the M2.7 judge in synthetic_evaluator.py still has its M2.7-specific
        rubric (mentions 'multiple approaches', 'error recovery strategies', etc.) and
        is NOT the same as _JUDGE_SYSTEM in src/evaluator.py."""
        from src.evaluator import _JUDGE_SYSTEM
        try:
            from src.synthetic_evaluator import _JUDGE_SYSTEM_M2_7
            # They should be different constants
            assert _JUDGE_SYSTEM != _JUDGE_SYSTEM_M2_7, (
                "_JUDGE_SYSTEM and _JUDGE_SYSTEM_M2_7 should be different"
            )
            # The M2.7 version should still mention M2.7-specific guidance
            assert (
                "multiple approaches" in _JUDGE_SYSTEM_M2_7
                or "error recovery" in _JUDGE_SYSTEM_M2_7
            ), (
                "_JUDGE_SYSTEM_M2_7 should still have its M2.7-specific rubric"
            )
        except ImportError:
            # If the M2.7 constant is not importable in this env, that's OK
            import pytest
            pytest.skip("_JUDGE_SYSTEM_M2_7 not importable in this test env")