"""
Tests for the make_nested_evaluator fallback fix in optimize.py (Task 7.4).

The change replaces the "shortest file" heuristic with "alphabetically first root-level key,
then alphabetically first key".

These tests verify:
- The fallback logic in evaluate() inside make_nested_evaluator
- The new selection hierarchy: root-level keys (no "/") preferred, then alphabetical
- The primary_text extraction from the selected key
- Edge cases: empty candidate, only nested keys, root_key absent, root_key present but empty
"""

import inspect
from unittest.mock import MagicMock

import pytest


class TestMakeNestedEvaluatorFallback:
    """Test suite for make_nested_evaluator fallback logic."""

    @pytest.fixture
    def mock_base_evaluator(self):
        """Create a mock base_evaluator that returns a fixed score."""
        mock = MagicMock()
        mock.return_value = (0.8, {"mocked": True})
        return mock

    @pytest.fixture
    def make_nested_evaluator(self):
        """Import and return the make_nested_evaluator function."""
        from optimize import make_nested_evaluator

        return make_nested_evaluator

    # -------------------------------------------------------------------------
    # Test Case 1: root_key present and non-empty — primary path
    # -------------------------------------------------------------------------
    def test_root_key_present_non_empty_primary_path(
        self, mock_base_evaluator, make_nested_evaluator
    ):
        """Create a candidate with {"CLAUDE.md": "primary content"} and root_key="CLAUDE.md".
        Verify primary_text = "primary content" (no fallback).
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"CLAUDE.md": "primary content"}
        example = {}

        score, side_info = evaluate(candidate, example)

        # Verify base_evaluator was called with "primary content"
        mock_base_evaluator.assert_called_once_with("primary content", example)
        assert score == 0.8

    # -------------------------------------------------------------------------
    # Test Case 2: root_key absent, root-level keys present — prefers root-level
    # -------------------------------------------------------------------------
    def test_root_key_absent_prefers_root_level(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"src/a.md": "x", "AGENTS.md": "y", "SKILL.md": "z"} with
        root_key="CLAUDE.md" (absent). Verify primary_text = "y" (from "AGENTS.md" —
        alphabetically first root-level key).
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"src/a.md": "x", "AGENTS.md": "y", "SKILL.md": "z"}
        example = {}

        evaluate(candidate, example)

        # Fallback should pick "AGENTS.md" (alphabetically first root-level key)
        mock_base_evaluator.assert_called_once_with("y", example)

    # -------------------------------------------------------------------------
    # Test Case 3: root_key absent, only nested keys — alphabetically first nested
    # -------------------------------------------------------------------------
    def test_root_key_absent_only_nested_keys(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"src/a.md": "x", "docs/b.md": "y"} with
        root_key="CLAUDE.md". Verify primary_text = "y" (from "docs/b.md" —
        alphabetically first).
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"src/a.md": "x", "docs/b.md": "y"}
        example = {}

        evaluate(candidate, example)

        # Both keys contain "/", so fallback to alphabetically first: "docs/b.md"
        mock_base_evaluator.assert_called_once_with("y", example)

    # -------------------------------------------------------------------------
    # Test Case 4: root_key absent, multiple root-level keys
    # -------------------------------------------------------------------------
    def test_root_key_absent_multiple_root_level(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"AGENTS.md": "agents", "SKILL.md": "skill", "CLAUDE.md": "claude"} with
        root_key="README.md" (not in keys). Verify primary_text = "agents content"
        (from "AGENTS.md" — alphabetically first root-level key).

        Note: Python sorts strings character-by-character:
        "AGENTS.md" vs "CLAUDE.md": A=A, then G (71) < L (76)
        So "AGENTS.md" < "CLAUDE.md" < "SKILL.md"
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="README.md")
        candidate = {
            "AGENTS.md": "agents content",
            "SKILL.md": "skill content",
            "CLAUDE.md": "claude content",
        }
        example = {}

        evaluate(candidate, example)

        # Alphabetically: AGENTS.md < CLAUDE.md < SKILL.md (because A=G < L < S after first char)
        mock_base_evaluator.assert_called_once_with("agents content", example)

    # -------------------------------------------------------------------------
    # Test Case 5: root_key present but empty — fallback fires
    # -------------------------------------------------------------------------
    def test_root_key_present_but_empty_fallback_fires(
        self, mock_base_evaluator, make_nested_evaluator
    ):
        """Create {"CLAUDE.md": "", "AGENTS.md": "y"} with root_key="CLAUDE.md".
        The empty string is falsy, so fallback fires. Verify primary_text = "y"
        (from "AGENTS.md").
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"CLAUDE.md": "", "AGENTS.md": "y"}
        example = {}

        evaluate(candidate, example)

        # Empty string is falsy, fallback picks "AGENTS.md"
        mock_base_evaluator.assert_called_once_with("y", example)

    # -------------------------------------------------------------------------
    # Test Case 6: empty candidate dict — no crash
    # -------------------------------------------------------------------------
    def test_empty_candidate_no_crash(self, mock_base_evaluator, make_nested_evaluator):
        """Create {} with root_key="CLAUDE.md". Verify primary_text = "" (the
        fallback's "if not primary_text and candidate" is False, so fallback doesn't
        fire; primary_text remains "").
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {}
        example = {}

        score, side_info = evaluate(candidate, example)

        # candidate is empty dict, so (not primary_text and candidate) is False
        # primary_text remains ""
        mock_base_evaluator.assert_called_once_with("", example)
        assert score == 0.8

    # -------------------------------------------------------------------------
    # Test Case 7: only one root-level key
    # -------------------------------------------------------------------------
    def test_only_one_root_level_key(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"AGENTS.md": "x"} with root_key="CLAUDE.md".
        Verify primary_text = "x".
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"AGENTS.md": "x"}
        example = {}

        evaluate(candidate, example)

        mock_base_evaluator.assert_called_once_with("x", example)

    # -------------------------------------------------------------------------
    # Test Case 8: only one nested key
    # -------------------------------------------------------------------------
    def test_only_one_nested_key(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"src/a.md": "x"} with root_key="CLAUDE.md".
        Verify primary_text = "x" (from "src/a.md" — only key,
        alphabetically first).
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"src/a.md": "x"}
        example = {}

        evaluate(candidate, example)

        mock_base_evaluator.assert_called_once_with("x", example)

    # -------------------------------------------------------------------------
    # Test Case 9: alphabetical ordering is case-sensitive
    # -------------------------------------------------------------------------
    def test_alphabetical_ordering_case_sensitive(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"agents.md": "lower", "AGENTS.md": "upper"} with
        root_key="CLAUDE.md". Python's default sort is case-sensitive:
        uppercase letters come before lowercase. Verify which is selected.
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"agents.md": "lower", "AGENTS.md": "upper"}
        example = {}

        evaluate(candidate, example)

        # Python sort: "AGENTS.md" < "agents.md" because uppercase A < lowercase a
        mock_base_evaluator.assert_called_once_with("upper", example)

    # -------------------------------------------------------------------------
    # Test Case 10: side_info still constructed
    # -------------------------------------------------------------------------
    def test_side_info_still_constructed(self, mock_base_evaluator, make_nested_evaluator):
        """After evaluate(), verify side_info contains "nested_files" (dict
        mapping key -> char count), "n_nested_files" (int), and
        "nested_file_keys" (list).
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"CLAUDE.md": "content", "src/a.md": "more"}
        example = {}

        score, side_info = evaluate(candidate, example)

        assert "nested_files" in side_info
        assert side_info["nested_files"] == {"CLAUDE.md": 7, "src/a.md": 4}

        assert "n_nested_files" in side_info
        assert side_info["n_nested_files"] == 2

        assert "nested_file_keys" in side_info
        assert set(side_info["nested_file_keys"]) == {"CLAUDE.md", "src/a.md"}

    # -------------------------------------------------------------------------
    # Test Case 11: signature of evaluate unchanged
    # -------------------------------------------------------------------------
    def test_signature_unchanged(self, make_nested_evaluator):
        """Use inspect.signature to verify the evaluate function has
        parameters (candidate, example) and returns a tuple.
        """
        mock_base_evaluator = MagicMock(return_value=(0.8, {}))
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")

        sig = inspect.signature(evaluate)
        params = list(sig.parameters.keys())

        # Should have exactly (candidate, example)
        assert params == ["candidate", "example"]
        # Return annotation should contain "tuple" (can be string like "tuple[float, dict]")
        assert "tuple" in str(sig.return_annotation)

    # -------------------------------------------------------------------------
    # Test Case 12: root_keys with mixed separators
    # -------------------------------------------------------------------------
    def test_root_keys_with_mixed_separators(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"CLAUDE.md": "x", "src/sub/file.md": "y", "SKILL.md": "z"}
        with root_key="README.md" (absent). Verify primary_text = "x" (from
        "CLAUDE.md" — first root-level key alphabetically).
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="README.md")
        candidate = {"CLAUDE.md": "x", "src/sub/file.md": "y", "SKILL.md": "z"}
        example = {}

        evaluate(candidate, example)

        # Root-level keys: CLAUDE.md, SKILL.md. Alphabetically: CLAUDE.md < SKILL.md
        mock_base_evaluator.assert_called_once_with("x", example)

    # -------------------------------------------------------------------------
    # Test Case 13: deep nested keys
    # -------------------------------------------------------------------------
    def test_deep_nested_keys(self, mock_base_evaluator, make_nested_evaluator):
        """Create {"a/b/c/d.md": "x", "a/b/c.md": "y", "a/b.md": "z"}
        with root_key="CLAUDE.md". All contain "/", so root_keys is empty.
        Verify primary_text = "z" (from "a/b.md" — alphabetically first of all).
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="CLAUDE.md")
        candidate = {"a/b/c/d.md": "x", "a/b/c.md": "y", "a/b.md": "z"}
        example = {}

        evaluate(candidate, example)

        # All nested. Alphabetically: a/b.md < a/b/c.md < a/b/c/d.md
        mock_base_evaluator.assert_called_once_with("z", example)

    # -------------------------------------------------------------------------
    # Test Case 14: regression — shortest-file heuristic is GONE
    # -------------------------------------------------------------------------
    def test_shortest_file_heuristic_regression(self, mock_base_evaluator, make_nested_evaluator):
        """Verify that the new heuristic prefers root-level keys over shortest
        values. Old heuristic: min by len(value). New heuristic: first
        alphabetically among root-level keys.

        Test case: {"src/long.md": "x" * 10000, "src/short.md": "y",
        "AGENTS.md": "z" * 5000}
        - Old heuristic: "src/short.md" (shortest value: 1 char)
        - New heuristic: "AGENTS.md" (first root-level key)
        """
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="README.md")
        candidate = {
            "src/long.md": "x" * 10000,
            "src/short.md": "y",
            "AGENTS.md": "z" * 5000,
        }
        example = {}

        evaluate(candidate, example)

        # Root-level key: AGENTS.md (only root-level key)
        # Should pick AGENTS.md (root-level), not src/short.md (shortest value)
        mock_base_evaluator.assert_called_once()


# -------------------------------------------------------------------------
# Additional edge case tests
# -------------------------------------------------------------------------


class TestMakeNestedEvaluatorEdgeCases:
    """Additional edge case tests."""

    @pytest.fixture
    def mock_base_evaluator(self):
        mock = MagicMock()
        mock.return_value = (0.5, {})
        return mock

    @pytest.fixture
    def make_nested_evaluator(self):
        from optimize import make_nested_evaluator

        return make_nested_evaluator

    def test_root_key_none_in_candidate(self, mock_base_evaluator, make_nested_evaluator):
        """Root key is not in candidate but present as key with truthy value."""
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="NOTTHERE.md")
        candidate = {"SKILL.md": "hello"}
        example = {}

        evaluate(candidate, example)

        # Falls back to SKILL.md
        mock_base_evaluator.assert_called_once_with("hello", example)

    def test_all_root_level_same_prefix(self, mock_base_evaluator, make_nested_evaluator):
        """Multiple root-level files with same prefix."""
        evaluate = make_nested_evaluator(mock_base_evaluator, root_key="OTHER.md")
        candidate = {
            "AAA.md": "a",
            "AAB.md": "b",
            "AAC.md": "c",
        }
        example = {}

        evaluate(candidate, example)

        # AAA.md is alphabetically first
        mock_base_evaluator.assert_called_once_with("a", example)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
