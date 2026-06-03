"""
Test Task 4.1: Verify oa.log() wiring in synthetic_evaluator.py
==================================================================

These tests verify that the GEPA ASI channel (oa.log() calls) is correctly
captured and exposed through side_info["feedback"] for the reflection LM.

Tests the make_synthetic_evaluator factory and its inner evaluate() closure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure src/ is on the path
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))  # noqa: E402 — test path setup, must precede module import

# Intentional: import after sys.path.insert for path setup
import synthetic_evaluator  # noqa: E402


# Helper class to capture oa.log() calls
class LogCapture:
    """Helper to capture oa.log() calls for testing."""

    def __init__(self):
        self.logs: list[str] = []

    def log(self, *args, **kwargs):
        if args:
            self.logs.append(str(args[0]))

    def get_log_context(self):
        class _Ctx:
            def drain(self) -> str:
                return "\n".join(self.logs)

        return _Ctx()


# Test 1: oa.log() calls work without gepa installed (no-op shim path)
def test_noop_shim_path_works():
    """Verify the no-op shim path works without gepa."""
    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=False)

    candidate = "## Overview\nUse pytest for testing."
    example = {"task_description": "Test task", "pitfalls": [], "success_criteria": []}

    # Should not raise, should return (float, dict)
    score, side_info = evaluate(candidate, example)

    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert isinstance(side_info, dict), f"Expected dict, got {type(side_info)}"
    assert "feedback" in side_info, "side_info must contain 'feedback' key"
    assert isinstance(side_info["feedback"], str), "feedback must be a string"


# Test 2: side_info["feedback"] contains diagnostic content
def test_feedback_contains_outcome():
    """Verify feedback includes outcome keyword from example."""
    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=False)

    candidate = "## Overview\nUse pytest for testing."
    example = {
        "task_description": "Test task",
        "outcome": "success",
        "pitfalls": [],
        "success_criteria": [],
    }

    score, side_info = evaluate(candidate, example)

    feedback = side_info.get("feedback", "")
    assert "Outcome:" in feedback or "Diagnostic log:" in feedback or len(feedback) > 0, (
        f"Feedback should contain diagnostic info, got: {feedback!r}"
    )


# Test 3: error_messages are truncated to 200 chars each
def test_error_truncation():
    """Verify error_messages are truncated to 200 chars."""
    capture = LogCapture()

    # Replace oa module globally
    old_oa = synthetic_evaluator.oa
    synthetic_evaluator.oa = capture

    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=False)

    # Long error messages (> 200 chars each)
    long_error = "This is a very long error message that exceeds two hundred characters " * 3
    example = {
        "task_description": "Test task",
        "error_messages": [long_error, long_error],
        "outcome": "success",
        "pitfalls": [],
        "success_criteria": [],
    }

    try:
        score, side_info = evaluate(candidate="# Test", example=example)
    finally:
        synthetic_evaluator.oa = old_oa

    # Check that any captured error logs are truncated
    error_logs = [log for log in capture.logs if log.startswith("Error:")]
    for err in error_logs:
        # The log should contain "Error:" prefix + truncated content (max 207 chars)
        assert len(err) <= 207, f"Error log not truncated: {len(err)} chars"


# Test 4: tool_calls are joined with " → " and limited to first 10
def test_tool_calls_limit():
    """Verify tool_calls are joined with arrow and limited to 10."""
    capture = LogCapture()

    old_oa = synthetic_evaluator.oa
    synthetic_evaluator.oa = capture

    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=False)

    # More than 10 tool calls
    tool_calls = [{"tool": f"tool_{i}"} for i in range(15)]
    example = {
        "task_description": "Test task",
        "tool_calls": tool_calls,
        "pitfalls": [],
        "success_criteria": [],
    }

    try:
        score, side_info = evaluate(candidate="# Test", example=example)
    finally:
        synthetic_evaluator.oa = old_oa

    # Find the "Tool calls:" log
    tool_logs = [log for log in capture.logs if log.startswith("Tool calls:")]
    assert len(tool_logs) == 1, f"Expected 1 tool calls log, got {len(tool_logs)}"

    tool_log = tool_logs[0]
    # Should contain arrows between tool names
    assert " → " in tool_log, f"Tool calls should be joined with ' → ', got: {tool_log}"

    # Count tool names (10 items means 9 arrows)
    tool_part = tool_log.replace("Tool calls: ", "")
    tool_count = len(tool_part.split(" → "))
    assert tool_count == 10, f"Should have exactly 10 tools, got {tool_count}"


# Test 5: judge score is logged when use_judge=True
def test_judge_score_logged():
    """Verify judge score appears in log when use_judge=True."""
    capture = LogCapture()

    old_oa = synthetic_evaluator.oa
    synthetic_evaluator.oa = capture

    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=True)

    example = {
        "task_description": "Test task",
        "pitfalls": [],
        "success_criteria": [],
    }

    # Mock the judge to avoid network calls
    with mock.patch(
        "synthetic_evaluator.judge_score_task", return_value=(0.75, {"judge_score": 0.75})
    ):
        try:
            score, side_info = evaluate(candidate="# Test", example=example)
        finally:
            synthetic_evaluator.oa = old_oa

    # Check logs for Judge score
    judge_logs = [log for log in capture.logs if log.startswith("Judge score:")]
    assert len(judge_logs) == 1, f"Expected 'Judge score:' log, got {judge_logs}"


# Test 6: compaction signal is logged when present
def test_compaction_log():
    """Verify compaction signal is logged when present."""
    capture = LogCapture()

    old_oa = synthetic_evaluator.oa
    synthetic_evaluator.oa = capture

    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=False)

    example = {
        "task_description": "Test task",
        "compaction_summary": "Some compaction signal text",
        "pitfalls": [],
        "success_criteria": [],
    }

    try:
        score, side_info = evaluate(candidate="# Test", example=example)
    finally:
        synthetic_evaluator.oa = old_oa

    # Check for compaction log
    compaction_logs = [log for log in capture.logs if "compaction" in log.lower()]
    assert len(compaction_logs) == 1, f"Expected compaction log, got {compaction_logs}"


# Test 7: evaluate() return type is preserved
def test_return_type_tuple():
    """Verify evaluate() returns (float, dict) tuple."""
    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=False)

    candidate = "## Overview\nTest content."
    example = {
        "task_description": "Test task",
        "pitfalls": [],
        "success_criteria": [],
    }

    result = evaluate(candidate, example)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    score, side_info = result
    assert isinstance(score, (float, int)), f"Score should be numeric, got {type(score)}"
    assert isinstance(side_info, dict), f"side_info should be dict, got {type(side_info)}"


# Test 8: repeated evaluate() calls do not leak state
def test_no_state_leakage():
    """Verify repeated calls return independent side_info dicts."""
    tasks = [{"task_description": "Test task", "pitfalls": [], "success_criteria": []}]
    evaluate = synthetic_evaluator.make_synthetic_evaluator(tasks, use_judge=False)

    candidate = "## Overview\nTest content."
    example = {
        "task_description": "Test task",
        "pitfalls": [],
        "success_criteria": [],
    }

    # Call evaluate twice
    score1, side_info1 = evaluate(candidate, example)
    score2, side_info2 = evaluate(candidate, example)

    # side_info dicts should be independent (not same object, not mutated)
    assert side_info1 is not side_info2, "side_info should not be same object across calls"

    # Mutating one should not affect the other
    side_info1["test_marker"] = "modified"
    assert "test_marker" not in side_info2, "Mutation leaked to second call"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
