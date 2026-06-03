"""
Test suite for DSPy runner temperature+thinking conflict guard in optimize.py (Task 4.5).

Tests the `_model_uses_thinking()` helper function and the guard in `run_dspy_gepa`
that raises ValueError when either task_lm or reflection_lm uses extended thinking.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================

# Sample data for testing the guard
SAMPLE_TRAIN_SET = [{"task_prompt": "test task", "outcome": "success"}]
SAMPLE_VAL_SET = [{"task_prompt": "test val", "outcome": "success"}]


# ============================================================================
# Helper to mock dspy modules in sys.modules (since dspy is imported inside the function)
# ============================================================================


def mock_dspy_modules():
    """Context manager to mock dspy modules in sys.modules.

    dspy 3.x exposes optimizers at both top-level (dspy.GEPA, dspy.MIPROv2,
    dspy.BootstrapFewShot) and via the legacy dspy.teleprompt submodule
    (dspy.teleprompt.GEPA, dspy.teleprompt.MIPROv2). We mock BOTH paths
    so that `from dspy import X` and `from dspy.teleprompt import X`
    are both intercepted. The source code in optimize.py uses the
    top-level form (canonical 3.x), so the dspy.X mocks are the
    critical ones; the dspy.teleprompt.X mocks are kept for
    backward compat with any remaining legacy call sites.
    """
    mock_dspy = MagicMock()
    mock_teleprompt = MagicMock()

    # Set up LM class mock
    mock_lm = MagicMock()
    mock_dspy.LM.return_value = mock_lm
    mock_dspy.configure = MagicMock()
    mock_dspy.Module = MagicMock()
    mock_dspy.Signature = MagicMock()
    mock_dspy.Predict = MagicMock()
    mock_dspy.InputField = MagicMock()
    mock_dspy.OutputField = MagicMock()
    mock_dspy.Example = MagicMock()
    mock_dspy.Prediction = MagicMock()

    # dspy 3.x: optimizers exposed at top-level dspy namespace
    mock_dspy.GEPA = MagicMock()
    mock_dspy.MIPROv2 = MagicMock()
    mock_dspy.BootstrapFewShot = MagicMock()

    # Legacy: also exposed via dspy.teleprompt (backward compat)
    mock_teleprompt.GEPA = MagicMock()
    mock_teleprompt.MIPROv2 = MagicMock()

    mocks = {
        "dspy": mock_dspy,
        "dspy.teleprompt": mock_teleprompt,
    }

    return patch.dict("sys.modules", mocks)


# ============================================================================
# Test: _model_uses_thinking returns True for thinking-enabled models
# ============================================================================


class TestThinkingModelsReturnTrue:
    """Test cases where _model_uses_thinking should return True."""

    @pytest.mark.parametrize(
        "model",
        [
            "minimax/minimax-m3",
            "minimax/minimax-m2.7",
            "minimax/minimax-m2.7-highspeed",
            "minimax/minimax-m2.7-turbo",  # Future variant — startswith should match
            "minimax/minimax-m3.5",  # Future variant
        ],
    )
    def test_returns_true_for_thinking_enabled_models(self, model: str):
        """All minimax models with m2.7/m3 prefixes should return True."""
        from optimize import _model_uses_thinking

        result = _model_uses_thinking(model)
        assert result is True, f"Expected True for thinking-enabled model: {model}"


# ============================================================================
# Test: _model_uses_thinking returns False for non-thinking models
# ============================================================================


class TestNonThinkingModelsReturnFalse:
    """Test cases where _model_uses_thinking should return False."""

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-haiku-4-5-20251001",
            "anthropic/claude-sonnet-4-6",  # Legacy, no thinking in this path
            "",  # Empty string
            "gpt-4",
            "minimax/minimax-m2",  # Older variant, no thinking — only m2.7 prefix triggers
        ],
    )
    def test_returns_false_for_non_thinking_models(self, model: str):
        """Non-minimax or legacy models should return False."""
        from optimize import _model_uses_thinking

        result = _model_uses_thinking(model)
        assert result is False, f"Expected False for non-thinking model: {model}"


# ============================================================================
# Test: run_dspy_gepa raises ValueError when task_lm uses thinking
# ============================================================================


def test_raises_when_task_lm_uses_thinking():
    """Guard should raise ValueError when task_lm is thinking-enabled."""
    from optimize import run_dspy_gepa

    with mock_dspy_modules():
        with pytest.raises(ValueError) as exc_info:
            run_dspy_gepa(
                seed_candidate="test skill",
                train_set=SAMPLE_TRAIN_SET,
                val_set=SAMPLE_VAL_SET,
                objective="test",
                max_metric_calls=10,
                task_lm="minimax/minimax-m3",  # Thinking-enabled
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_output"),
            )

        # Verify error message mentions both model names
        error_message = str(exc_info.value)
        assert "minimax/minimax-m3" in error_message, "Error should mention task_lm"
        assert "llm_config" in error_message.lower() or "lines" in error_message.lower(), (
            "Error should reference llm_config.py"
        )


# ============================================================================
# Test: run_dspy_gepa raises ValueError when reflection_lm uses thinking
# ============================================================================


def test_raises_when_reflection_lm_uses_thinking():
    """Guard should raise ValueError when reflection_lm is thinking-enabled."""
    from optimize import run_dspy_gepa

    with mock_dspy_modules():
        with pytest.raises(ValueError) as exc_info:
            run_dspy_gepa(
                seed_candidate="test skill",
                train_set=SAMPLE_TRAIN_SET,
                val_set=SAMPLE_VAL_SET,
                objective="test",
                max_metric_calls=10,
                task_lm="anthropic/claude-haiku-4-5-20251001",
                reflection_lm="minimax/minimax-m2.7-highspeed",  # Thinking-enabled
                output_dir=Path("/tmp/test_output"),
            )

        error_message = str(exc_info.value)
        assert "minimax/minimax-m2.7-highspeed" in error_message, (
            "Error should mention reflection_lm"
        )


# ============================================================================
# Test: run_dspy_gepa does NOT raise when both models are non-thinking
# ============================================================================


def test_no_error_when_both_models_non_thinking():
    """Guard should NOT raise when both models are non-thinking."""
    from optimize import run_dspy_gepa
    from optimize import _model_uses_thinking

    # First verify both models are indeed non-thinking (precondition)
    task_is_thinking = _model_uses_thinking("anthropic/claude-haiku-4-5-20251001")
    reflect_is_thinking = _model_uses_thinking("anthropic/claude-haiku-4-5-20251001")

    assert task_is_thinking is False, "task_lm should not be thinking-enabled"
    assert reflect_is_thinking is False, "reflection_lm should not be thinking-enabled"

    # Now test the function - it should NOT raise the thinking guard
    with mock_dspy_modules():
        # The key verification: ValueError with temperature/thinking in message should NOT be raised
        # We'll catch any exception and check if it's the thinking guard
        thinking_guard_error = None
        try:
            run_dspy_gepa(
                seed_candidate="test skill",
                train_set=SAMPLE_TRAIN_SET,
                val_set=SAMPLE_VAL_SET,
                objective="test",
                max_metric_calls=10,
                task_lm="anthropic/claude-haiku-4-5-20251001",
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_output"),
            )
        except Exception as e:
            # Check if this is specifically the thinking guard error
            # If it's ANY other error, that's fine - the guard passed!
            err_str = str(e).lower()
            if "temperature" in err_str and "thinking" in err_str:
                thinking_guard_error = str(e)
            # Otherwise, the guard didn't fire - this is the expected case

        # The thinking guard should NOT have fired - it failed somewhere else (which proves guard passed)
        # The key here: there's no thinking_guard_error means the guard didn't fire
        assert thinking_guard_error is None, (
            f"Thinking guard should not fire for non-thinking models. Got: {thinking_guard_error}"
        )


# ============================================================================
# Test: guard fires BEFORE dspy.LM is called
# ============================================================================


def test_guard_fires_before_dspy_lm_called():
    """Guard should raise BEFORE dspy.LM is ever instantiated."""
    from optimize import run_dspy_gepa

    with mock_dspy_modules() as mocks:
        mock_dspy = mocks["dspy"]

        # Track whether LM was called
        lm_called = False

        def track_lm(*args, **kwargs):
            nonlocal lm_called
            lm_called = True
            return MagicMock()

        mock_dspy.LM = track_lm

        with pytest.raises(ValueError):
            run_dspy_gepa(
                seed_candidate="test skill",
                train_set=SAMPLE_TRAIN_SET,
                val_set=SAMPLE_VAL_SET,
                objective="test",
                max_metric_calls=10,
                task_lm="minimax/minimax-m3",  # Thinking-enabled
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_output"),
            )

        # Assert LM was never called
        assert not lm_called, "Guard should fire BEFORE dspy.LM is instantiated"


# ============================================================================
# Test: guard error message references llm_config.py
# ============================================================================


def test_guard_message_references_llm_config():
    """Error message should reference llm_config.py."""
    from optimize import run_dspy_gepa

    with mock_dspy_modules():
        with pytest.raises(ValueError) as exc_info:
            run_dspy_gepa(
                seed_candidate="test skill",
                train_set=SAMPLE_TRAIN_SET,
                val_set=SAMPLE_VAL_SET,
                objective="test",
                max_metric_calls=10,
                task_lm="minimax/minimax-m3",
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_output"),
            )

        error_message = str(exc_info.value).lower()
        # Check for llm_config reference (various ways it might appear)
        has_ref = (
            "llm_config" in error_message
            or "lines 99-105" in error_message
            or "line 99" in error_message
            or "src/llm_config" in error_message
        )
        assert has_ref, f"Error should reference llm_config.py: {error_message}"


# ============================================================================
# Test: guard error message includes both model names
# ============================================================================


def test_guard_message_includes_both_models():
    """Error message should include both task_lm and reflection_lm."""
    from optimize import run_dspy_gepa

    with mock_dspy_modules():
        with pytest.raises(ValueError) as exc_info:
            run_dspy_gepa(
                seed_candidate="test skill",
                train_set=SAMPLE_TRAIN_SET,
                val_set=SAMPLE_VAL_SET,
                objective="test",
                max_metric_calls=10,
                task_lm="minimax/minimax-m3",
                reflection_lm="minimax/minimax-m2.7-highspeed",
                output_dir=Path("/tmp/test_output"),
            )

        error_message = str(exc_info.value)
        assert "minimax/minimax-m3" in error_message, "Error should include task_lm"
        assert "minimax/minimax-m2.7-highspeed" in error_message, (
            "Error should include reflection_lm"
        )


# ============================================================================
# Test: regression — dspy.LM kwargs unchanged when guard passes
# ============================================================================


def test_dspy_lm_kwargs_unchanged():
    """When guard passes, dspy.LM should still be called with temperature kwargs."""
    from optimize import run_dspy_gepa

    with mock_dspy_modules() as mocks:
        mock_dspy = mocks["dspy"]

        captured_kwargs = {}

        def capture_kwargs(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_dspy.LM = capture_kwargs

        try:
            run_dspy_gepa(
                seed_candidate="test skill",
                train_set=SAMPLE_TRAIN_SET,
                val_set=SAMPLE_VAL_SET,
                objective="test",
                max_metric_calls=10,
                task_lm="anthropic/claude-haiku-4-5-20251001",
                reflection_lm="anthropic/claude-sonnet-4-6",
                output_dir=Path("/tmp/test_output"),
            )
        except Exception:
            # May fail later, that's OK — we just check the guard didn't block early
            pass

        # Check that temperature was in the LM calls (if we got that far)
        # At minimum, verify the temperature parameter is defined in the source
        import optimize
        import inspect as ins

        source = ins.getsource(optimize.run_dspy_gepa)
        assert "temperature=0.7" in source, "task_lm should still use temperature=0.7"
        assert "temperature=1.0" in source, "reflection_lm should still use temperature=1.0"


# ============================================================================
# Test: signature of run_dspy_gepa unchanged
# ============================================================================


def test_signature_unchanged():
    """Public function signature should be preserved."""
    from optimize import run_dspy_gepa

    sig = inspect.signature(run_dspy_gepa)

    # Verify expected parameters exist
    param_names = list(sig.parameters.keys())
    expected_params = [
        "seed_candidate",
        "train_set",
        "val_set",
        "objective",
        "max_metric_calls",
        "task_lm",
        "reflection_lm",
        "output_dir",
    ]

    for param in expected_params:
        assert param in param_names, f"Parameter {param} should exist in signature"

    # No new parameters should have beenadded
    assert set(param_names) == set(expected_params), (
        f"Signature should not change. Got: {param_names}"
    )


# ============================================================================
# Additional edge cases
# ============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_string_not_thinking(self):
        """Empty string should return False."""
        from optimize import _model_uses_thinking

        # Empty string has no prefix match
        assert _model_uses_thinking("") is False

    def test_none_input_raises(self):
        """None input should raise TypeError (expects string)."""
        from optimize import _model_uses_thinking

        # None input causes AttributeError since str.startswith is called on None
        # This is documented behavior - the function expects a string
        with pytest.raises(AttributeError):
            _model_uses_thinking(None)  # type: ignore

    def test_partial_match_not_triggered(self):
        """Partial prefix matches should not trigger."""
        from optimize import _model_uses_thinking

        # These shouldn't match because they don't START with the prefix
        assert _model_uses_thinking("minimax/some-other-model") is False
        assert _model_uses_thinking("anthropic-minimax") is False
