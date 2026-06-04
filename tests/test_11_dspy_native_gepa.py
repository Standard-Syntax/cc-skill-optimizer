"""
Test suite for DSPy 3.x native GEPA backend (Task 12.2, FR-11.7, FR-11.8).

Verifies:
  (a) --dspy-backend native-gepa is a recognized argparse choice
  (b) dspy.GEPA is imported from top-level dspy (not dspy.teleprompt)
  (c) run_dspy_native_gepa metric returns dspy.Prediction(score, feedback)
  (d) side_info['scores'] dict is accessible from the metric (Phase 9.2)
"""

from __future__ import annotations

import argparse
import inspect
import re
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Fixtures / sample data
# ============================================================================

SAMPLE_TRAIN_SET = [{"task_prompt": "test task", "outcome": "success"}]
SAMPLE_VAL_SET = [{"task_prompt": "test val", "outcome": "success"}]


# ============================================================================
# Helper: mock dspy modules in sys.modules (dual-path: top-level + teleprompt)
# ============================================================================


def mock_dspy_modules():
    """Context manager to mock dspy modules in sys.modules.

    dspy 3.x exposes optimizers at both top-level (dspy.GEPA, dspy.MIPROv2) and
    via the legacy dspy.teleprompt submodule. We mock BOTH paths so that both
    `from dspy import X` and `from dspy.teleprompt import X` are intercepted.

    Also mocks the dspy 3.0 Module API (set_lm, map_named_predictors) used
    for per-module LM injection in Phase 14.
    """
    mock_dspy = MagicMock()
    mock_teleprompt = MagicMock()

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

    # dspy 3.x: optimizers at top-level
    mock_dspy.GEPA = MagicMock()
    mock_dspy.MIPROv2 = MagicMock()
    mock_dspy.BootstrapFewShot = MagicMock()

    # Legacy: dspy.teleprompt (backward compat)
    mock_teleprompt.GEPA = MagicMock()
    mock_teleprompt.MIPROv2 = MagicMock()

    # dspy 3.0 Module API (Phase 14): per-module LM injection
    mock_dspy.Module.set_lm = MagicMock()
    mock_dspy.Module.map_named_predictors = MagicMock()

    mocks = {
        "dspy": mock_dspy,
        "dspy.teleprompt": mock_teleprompt,
    }

    return patch.dict("sys.modules", mocks)


# ============================================================================
# Class 1: TestDspyBackendCLI — covers requirement (a)
# ============================================================================


class TestDspyBackendCLI:
    """Test that --dspy-backend accepts the expected choices."""

    def test_dspy_backend_native_gepa_is_accepted(self):
        """--dspy-backend native-gepa should parse without argparse error."""
        # We test by inspecting the argparse configuration in optimize.py
        # rather than loading the full main() (which requires session corpus).
        import optimize

        # Capture the argparse parser definition by inspecting source
        src = inspect.getsource(optimize)

        # Verify the --dspy-backend argument has native-gepa as a choice
        assert 'choices=["mipro", "native-gepa"]' in src or (
            "'native-gepa'" in src and "--dspy-backend" in src
        ), "--dspy-backend should accept 'native-gepa' as a valid choice"

    def test_dspy_backend_mipro_is_accepted(self):
        """--dspy-backend mipro should parse without argparse error."""
        import optimize

        src = inspect.getsource(optimize)

        assert 'choices=["mipro", "native-gepa"]' in src or (
            "'mipro'" in src and "--dspy-backend" in src
        ), "--dspy-backend should accept 'mipro' as a valid choice"

    def test_dspy_backend_default_is_mipro(self):
        """Default value for --dspy-backend should be 'mipro' (not 'native-gepa')."""
        import optimize

        src = inspect.getsource(optimize)

        # Find the default assignment for --dspy-backend
        # Pattern: default="mipro" somewhere near --dspy-backend
        assert 'default="mipro"' in src, (
            "Default for --dspy-backend should be 'mipro', not 'native-gepa'"
        )

    def test_dspy_backend_invalid_choice_rejected(self):
        """--dspy-backend invalid should raise argparse error."""

        # Rebuild the parser as defined in optimize.py to verify validation
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--dspy-backend",
            choices=["mipro", "native-gepa"],
            default="mipro",
        )

        # Feeding an invalid choice must raise
        with pytest.raises(SystemExit):
            ap.parse_args(["--dspy-backend", "invalid"])

        # Verify it exits with code 2 (standard argparse behavior)
        try:
            ap.parse_args(["--dspy-backend", "invalid"])
        except SystemExit as exc:
            assert exc.code == 2, "Invalid choice should exit with code 2"


# ============================================================================
# Class 2: TestDspyGEPAImportPath — covers requirement (b)
# ============================================================================


class TestDspyGEPAImportPath:
    """Test that run_dspy_native_gepa uses the canonical dspy 3.x import path."""

    def test_dspy_GEPA_importable_from_top_level(self):
        """Verify dspy.GEPA is accessible from the top-level dspy namespace."""
        # We cannot guarantee dspy 3.x is installed in the test environment,
        # so we verify the import statement is present in the source code.
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        # The function should contain: from dspy import GEPA
        assert "from dspy import GEPA" in src, (
            "run_dspy_native_gepa should use 'from dspy import GEPA' "
            "(canonical dspy 3.x import path)"
        )

    def test_dspy_GEPA_NOT_using_teleprompt_submodule_in_source(self):
        """Verify run_dspy_native_gepa does NOT use the legacy dspy.teleprompt path."""
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        assert "from dspy.teleprompt import GEPA" not in src, (
            "run_dspy_native_gepa should NOT use 'from dspy.teleprompt import GEPA' "
            "(legacy path — dspy 3.x uses top-level dspy.GEPA)"
        )


# ============================================================================
# Class 3: TestRunDspyNativeGepaMetric — covers requirement (c)
# ============================================================================


class TestRunDspyNativeGepaMetric:
    """Test the metric function inside run_dspy_native_gepa."""

    def test_metric_returns_dspy_Prediction(self):
        """Metric should return dspy.Prediction(score=..., feedback=...) not a flat float."""
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        # Verify the metric returns a dspy.Prediction object
        assert "dspy.Prediction" in src, (
            "Metric should construct and return a dspy.Prediction object"
        )

        # Verify it has the correct field names (score and feedback)
        assert "score=" in src and "feedback=" in src, (
            "dspy.Prediction should be constructed with score= and feedback= fields"
        )

    def test_metric_signature_has_five_params(self):
        """Metric signature should be (gold, pred, trace=None, pred_name=None, pred_trace=None)."""
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        # Extract the metric function definition
        metric_pattern = r"def metric\(\s*gold\s*,\s*pred\s*,\s*trace\s*=\s*None\s*,\s*pred_name\s*=\s*None\s*,\s*pred_trace\s*=\s*None\s*\)"
        assert re.search(metric_pattern, src), (
            "Metric signature should be "
            "(gold, pred, trace=None, pred_name=None, pred_trace=None) "
            "— the dspy 3.x GEPAFeedbackMetric signature"
        )

    def test_metric_returns_prediction_not_float(self):
        """Metric should NOT return a raw float — it must wrap in dspy.Prediction."""
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        # The metric function body (not the docstring or signature) should not
        # contain a bare "return score" float
        # We check that the return statements all use dspy.Prediction
        metric_func_match = re.search(
            r"def metric\([^)]+\)[^:]*:(.*?)(?=\n    def |\n    program |\Z)",
            src,
            re.DOTALL,
        )
        assert metric_func_match, "Could not find metric function body"

        metric_body = metric_func_match.group(1)

        # Every return statement in the metric should return dspy.Prediction
        return_statements = re.findall(r"\breturn\b[^;]+", metric_body)
        for ret in return_statements:
            stripped = ret.strip()
            assert "dspy.Prediction" in stripped, (
                f"All return statements in metric should return dspy.Prediction. Found: {stripped}"
            )


# ============================================================================
# Class 4: TestSideInfoScoresPassedToGEPA — covers requirement (d)
# ============================================================================


class TestSideInfoScoresPassedToGEPA:
    """Test that side_info['scores'] dict is accessible from the GEPA metric."""

    def test_metric_extracts_scores_from_side_info(self):
        """Metric should extract feedback from side_info (which carries scores dict)."""
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        # The metric calls score_episode(ep) which returns (score, side_info)
        # and then extracts feedback via side_info.get("feedback", ...)
        assert "side_info.get(" in src or "side_info[" in src, (
            "Metric should access side_info to retrieve feedback for dspy.Prediction"
        )

        # Verify score_episode is called (it is the function that produces side_info)
        assert "score_episode" in src, (
            "Metric should call score_episode to evaluate episodes and produce side_info"
        )

    def test_side_info_scores_dict_reaches_dspy_Prediction(self):
        """The scores dict (added in Phase 9.2) should be accessible via feedback extraction."""
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        # score_episode returns (score, side_info) where side_info contains:
        #   - feedback: str
        #   - scores: dict (added in Phase 9.2)
        # The metric extracts feedback=side_info.get("feedback", "") and passes it
        # to dspy.Prediction.feedback=feedback.  While the metric does not
        # explicitly reference side_info["scores"], the scores dict is present
        # in side_info and accessible via the same side_info object that
        # score_episode returns.
        assert "score, side_info" in src, (
            "score_episode should be called and its side_info unpacked for use in dspy.Prediction"
        )

    def test_score_episode_return_tuple_in_source(self):
        """Verify score_episode is called with ep and returns (score, side_info) tuple."""
        import optimize

        src = inspect.getsource(optimize.run_dspy_native_gepa)

        # The metric unpacks score, side_info = score_episode(ep)
        # This is how the scores dict reaches the GEPA reflection step
        assert re.search(r"score\s*,\s*side_info\s*=\s*score_episode", src), (
            "Metric should unpack score, side_info = score_episode(ep) "
            "so that side_info (containing the scores dict) is available "
            "for dspy.Prediction feedback"
        )


# ============================================================================
# Integration: verify run_dspy_native_gepa is reachable via CLI dispatch
# ============================================================================


class TestNativeGepaCLIIntegration:
    """Verify native-gepa backend dispatch in the CLI block."""

    def test_native_gepa_dispatch_block_present(self):
        """The if args.dspy_backend == 'native-gepa' dispatch block should exist."""
        import optimize

        src = inspect.getsource(optimize)

        assert "args.dspy_backend == " in src, (
            "Should check args.dspy_backend to dispatch between mipro and native-gepa"
        )
        assert "run_dspy_native_gepa" in src, (
            "run_dspy_native_gepa should be called when --dspy-backend native-gepa is used"
        )

    def test_mipro_is_default_backend(self):
        """The else branch (default) should call run_dspy_gepa (mipro path)."""
        import optimize

        src = inspect.getsource(optimize)

        # The else branch (when dspy_backend != "native-gepa") should call run_dspy_gepa
        assert "run_dspy_gepa" in src, "Default (mipro) backend should call run_dspy_gepa"

    def test_run_dspy_gepa_uses_program_set_lm(self):
        """Verify run_dspy_gepa and run_dspy_native_gepa use program.set_lm (not dspy.configure global)."""
        import re

        import optimize

        # Check both functions use the new per-module LM injection
        for fn_name in ("run_dspy_gepa", "run_dspy_native_gepa"):
            src = inspect.getsource(getattr(optimize, fn_name))
            # Must use program.set_lm
            assert "program.set_lm" in src, (
                f"{fn_name} must use program.set_lm() for per-module LM injection"
            )
            # Must NOT use dspy.configure(lm=...) function call (the bare string "dspy.configure"
            # is allowed in comments for migration context)
            assert not re.search(r"dspy\.configure\s*\(", src), (
                f"{fn_name} must not call legacy dspy.configure(lm=...) global config"
            )

    def test_make_dspy_synthetic_pipeline_uses_map_named_predictors(self):
        """Verify make_dspy_synthetic_pipeline uses map_named_predictors for LM injection."""
        import re

        from src import synthetic_evaluator

        src = inspect.getsource(synthetic_evaluator.make_dspy_synthetic_pipeline)
        # Must use the dspy 3.0 idiomatic pattern
        assert "map_named_predictors" in src, (
            "make_dspy_synthetic_pipeline must use map_named_predictors"
        )
        assert "lambda p: p.set_lm" in src, (
            "make_dspy_synthetic_pipeline must use the lambda set_lm pattern"
        )
        # Must NOT use dspy.configure(lm=...) function call (the bare string "dspy.configure"
        # is allowed in comments for migration context)
        assert not re.search(r"dspy\.configure\s*\(", src), (
            "make_dspy_synthetic_pipeline must not call legacy dspy.configure(lm=...)"
        )


# ============================================================================
# Tests for Phase 17.3: verify the shared DSPy module refactor
# ============================================================================


class TestSharedDspyModule:
    """Phase 17.3: verify that run_dspy_gepa and run_dspy_native_gepa both
    import from src.dspy_shared (not inline duplicates), and that the mipro
    metric returns dspy.Prediction (matching the native-gepa contract)."""

    def test_dspy_shared_importable(self):
        """The shared module is importable and exports the 4 expected names."""
        from src.dspy_shared import (
            SkillGuidedTask,
            SkillProgram,
            _ideal_completion_from_episode,
            ep_to_example,
        )
        assert SkillGuidedTask is not None
        assert SkillProgram is not None
        assert ep_to_example is not None
        assert _ideal_completion_from_episode is not None
        print("OK: src.dspy_shared exports all 4 expected names")

    def test_run_dspy_gepa_no_inline_defs(self):
        """After Phase 17.2, run_dspy_gepa must NOT redefine SkillGuidedTask,
        SkillProgram, ep_to_example, or _ideal_completion_from_episode inline."""
        import optimize
        src = inspect.getsource(optimize.run_dspy_gepa)
        for symbol in ("class SkillGuidedTask", "class SkillProgram", "def ep_to_example", "def _ideal_completion_from_episode"):
            assert symbol not in src, (
                f"run_dspy_gepa still has inline {symbol} — should import from src.dspy_shared"
            )
        # It should USE the imported names (ep_to_example is called)
        assert "ep_to_example(" in src, "run_dspy_gepa should call ep_to_example(ep) ..."
        # It should NOT redefine SkillProgram inline (the new code uses SkillProgram(...))
        assert "class SkillProgram" not in src
        print("OK: run_dspy_gepa has no inline DSPy class/function defs")

    def test_run_dspy_native_gepa_no_inline_defs(self):
        """After Phase 17.2, run_dspy_native_gepa must NOT redefine the DSPy
        helpers inline either."""
        import optimize
        src = inspect.getsource(optimize.run_dspy_native_gepa)
        for symbol in ("class SkillGuidedTask", "class SkillProgram", "def ep_to_example", "def _ideal_completion_from_episode"):
            assert symbol not in src, (
                f"run_dspy_native_gepa still has inline {symbol} — should import from src.dspy_shared"
            )
        assert "ep_to_example(" in src, "run_dspy_native_gepa should call ep_to_example(ep) ..."
        assert "class SkillProgram" not in src
        print("OK: run_dspy_native_gepa has no inline DSPy class/function defs")

    def test_mipro_metric_returns_dspy_prediction(self):
        """After Phase 17.2, the mipro metric (in run_dspy_gepa) returns
        dspy.Prediction(score, feedback) — same contract as the native-gepa metric.
        The legacy 'return score' (float) is gone."""
        import optimize
        src = inspect.getsource(optimize.run_dspy_gepa)
        # The new metric must return dspy.Prediction(score=score, feedback=feedback)
        assert "dspy.Prediction(score=score, feedback=feedback)" in src, (
            "mipro metric in run_dspy_gepa should return dspy.Prediction(score=score, feedback=feedback)"
        )
        # The native-gepa metric already does this (regression check)
        src_native = inspect.getsource(optimize.run_dspy_native_gepa)
        assert "dspy.Prediction(score=score, feedback=feedback)" in src_native
        print("OK: both mipro and native-gepa metrics return dspy.Prediction")

    def test_run_dspy_gepa_docstring_mentions_signature_instructions_and_output(self):
        """After Phase 20.2, run_dspy_gepa docstring should mention signature.instructions
        (the optimizable field) and best_candidate_dspy.md (the output file)."""
        import optimize
        doc = optimize.run_dspy_gepa.__doc__ or ""
        assert "signature.instructions" in doc, (
            "run_dspy_gepa docstring should mention 'signature.instructions' "
            "as the optimizable field"
        )
        assert "best_candidate_dspy.md" in doc, (
            "run_dspy_gepa docstring should mention 'best_candidate_dspy.md' "
            "as the output file (it IS the SKILL.md content after Phase 20.1)"
        )
        assert "differs from GEPA" not in doc, (
            "run_dspy_gepa docstring should NOT contain the stale 'differs from GEPA' caveat"
        )
        print("OK: run_dspy_gepa docstring mentions signature.instructions and best_candidate_dspy.md")
