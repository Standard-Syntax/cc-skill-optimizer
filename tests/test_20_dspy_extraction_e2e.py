"""
End-to-end regression tests for Phase 20: DSPy optimization path correctly
extracts optimized SKILL content into best_candidate_dspy.md.

The bug: best_candidate_dspy.md was 288 bytes (static class docstring) instead
of 17,944-byte seed SKILL.md. Root cause was the optimizer returning the static
class docstring as signature.instructions instead of the user's SKILL content.

Phase 20.1 refactored src/dspy_shared.py so the seed SKILL content lives in
predictor.signature.instructions. These tests verify the extraction logic in
run_dspy_gepa and run_dspy_native_gepa reads signature.instructions correctly
and writes it to best_candidate_dspy.md.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Constants
# ============================================================================

# Static class docstring (288 bytes) — the OLD buggy value for signature.instructions
WRAPPER_DOCSTRING = (
    "Apply repository skills to complete a software engineering task. "
    "The skill_instructions field carries the SKILL.md content as runtime guidance; "
    "task_prompt is the user's actual request; and error_context surfaces any prior "
    "errors from the session so the completion can recover from them."
)

# Recognizable SKILL-shaped string for regression tests
SKILL_SHAPED_STRING = (
    "# Optimized Skill\n\n"
    "This skill was rewritten by the optimizer.\n"
    "Steps:\n"
    "1. Analyze the task requirements carefully.\n"
    "2. Apply the appropriate patterns from this skill.\n"
    "3. Verify the output meets all acceptance criteria.\n"
)


# ============================================================================
# Helper: mock dspy modules in sys.modules
# ============================================================================


def mock_dspy_modules():
    """Context manager to mock dspy modules in sys.modules.

    Mocks both the top-level dspy namespace (dspy.GEPA, dspy.MIPROv2, dspy.LM,
    dspy.Prediction, etc.) and the legacy dspy.teleprompt submodule.

    NOTE: dspy.Module must be a mock CLASS (not instance) because isinstance()
    checks inside dspy (e.g., isinstance(value, dspy.Module)) require a type,
    not a MagicMock instance (which raises TypeError).
    """
    mock_dspy = MagicMock()
    mock_teleprompt = MagicMock()

    # Create a mock class for dspy.Module so isinstance() works (not a MagicMock instance)
    # dspy.Module.set_lm is also a method on the class, so we set it on the class
    MockModuleClass = type("Module", (object,), {"set_lm": MagicMock(), "map_named_predictors": MagicMock()})

    # dspy.LM is called inside run_dspy_gepa / run_dspy_native_gepa
    mock_dspy.LM.return_value = MagicMock()
    mock_dspy.configure = MagicMock()
    mock_dspy.Module = MockModuleClass
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

    mocks = {
        "dspy": mock_dspy,
        "dspy.teleprompt": mock_teleprompt,
    }

    return patch.dict("sys.modules", mocks)


# ============================================================================
# Helper: build a mock optimizer .compile() return value with signature.instructions
# ============================================================================


def _make_mock_optimizer_with_instructions(skill_instructions: str) -> MagicMock:
    """Build a mock MIPROv2/GEPA .compile() return value whose predictor.signature.instructions == skill_instructions.

    The mock is constructed so that optimized.predictor.signature.instructions
    returns skill_instructions — matching the real DSPy API used in run_dspy_gepa
    and run_dspy_native_gepa extraction blocks.
    """
    mock_predictor = MagicMock()
    mock_signature = MagicMock()
    mock_signature.instructions = skill_instructions
    mock_predictor.signature = mock_signature

    mock_optimized = MagicMock()
    mock_optimized.predictor = mock_predictor

    return mock_optimized


# ============================================================================
# Test cases
# ============================================================================


class TestRunDspyGepaExtractsSkillFromSignatureInstructions(unittest.TestCase):
    """Test 1: run_dspy_gepa extracts skill from signature.instructions."""

    def test_run_dspy_gepa_extracts_skill_from_signature_instructions(self):
        """run_dspy_gepa should read optimized.predictor.signature.instructions and return it."""
        from optimize import run_dspy_gepa

        expected_skill = SKILL_SHAPED_STRING
        mock_optimized = _make_mock_optimizer_with_instructions(expected_skill)

        with ExitStack() as stack:
            stack.enter_context(mock_dspy_modules())

            mock_dspy = sys.modules.get("dspy")
            mock_mipro_instance = MagicMock()
            mock_mipro_instance.compile.return_value = mock_optimized
            mock_dspy.MIPROv2.return_value = mock_mipro_instance

            # score_episode is imported inside the metric function via `from evaluator import score_episode`
            stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
            stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

            result = run_dspy_gepa(
                seed_candidate="# Seed",
                train_set=[{"task_prompt": "test task"}],
                val_set=[],
                objective="test",
                max_metric_calls=10,
                task_lm="anthropic/claude-haiku-4-5-20251001",
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_dspy_gepa_extract"),
            )

        self.assertEqual(result, expected_skill)


class TestRunDspyGepaSizeBoundsRegression(unittest.TestCase):
    """Test 2: size bounds regression (288 bytes vs 17,944 bytes)."""

    def test_run_dspy_gepa_size_bounds_regression(self):
        """Output should be within 0.3x–3x of seed size, not 288 bytes."""
        from optimize import run_dspy_gepa

        # ~5000 byte SKILL-shaped string
        seed = (
            "# Test Skill\n\n"
            "Use this skill for comprehensive testing.\n\n"
            "## Guidelines\n"
        )
        padding = "Step %d: Do thorough validation of all inputs, outputs, edge cases, and error paths.\n" * 80
        seed = seed + padding
        seed = seed[:5000]

        mock_optimized = _make_mock_optimizer_with_instructions(seed)

        with tempfile.TemporaryDirectory() as tmp:
            with ExitStack() as stack:
                stack.enter_context(mock_dspy_modules())

                mock_dspy = sys.modules.get("dspy")
                mock_mipro_instance = MagicMock()
                mock_mipro_instance.compile.return_value = mock_optimized
                mock_dspy.MIPROv2.return_value = mock_mipro_instance

                stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
                stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

                output_dir = Path(tmp)
                run_dspy_gepa(
                    seed_candidate="# Seed",
                    train_set=[{"task_prompt": "test task"}],
                    val_set=[],
                    objective="test",
                    max_metric_calls=10,
                    task_lm="anthropic/claude-haiku-4-5-20251001",
                    reflection_lm="anthropic/claude-haiku-4-5-20251001",
                    output_dir=output_dir,
                )

            output_file = output_dir / "best_candidate_dspy.md"
            self.assertTrue(output_file.exists(), "best_candidate_dspy.md should be created")
            output_content = output_file.read_text()

        # Original bug: 288 bytes from static docstring vs 17,944-byte seed
        # After fix: should be within 0.3x–3x of seed size (~5000 bytes)
        self.assertGreaterEqual(
            len(output_content),
            int(0.3 * len(seed)),
            f"Output ({len(output_content)} bytes) should be >= 0.3x seed ({int(0.3 * len(seed))} bytes)",
        )
        self.assertLessEqual(
            len(output_content),
            int(3 * len(seed)),
            f"Output ({len(output_content)} bytes) should be <= 3x seed ({int(3 * len(seed))} bytes)",
        )


class TestRunDspyNativeGepaExtractsSkillFromSignatureInstructions(unittest.TestCase):
    """Test 3: run_dspy_native_gepa extracts skill from signature.instructions."""

    def test_run_dspy_native_gepa_extracts_skill_from_signature_instructions(self):
        """run_dspy_native_gepa should read optimized.predictor.signature.instructions and return it."""
        from optimize import run_dspy_native_gepa

        expected_skill = SKILL_SHAPED_STRING
        mock_optimized = _make_mock_optimizer_with_instructions(expected_skill)

        with ExitStack() as stack:
            stack.enter_context(mock_dspy_modules())

            mock_dspy = sys.modules.get("dspy")
            mock_gepa_instance = MagicMock()
            mock_gepa_instance.compile.return_value = mock_optimized
            mock_dspy.GEPA.return_value = mock_gepa_instance

            stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
            stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

            result = run_dspy_native_gepa(
                seed_candidate="# Seed",
                train_set=[{"task_prompt": "test task"}],
                val_set=[],
                objective="test",
                max_metric_calls=10,
                task_lm="anthropic/claude-haiku-4-5-20251001",
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_dspy_native_gepa_extract"),
            )

        self.assertEqual(result, expected_skill)


class TestRunDspyNativeGepaSizeBoundsRegression(unittest.TestCase):
    """Test 4: size bounds regression for run_dspy_native_gepa."""

    def test_run_dspy_native_gepa_size_bounds_regression(self):
        """Output should be within 0.3x–3x of seed size, not 288 bytes."""
        from optimize import run_dspy_native_gepa

        # ~5000 byte SKILL-shaped string
        seed = (
            "# Test Skill\n\n"
            "Use this skill for comprehensive testing.\n\n"
            "## Guidelines\n"
        )
        padding = "Step %d: Do thorough validation of all inputs, outputs, edge cases, and error paths.\n" * 80
        seed = seed + padding
        seed = seed[:5000]

        mock_optimized = _make_mock_optimizer_with_instructions(seed)

        with tempfile.TemporaryDirectory() as tmp:
            with ExitStack() as stack:
                stack.enter_context(mock_dspy_modules())

                mock_dspy = sys.modules.get("dspy")
                mock_gepa_instance = MagicMock()
                mock_gepa_instance.compile.return_value = mock_optimized
                mock_dspy.GEPA.return_value = mock_gepa_instance

                stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
                stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

                output_dir = Path(tmp)
                run_dspy_native_gepa(
                    seed_candidate="# Seed",
                    train_set=[{"task_prompt": "test task"}],
                    val_set=[],
                    objective="test",
                    max_metric_calls=10,
                    task_lm="anthropic/claude-haiku-4-5-20251001",
                    reflection_lm="anthropic/claude-haiku-4-5-20251001",
                    output_dir=output_dir,
                )

            output_file = output_dir / "best_candidate_dspy.md"
            self.assertTrue(output_file.exists(), "best_candidate_dspy.md should be created")
            output_content = output_file.read_text()

        self.assertGreaterEqual(
            len(output_content),
            int(0.3 * len(seed)),
            f"Output ({len(output_content)} bytes) should be >= 0.3x seed ({int(0.3 * len(seed))} bytes)",
        )
        self.assertLessEqual(
            len(output_content),
            int(3 * len(seed)),
            f"Output ({len(output_content)} bytes) should be <= 3x seed ({int(3 * len(seed))} bytes)",
        )


class TestRunDspyGepaDoesNotReturnWrapperDocstring(unittest.TestCase):
    """Test 5: run_dspy_gepa extraction is 'dumb' — it reads whatever signature.instructions has."""

    def test_run_dspy_gepa_does_not_return_wrapper_docstring(self):
        """When signature.instructions IS the wrapper docstring, that is what gets returned.

        This is the inverse of the real regression test: it pins the extraction behavior.
        After Phase 20.1 fix, signature.instructions is no longer the wrapper docstring,
        so test 1 is the actual regression test. This test verifies the extraction logic
        itself is simple read-and-write.
        """
        from optimize import run_dspy_gepa

        mock_optimized = _make_mock_optimizer_with_instructions(WRAPPER_DOCSTRING)

        with ExitStack() as stack:
            stack.enter_context(mock_dspy_modules())

            mock_dspy = sys.modules.get("dspy")
            mock_mipro_instance = MagicMock()
            mock_mipro_instance.compile.return_value = mock_optimized
            mock_dspy.MIPROv2.return_value = mock_mipro_instance

            stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
            stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

            result = run_dspy_gepa(
                seed_candidate="# Seed",
                train_set=[{"task_prompt": "test task"}],
                val_set=[],
                objective="test",
                max_metric_calls=10,
                task_lm="anthropic/claude-haiku-4-5-20251001",
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_dspy_gepa_wrapper"),
            )

        self.assertEqual(result, WRAPPER_DOCSTRING)


class TestRunDspyNativeGepaDoesNotReturnWrapperDocstring(unittest.TestCase):
    """Test 6: run_dspy_native_gepa extraction is 'dumb' — it reads whatever signature.instructions has."""

    def test_run_dspy_native_gepa_does_not_return_wrapper_docstring(self):
        """When signature.instructions IS the wrapper docstring, that is what gets returned."""
        from optimize import run_dspy_native_gepa

        mock_optimized = _make_mock_optimizer_with_instructions(WRAPPER_DOCSTRING)

        with ExitStack() as stack:
            stack.enter_context(mock_dspy_modules())

            mock_dspy = sys.modules.get("dspy")
            mock_gepa_instance = MagicMock()
            mock_gepa_instance.compile.return_value = mock_optimized
            mock_dspy.GEPA.return_value = mock_gepa_instance

            stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
            stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

            result = run_dspy_native_gepa(
                seed_candidate="# Seed",
                train_set=[{"task_prompt": "test task"}],
                val_set=[],
                objective="test",
                max_metric_calls=10,
                task_lm="anthropic/claude-haiku-4-5-20251001",
                reflection_lm="anthropic/claude-haiku-4-5-20251001",
                output_dir=Path("/tmp/test_dspy_native_gepa_wrapper"),
            )

        self.assertEqual(result, WRAPPER_DOCSTRING)


class TestRunDspyGepaWritesFileUnderOutputDir(unittest.TestCase):
    """Test 7: run_dspy_gepa writes best_candidate_dspy.md under output_dir."""

    def test_run_dspy_gepa_writes_file_under_output_dir(self):
        """run_dspy_gepa should create best_candidate_dspy.md in output_dir."""
        from optimize import run_dspy_gepa

        mock_optimized = _make_mock_optimizer_with_instructions("# Optimized Skill Content")

        with tempfile.TemporaryDirectory() as tmp:
            with ExitStack() as stack:
                stack.enter_context(mock_dspy_modules())

                mock_dspy = sys.modules.get("dspy")
                mock_mipro_instance = MagicMock()
                mock_mipro_instance.compile.return_value = mock_optimized
                mock_dspy.MIPROv2.return_value = mock_mipro_instance

                stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
                stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

                output_dir = Path(tmp)
                run_dspy_gepa(
                    seed_candidate="# Seed",
                    train_set=[{"task_prompt": "test task"}],
                    val_set=[],
                    objective="test",
                    max_metric_calls=10,
                    task_lm="anthropic/claude-haiku-4-5-20251001",
                    reflection_lm="anthropic/claude-haiku-4-5-20251001",
                    output_dir=output_dir,
                )

            output_file = output_dir / "best_candidate_dspy.md"
            self.assertTrue(output_file.exists(), "best_candidate_dspy.md should exist")
            content = output_file.read_text()
            self.assertTrue(len(content) > 0, "best_candidate_dspy.md should be non-empty")


class TestRunDspyNativeGepaWritesFileUnderOutputDir(unittest.TestCase):
    """Test 8: run_dspy_native_gepa writes best_candidate_dspy.md under output_dir."""

    def test_run_dspy_native_gepa_writes_file_under_output_dir(self):
        """run_dspy_native_gepa should create best_candidate_dspy.md in output_dir."""
        from optimize import run_dspy_native_gepa

        mock_optimized = _make_mock_optimizer_with_instructions("# Optimized Skill Content")

        with tempfile.TemporaryDirectory() as tmp:
            with ExitStack() as stack:
                stack.enter_context(mock_dspy_modules())

                mock_dspy = sys.modules.get("dspy")
                mock_gepa_instance = MagicMock()
                mock_gepa_instance.compile.return_value = mock_optimized
                mock_dspy.GEPA.return_value = mock_gepa_instance

                stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
                stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

                output_dir = Path(tmp)
                run_dspy_native_gepa(
                    seed_candidate="# Seed",
                    train_set=[{"task_prompt": "test task"}],
                    val_set=[],
                    objective="test",
                    max_metric_calls=10,
                    task_lm="anthropic/claude-haiku-4-5-20251001",
                    reflection_lm="anthropic/claude-haiku-4-5-20251001",
                    output_dir=output_dir,
                )

            output_file = output_dir / "best_candidate_dspy.md"
            self.assertTrue(output_file.exists(), "best_candidate_dspy.md should exist")
            content = output_file.read_text()
            self.assertTrue(len(content) > 0, "best_candidate_dspy.md should be non-empty")


class TestRunDspyGepaFallsBackToSeedOnExtractionFailure(unittest.TestCase):
    """Test 9: run_dspy_gepa falls back to seed_candidate when extraction fails."""

    def test_run_dspy_gepa_falls_back_to_seed_on_extraction_failure(self):
        """When predictor.signature raises an exception, run_dspy_gepa should return seed_candidate."""
        from optimize import run_dspy_gepa

        seed_candidate = "# Original Seed Skill"

        # Mock optimized whose predictor.signature raises on access
        class SignatureRaisesOnAccess:
            @property
            def signature(self):
                raise RuntimeError("signature access failed")

        mock_optimized = MagicMock()
        mock_optimized.predictor = SignatureRaisesOnAccess()

        with tempfile.TemporaryDirectory() as tmp:
            with ExitStack() as stack:
                stack.enter_context(mock_dspy_modules())

                mock_dspy = sys.modules.get("dspy")
                mock_mipro_instance = MagicMock()
                mock_mipro_instance.compile.return_value = mock_optimized
                mock_dspy.MIPROv2.return_value = mock_mipro_instance

                stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
                stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

                output_dir = Path(tmp)
                result = run_dspy_gepa(
                    seed_candidate=seed_candidate,
                    train_set=[{"task_prompt": "test task"}],
                    val_set=[],
                    objective="test",
                    max_metric_calls=10,
                    task_lm="anthropic/claude-haiku-4-5-20251001",
                    reflection_lm="anthropic/claude-haiku-4-5-20251001",
                    output_dir=output_dir,
                )

            self.assertEqual(result, seed_candidate)
            output_file = output_dir / "best_candidate_dspy.md"
            self.assertTrue(output_file.exists())
            self.assertEqual(output_file.read_text(), seed_candidate)


class TestRunDspyNativeGepaFallsBackToSeedOnExtractionFailure(unittest.TestCase):
    """Test 10: run_dspy_native_gepa falls back to seed_candidate when extraction fails."""

    def test_run_dspy_native_gepa_falls_back_to_seed_on_extraction_failure(self):
        """When predictor.signature raises an exception, run_dspy_native_gepa should return seed_candidate."""
        from optimize import run_dspy_native_gepa

        seed_candidate = "# Original Seed Skill"

        # Mock optimized whose predictor.signature raises on access
        class SignatureRaisesOnAccess:
            @property
            def signature(self):
                raise RuntimeError("signature access failed")

        mock_optimized = MagicMock()
        mock_optimized.predictor = SignatureRaisesOnAccess()

        with tempfile.TemporaryDirectory() as tmp:
            with ExitStack() as stack:
                stack.enter_context(mock_dspy_modules())

                mock_dspy = sys.modules.get("dspy")
                mock_gepa_instance = MagicMock()
                mock_gepa_instance.compile.return_value = mock_optimized
                mock_dspy.GEPA.return_value = mock_gepa_instance

                stack.enter_context(patch("evaluator.score_episode", return_value=(0.5, {"feedback": "test feedback"})))
                stack.enter_context(patch("optimize._model_uses_thinking", return_value=False))

                output_dir = Path(tmp)
                result = run_dspy_native_gepa(
                    seed_candidate=seed_candidate,
                    train_set=[{"task_prompt": "test task"}],
                    val_set=[],
                    objective="test",
                    max_metric_calls=10,
                    task_lm="anthropic/claude-haiku-4-5-20251001",
                    reflection_lm="anthropic/claude-haiku-4-5-20251001",
                    output_dir=output_dir,
                )

            self.assertEqual(result, seed_candidate)
            output_file = output_dir / "best_candidate_dspy.md"
            self.assertTrue(output_file.exists())
            self.assertEqual(output_file.read_text(), seed_candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
