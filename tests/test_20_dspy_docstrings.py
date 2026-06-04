"""Test docstring accuracy for DSPy runners (Phase 20.2).

These tests verify that the docstrings of run_dspy_gepa and run_dspy_native_gepa
have been correctly updated after Phase 20.2 to reflect that:
1. The seed skill content is in signature.instructions (via SkillGuidedTask.with_instructions)
2. The output is an optimized SKILL.md (not "NOT a SKILL.md")
3. Stale caveats like "differs from GEPA" have been removed
"""
import inspect
import unittest
from typing import Any

import optimize


def _get_doc(func: Any) -> str:
    """Get docstring, failing with clear message if None."""
    doc = inspect.getdoc(func)
    if doc is None:
        raise AssertionError(f"{func.__name__} has no docstring")
    return doc


class TestRunDspyGepaDocstring(unittest.TestCase):
    """Tests for run_dspy_gepa docstring accuracy."""

    def test_run_dspy_gepa_docstring_no_stale_text(self):
        """Assert the docstring does NOT contain stale claim text."""
        docstring = _get_doc(optimize.run_dspy_gepa)

        # These are the two pieces of stale text that Phase 20.2 removed
        self.assertNotIn(
            "NOT a SKILL.md",
            docstring,
            "run_dspy_gepa still contains stale claim 'NOT a SKILL.md'"
        )
        self.assertNotIn(
            "differs from GEPA",
            docstring,
            "run_dspy_gepa still contains stale caveat 'differs from GEPA'"
        )

    def test_run_dspy_gepa_docstring_mentions_signature_instructions(self):
        """Assert the docstring references signature.instructions."""
        docstring = _get_doc(optimize.run_dspy_gepa)

        self.assertIn(
            "signature.instructions",
            docstring,
            "run_dspy_gepa docstring must mention signature.instructions"
        )

    def test_run_dspy_gepa_docstring_mentions_output_file(self):
        """Assert the docstring mentions the output file."""
        docstring = _get_doc(optimize.run_dspy_gepa)

        self.assertIn(
            "best_candidate_dspy.md",
            docstring,
            "run_dspy_gepa docstring must mention best_candidate_dspy.md"
        )


class TestRunDspyNativeGepaDocstring(unittest.TestCase):
    """Tests for run_dspy_native_gepa docstring accuracy."""

    def test_run_dspy_native_gepa_docstring_no_stale_text(self):
        """Assert the docstring does NOT contain stale claim text."""
        docstring = _get_doc(optimize.run_dspy_native_gepa)

        # These are the two pieces of stale text that Phase 20.2 removed
        self.assertNotIn(
            "NOT a SKILL.md",
            docstring,
            "run_dspy_native_gepa still contains stale claim 'NOT a SKILL.md'"
        )
        self.assertNotIn(
            "differs from GEPA",
            docstring,
            "run_dspy_native_gepa still contains stale caveat 'differs from GEPA'"
        )

    def test_run_dspy_native_gepa_docstring_mentions_signature_instructions(self):
        """Assert the docstring references signature.instructions."""
        docstring = _get_doc(optimize.run_dspy_native_gepa)

        self.assertIn(
            "signature.instructions",
            docstring,
            "run_dspy_native_gepa docstring must mention signature.instructions"
        )

    def test_run_dspy_native_gepa_docstring_mentions_output_file(self):
        """Assert the docstring mentions the output file."""
        docstring = _get_doc(optimize.run_dspy_native_gepa)

        self.assertIn(
            "best_candidate_dspy.md",
            docstring,
            "run_dspy_native_gepa docstring must mention best_candidate_dspy.md"
        )


class TestDocstringsParallel(unittest.TestCase):
    """Tests for parallel correctness across both docstrings."""

    def test_docstrings_mention_refactor_details(self):
        """Both docstrings should mention the Phase 20.1 refactor details."""
        docstring_gepa = _get_doc(optimize.run_dspy_gepa)
        docstring_native = _get_doc(optimize.run_dspy_native_gepa)

        # Both should mention SkillGuidedTask.with_instructions
        self.assertIn(
            "SkillGuidedTask.with_instructions(seed_candidate)",
            docstring_gepa,
            "run_dspy_gepa must mention SkillGuidedTask.with_instructions"
        )
        self.assertIn(
            "SkillGuidedTask.with_instructions(seed_candidate)",
            docstring_native,
            "run_dspy_native_gepa must mention SkillGuidedTask.with_instructions"
        )

        # Both should mention src.dspy_shared.SkillProgram
        self.assertIn(
            "src.dspy_shared.SkillProgram",
            docstring_gepa,
            "run_dspy_gepa must mention src.dspy_shared.SkillProgram"
        )
        self.assertIn(
            "src.dspy_shared.SkillProgram",
            docstring_native,
            "run_dspy_native_gepa must mention src.dspy_shared.SkillProgram"
        )


class TestFunctionSignaturesUnchanged(unittest.TestCase):
    """Verify function signatures have not changed (docstring update is pure)."""

    def test_function_signatures_unchanged(self):
        """Assert run_dspy_gepa and run_dspy_native_gepa have same parameters."""
        sig_gepa = inspect.signature(optimize.run_dspy_gepa)
        sig_native = inspect.signature(optimize.run_dspy_native_gepa)

        # Extract parameter names
        params_gepa = list(sig_gepa.parameters.keys())
        params_native = list(sig_native.parameters.keys())

        # Expected parameters for both functions
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

        # Assert both have the expected parameters
        self.assertEqual(
            params_gepa,
            expected_params,
            "run_dspy_gepa signature mismatch"
        )
        self.assertEqual(
            params_native,
            expected_params,
            "run_dspy_native_gepa signature mismatch"
        )


if __name__ == "__main__":
    unittest.main()