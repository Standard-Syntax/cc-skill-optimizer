"""
Verify Phase 20 refactor: SkillProgram / SkillGuidedTask in src/dspy_shared.py.

The bug: outputs/skill/best_candidate_dspy.md was 288 bytes (static class docstring)
instead of 17,944-byte seed SKILL.md. Root cause was skill_content
passed as a dspy.InputField that MIPROv2/GEPA don't rewrite.

The fix: skill content lives in signature.instructions, not an InputField.
"""

import inspect
import sys
import unittest
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from src.dspy_shared import SkillProgram


class TestSkillContentInSignature(unittest.TestCase):
    """Tests for the core fix: skill content lands in signature.instructions."""

    def test_skill_content_lands_in_signature_instructions(self):
        """Core regression test: skill content must be in signature.instructions."""
        skill_text = "Use this skill to do X, Y, Z. Steps: 1) ... 2) ..."
        program = SkillProgram(skill_text)

        # The fix: skill content is now in signature.instructions
        self.assertIn(
            "Use this skill to do X, Y, Z.",
            program.predictor.signature.instructions,
        )

    def test_skill_instructions_field_removed_from_signature(self):
        """Verify skill_instructions InputField is no longer in signature."""
        program = SkillProgram("# any skill content")

        field_keys = list(program.predictor.signature.fields.keys())
        self.assertNotIn("skill_instructions", field_keys)

    def test_signature_has_expected_fields_only(self):
        """Signature should have exactly task_prompt, error_context, completion."""
        program = SkillProgram("# any skill content")

        actual_fields = set(program.predictor.signature.fields.keys())
        expected_fields = {"task_prompt", "error_context", "completion"}
        self.assertEqual(actual_fields, expected_fields)


class TestSizeBoundsRegression(unittest.TestCase):
    """Test size bounds that would catch 288-byte vs 17,944-byte regression."""

    def test_size_bounds_regression(self):
        """Assert signature instructions are at least 0.3x the seed size."""
        # Create a ~5000 byte skill body
        seed = (
            "# Test Skill\n\n"
            "Use this skill for comprehensive testing.\n\n"
            "## Guidelines\n"
        )
        # Pad to ~5000 characters
        padding = "Step %d: Do thorough validation of all inputs, outputs, edge cases, and error paths.\n" * 80
        seed = seed + padding
        # Ensure it's around 5000 bytes as requested
        seed = seed[:5000]

        program = SkillProgram(seed)
        sig_instructions = program.predictor.signature.instructions

        # Should not be empty (the old bug was empty or static docstring)
        self.assertTrue(len(sig_instructions) > 0)
        # Should be at least 30% of seed size
        min_expected = int(0.3 * len(seed))
        self.assertGreaterEqual(
            len(sig_instructions),
            min_expected,
            f"Expected >= {min_expected} chars, got {len(sig_instructions)}",
        )


class TestSkillContentAttributePreserved(unittest.TestCase):
    """Verify self.skill_content is still preserved as instance attribute."""

    def test_program_skill_content_attribute_preserved(self):
        """The refactor keeps self.skill_content as an instance attribute."""
        program = SkillProgram("original content")

        self.assertEqual(program.skill_content, "original content")


class TestForwardCallSignature(unittest.TestCase):
    """Verify SkillProgram.forward has correct signature (no skill_instructions)."""

    def test_forward_call_signature(self):
        """Forward should take task_prompt and error_context, not skill_instructions."""
        sig = inspect.signature(SkillProgram.forward)
        param_names = list(sig.parameters.keys())

        # Should be ('self', 'task_prompt', 'error_context')
        self.assertEqual(param_names, ["self", "task_prompt", "error_context"])

    def test_forward_call_accepts_error_context_default(self):
        """Verify error_context has empty string default."""
        sig = inspect.signature(SkillProgram.forward)
        error_context_param = sig.parameters.get("error_context")

        self.assertIsNotNone(error_context_param)
        self.assertEqual(error_context_param.default, "")


class TestSourceLevelChecks(unittest.TestCase):
    """Source-level checks to verify the implementation approach."""

    def test_dspy_predict_uses_with_instructions(self):
        """Verify dspy.Predict uses with_instructions() method."""
        from src import dspy_shared

        source = inspect.getsource(dspy_shared)
        # Find where Predict is instantiated
        predict_line = None
        for line in source.split("\n"):
            if "dspy.Predict(" in line:
                predict_line = line
                break

        self.assertIsNotNone(predict_line, "Could not find dspy.Predict instantiation")
        self.assertIn(
            "with_instructions(",
            predict_line,
            f"Expected 'with_instructions' in dspy.Predict call: {predict_line}",
        )

    def test_no_skill_instructions_inputfield_in_source(self):
        """Verify no skill_instructions InputField exists in source."""
        from src import dspy_shared

        source = inspect.getsource(dspy_shared)

        # The buggy pattern was: skill_instructions: str = dspy.InputField(
        bad_pattern = "skill_instructions: str = dspy.InputField("
        self.assertNotIn(
            bad_pattern,
            source,
            "Found buggy skill_instructions InputField in source",
        )


if __name__ == "__main__":
    unittest.main()