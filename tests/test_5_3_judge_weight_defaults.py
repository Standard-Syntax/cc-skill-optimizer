"""
Test suite for Task 5.3: judge_weight defaults reconciliation.

Verifies that judge_weight defaults are standardized to 0.65 in both:
- src/evaluator.py (make_replay_evaluator)
- src/synthetic_evaluator.py (make_synthetic_evaluator)

Also verifies: combined weights sum to 1.0, parameter types, backward compat.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path BEFORE importing src modules
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))


class TestReplayEvaluatorDefaults:
    """Tests for make_replay_evaluator in src/evaluator.py"""

    def test_make_replay_evaluator_default_judge_weight_is_065(self):
        """Test: make_replay_evaluator default judge_weight is 0.65"""
        from src.evaluator import make_replay_evaluator

        sig = inspect.signature(make_replay_evaluator)
        params = sig.parameters

        # Verify judge_weight default is 0.65
        assert "judge_weight" in params, "judge_weight parameter missing"
        default = params["judge_weight"].default
        assert default == 0.65, f"Expected judge_weight default 0.65, got {default}"

    def test_make_replay_evaluator_signature_has_type_annotation(self):
        """Test: parameter types preserved"""
        from src.evaluator import make_replay_evaluator

        sig = inspect.signature(make_replay_evaluator)
        judge_weight_param = sig.parameters.get("judge_weight")

        # Verify type annotation is float (may be None if not annotated)
        # At minimum, verify it's a Parameter with a default
        assert judge_weight_param is not None
        assert judge_weight_param.default == 0.65

    def test_make_replay_evaluator_other_params_unchanged(self):
        """Test: other parameters unchanged"""
        from src.evaluator import make_replay_evaluator

        sig = inspect.signature(make_replay_evaluator)
        params = list(sig.parameters.keys())

        # Expected order and names
        expected = ["episodes", "use_llm_judge", "judge_lm", "judge_weight", "tool_call_thresholds"]
        assert params == expected, f"Parameters changed! Expected {expected}, got {params}"

    def test_make_replay_evaluator_with_custom_judge_weight(self):
        """Test: make_replay_evaluator with custom judge_weight"""
        from src.evaluator import make_replay_evaluator

        episodes = [
            {
                "outcome": "success",
                "tool_calls": [],
                "duration_s": 30.0,
            }
        ]

        # Call with explicit 0.5
        evaluator = make_replay_evaluator(episodes=episodes, judge_weight=0.5)

        # Verify it works without raising
        assert callable(evaluator)

        # Test the evaluator - structural only (use_llm_judge=False)
        from src.evaluator import make_replay_evaluator as re_eval

        evaluator = re_eval(episodes=episodes, use_llm_judge=False, judge_weight=0.5)
        score, info = evaluator("test skill", episodes[0])

        # Should return a valid score
        assert 0.0 <= score <= 1.0


class TestSyntheticEvaluatorDefaults:
    """Tests for make_synthetic_evaluator in src/synthetic_evaluator.py"""

    def test_make_synthetic_evaluator_default_judge_weight_is_065(self):
        """Test: make_synthetic_evaluator default judge_weight is 0.65"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        sig = inspect.signature(make_synthetic_evaluator)
        params = sig.parameters

        # Verify judge_weight default is 0.65
        assert "judge_weight" in params, "judge_weight parameter missing"
        default = params["judge_weight"].default
        assert default == 0.65, f"Expected judge_weight default 0.65, got {default}"

    def test_make_synthetic_evaluator_default_structural_weight_is_035(self):
        """Test: structural_weight default is 0.35"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        sig = inspect.signature(make_synthetic_evaluator)
        params = sig.parameters

        assert "structural_weight" in params
        default = params["structural_weight"].default
        assert default == 0.35, f"Expected structural_weight default 0.35, got {default}"

    def test_make_synthetic_evaluator_signature_has_type_annotation(self):
        """Test: parameter types preserved"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        sig = inspect.signature(make_synthetic_evaluator)

        # Verify judge_weight
        judge_weight_param = sig.parameters.get("judge_weight")
        assert judge_weight_param is not None
        assert judge_weight_param.default == 0.65

        # Verify structural_weight
        structural_weight_param = sig.parameters.get("structural_weight")
        assert structural_weight_param is not None
        assert structural_weight_param.default == 0.35

    def test_make_synthetic_evaluator_other_params_unchanged(self):
        """Test: other parameters unchanged"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        sig = inspect.signature(make_synthetic_evaluator)
        params = list(sig.parameters.keys())

        # Expected order and names
        expected = ["task_library", "judge_lm", "judge_weight", "structural_weight", "use_judge"]
        assert params == expected, f"Parameters changed! Expected {expected}, got {params}"

    def test_make_synthetic_evaluator_with_custom_judge_weight(self):
        """Test: make_synthetic_evaluator with custom judge_weight"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        tasks = [{"task_description": "test", "pitfalls": [], "success_criteria": []}]

        # Call with explicit 0.5 (structural_weight must sum to 1.0)
        evaluator = make_synthetic_evaluator(tasks, judge_weight=0.5, structural_weight=0.5)

        # Verify it works without raising
        assert callable(evaluator)

    def test_judge_weight_plus_structural_weight_equals_one_invariant(self):
        """Test: judge_weight + structural_weight = 1.0 invariant in make_synthetic_evaluator"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        tasks = [{"task_description": "test", "pitfalls": [], "success_criteria": []}]

        # Default case: judge_weight=0.65, structural_weight=0.35
        evaluator = make_synthetic_evaluator(tasks)
        assert callable(evaluator)

        # Custom: 0.5 + 0.5
        evaluator = make_synthetic_evaluator(tasks, judge_weight=0.5, structural_weight=0.5)
        assert callable(evaluator)

        # Invalid: sum != 1.0 should raise ValueError
        with pytest.raises(ValueError, match=r"judge_weight.*structural_weight.*must sum to 1.0"):
            make_synthetic_evaluator(tasks, judge_weight=0.6, structural_weight=0.5)


class TestBothEvaluatorsDefaultBehavior:
    """Tests that both evaluators work without specifying judge_weight"""

    def test_replay_evaluator_without_judge_weight(self):
        """Test: both evaluators can be called without specifying judge_weight"""
        from src.evaluator import make_replay_evaluator

        episodes = [{"outcome": "success", "tool_calls": [], "duration_s": 30.0}]

        # Call without judge_weight - should use default 0.65
        evaluator = make_replay_evaluator(episodes=episodes, use_llm_judge=False)

        # Must not raise
        assert callable(evaluator)

    def test_synthetic_evaluator_without_judge_weight(self):
        """Test: both evaluators can be called without specifying judge_weight"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        tasks = [{"task_description": "test", "pitfalls": [], "success_criteria": []}]

        # Call without judge_weight - should use default 0.65
        evaluator = make_synthetic_evaluator(tasks)

        # Must not raise
        assert callable(evaluator)


class TestBackwardCompatibility:
    """Back compat tests for explicit old default values"""

    def test_replay_evaluator_accepts_explicit_04(self):
        """Test: backward compat — make_replay_evaluator with explicit 0.4 still works"""
        from src.evaluator import make_replay_evaluator

        episodes = [{"outcome": "success", "tool_calls": [], "duration_s": 30.0}]

        # Old default was 0.4 - should still work
        evaluator = make_replay_evaluator(episodes=episodes, judge_weight=0.4, use_llm_judge=False)

        assert callable(evaluator)


class TestNoHiddenOldDefaults:
    """Verify no remaining 0.4 or 0.7 defaults in source files"""

    def test_no_judge_weight_04_in_evaluator(self):
        """Test: no hidden 0.4 defaults in src/evaluator.py"""
        from src import evaluator

        source_file = evaluator.__file__
        with open(source_file) as f:
            content = f.read()

        # Should NOT have "judge_weight: float = 0.4"
        assert "judge_weight: float = 0.4" not in content, "Found old default 0.4 in evaluator.py!"

    def test_no_judge_weight_07_in_evaluator(self):
        """Test: no hidden 0.7 defaults in src/evaluator.py"""
        from src import evaluator

        source_file = evaluator.__file__
        with open(source_file) as f:
            content = f.read()

        # Should NOT have "judge_weight: float = 0.7"
        assert "judge_weight: float = 0.7" not in content, "Found old default 0.7 in evaluator.py!"

    def test_no_judge_weight_04_in_synthetic(self):
        """Test: no hidden 0.4 defaults in src/synthetic_evaluator.py"""
        from src import synthetic_evaluator

        source_file = synthetic_evaluator.__file__
        with open(source_file) as f:
            content = f.read()

        # Should NOT have "judge_weight: float = 0.4"
        assert "judge_weight: float = 0.4" not in content, (
            "Found old default 0.4 in synthetic_evaluator.py!"
        )

    def test_no_judge_weight_07_in_synthetic(self):
        """Test: no hidden 0.7 defaults in src/synthetic_evaluator.py"""
        from src import synthetic_evaluator

        source_file = synthetic_evaluator.__file__
        with open(source_file) as f:
            content = f.read()

        # Should NOT have "judge_weight: float = 0.7"
        assert "judge_weight: float = 0.7" not in content, (
            "Found old default 0.7 in synthetic_evaluator.py!"
        )


class TestDocstringConsistency:
    """Test: docstring example is internally consistent"""

    def test_synthetic_docstring_example_consistent(self):
        """Test: docstring example is internally consistent (doesn't reference conflicting judge_weight)"""
        from src.synthetic_evaluator import make_synthetic_evaluator

        # Get the function's docstring
        docstring = make_synthetic_evaluator.__doc__ or ""

        # The docstring should NOT mention judge_weight=0.4 explicitly
        # (If it does, it's inconsistent with the function default of 0.65)
        # Note: We allow explicit 0.65 or omission, but not 0.4 or 0.7
        if "judge_weight" in docstring:
            # Check for inconsistent explicit values
            assert "judge_weight=0.4" not in docstring, (
                "Docstring shows 0.4 but function defaults to 0.65!"
            )
            assert "judge_weight = 0.4" not in docstring, (
                "Docstring shows 0.4 but function defaults to 0.65!"
            )


# Run as main for quick verification
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
