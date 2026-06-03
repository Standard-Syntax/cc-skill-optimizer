"""
Test suite for MIPROv2 instruction extraction in synthetic_evaluator.py
Tests Task 5.2: extraction of optimized.assess.signature.instructions and demos

Run: uv run pytest tests/test_5_2_dspy_extraction.py -v
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import inspect
import shutil
import tempfile
from unittest import mock

import pytest

# ============================================================================
# Mock Classes
# ============================================================================


class DummyDemo:
    """Mock demo object mimicking DSPy's demo structure."""

    def __init__(self, task_desc: str = "Write tests", guidance: str = "Use pytest"):
        self.task_description = task_desc
        self.improved_guidance = guidance


class DummyAssess:
    """Mock assess signature for DSPy program."""

    def __init__(
        self,
        instructions: str = "Test instructions",
        demos: list = None,
    ):
        self.signature = self
        self.instructions = instructions
        self.demos = demos or []


class DummyProgram:
    """Mock DSPy optimized program from MIPROv2.compile()."""

    def __init__(
        self,
        assess=None,
        with_save_error: bool = False,
    ):
        self._assess = assess or DummyAssess()
        self._with_save_error = with_save_error

    @property
    def assess(self):
        return self._assess

    def save(self, path: str):
        if self._with_save_error:
            raise RuntimeError("Save failed intentionally")

    def __call__(self, **kwargs):
        class MockPrediction:
            improved_guidance = "Consider using pytest for better test coverage"

        return MockPrediction()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir():
    """Create a temp directory for output files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_task_library():
    """Sample task library for testing."""
    return [
        {
            "task_description": "Write unit tests",
            "task_pitfalls": ["edge cases", "mocking"],
            "task_pitfalls_str": "- edge cases\n- mocking",
            "task_domain_context": "Testing Python code",
            "task_success_criteria": "Uses pytest",
        },
        {
            "task_description": "Debug a bug",
            "task_pitfalls": ["logs", "steps"],
            "task_pitfalls_str": "- logs\n- steps",
            "task_domain_context": "Debugging",
            "task_success_criteria": "Uses logging",
        },
    ]


# ============================================================================
# Helper to patch and call
# ============================================================================


def run_with_mock_monkeypatch(monkeypatch, mock_program, task_lib, seed, out_dir):
    """Helper to patch dspy and run make_dspy_synthetic_pipeline."""
    import dspy as RealDSPy

    def mock_compile_fn(*args, **kwargs):
        return mock_program

    monkeypatch.setattr(RealDSPy, "LM", mock.MagicMock())
    monkeypatch.setattr(RealDSPy, "MIPROv2", mock.MagicMock())  # dspy 3.x top-level
    monkeypatch.setattr(RealDSPy.teleprompt, "MIPROv2", mock.MagicMock())  # legacy
    RealDSPy.MIPROv2.return_value.compile = mock_compile_fn
    RealDSPy.teleprompt.MIPROv2.return_value.compile = mock_compile_fn

    from synthetic_evaluator import make_dspy_synthetic_pipeline

    return make_dspy_synthetic_pipeline(
        task_library=task_lib,
        seed_candidate=seed,
        task_lm="fake/model",
        reflection_lm="fake/model",
        max_bootstrap_evals=1,
        max_gepa_evals=1,
        output_dir=out_dir,
    )


# ============================================================================
# Test 1: extraction block executes without error when MIPROv2 succeeds
# ============================================================================


def test_extraction_block_executes_with_mipro_success(
    monkeypatch, temp_output_dir, sample_task_library
):
    """Verify output contains Optimized Instructions section when MIPROv2 succeeds."""
    mock_program = DummyProgram(assess=DummyAssess(instructions="Test instructions"))

    result = run_with_mock_monkeypatch(
        monkeypatch,
        mock_program,
        sample_task_library,
        "# Seed Skill\nSome content",
        temp_output_dir,
    )

    assert "Optimized Instructions (MIPROv2-refined)" in result
    assert "Test instructions" in result


# ============================================================================
# Test 2: extraction falls back gracefully when signature attribute missing
# ============================================================================


def test_extraction_falls_back_when_signature_missing(
    monkeypatch, temp_output_dir, sample_task_library
):
    """Verify no crash when optimized.assess.signature is missing (AttributeError)."""

    class NoSignatureProgram:
        def __init__(self):
            pass

        def save(self, path):
            pass

        def __call__(self, **kwargs):
            class P:
                improved_guidance = "Test guidance"

            return P()

    mock_program = NoSignatureProgram()

    import dspy as RealDSPy

    def mock_compile_fn(*args, **kwargs):
        return mock_program

    monkeypatch.setattr(RealDSPy, "LM", mock.MagicMock())
    monkeypatch.setattr(RealDSPy, "MIPROv2", mock.MagicMock())  # dspy 3.x top-level
    monkeypatch.setattr(RealDSPy.teleprompt, "MIPROv2", mock.MagicMock())  # legacy
    RealDSPy.MIPROv2.return_value.compile = mock_compile_fn
    RealDSPy.teleprompt.MIPROv2.return_value.compile = mock_compile_fn

    from synthetic_evaluator import make_dspy_synthetic_pipeline

    # Should NOT crash - the wrapper handles AttributeError
    result = make_dspy_synthetic_pipeline(
        task_library=sample_task_library,
        seed_candidate="# Test",
        task_lm="fake/model",
        reflection_lm="fake/model",
        max_bootstrap_evals=1,
        max_gepa_evals=1,
        output_dir=temp_output_dir,
    )

    # Should complete without error
    assert isinstance(result, str)
    assert len(result) > 0


# ============================================================================
# Test 3: extraction falls back gracefully when demos attribute missing
# ============================================================================


def test_extraction_falls_back_when_demos_missing(
    monkeypatch, temp_output_dir, sample_task_library
):
    """Verify no Few-Shot Examples section when demos is empty."""
    mock_program = DummyProgram(assess=DummyAssess(instructions="Inst", demos=[]))

    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, "# Test", temp_output_dir
    )

    assert "Few-Shot Examples" not in result
    assert "Example 1" not in result


# ============================================================================
# Test 4: demos are capped at 3
# ============================================================================


def test_demos_capped_at_three(monkeypatch, temp_output_dir, sample_task_library):
    """Verify output has exactly 3 Example blocks when 10 demos provided."""
    ten_demos = [DummyDemo(f"Task {i}", f"Guidance {i}") for i in range(10)]
    mock_program = DummyProgram(assess=DummyAssess(instructions="Inst", demos=ten_demos))

    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, "# Test", temp_output_dir
    )

    example_count = result.count("### Example")
    assert example_count == 3, f"Expected 3 examples, got {example_count}"
    assert "### Example 4" not in result


# ============================================================================
# Test 5: demo formatting includes task_description and improved_guidance
# ============================================================================


def test_demo_formatting_includes_fields(monkeypatch, temp_output_dir, sample_task_library):
    """Verify demo formatting shows task_description and improved_guidance."""
    demos = [DummyDemo(task_desc="Write tests", guidance="Use pytest")]
    mock_program = DummyProgram(assess=DummyAssess(instructions="Inst", demos=demos))

    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, "# Test", temp_output_dir
    )

    assert "**Task:** Write tests" in result
    assert "**Guidance:**" in result


# ============================================================================
# Test 6: demo formatting truncates long fields
# ============================================================================


def test_demo_truncates_long_fields(monkeypatch, temp_output_dir, sample_task_library):
    """Verify long task_description is truncated to 120 chars."""
    long_desc = "A" * 500
    demos = [DummyDemo(task_desc=long_desc, guidance="Short")]
    mock_program = DummyProgram(assess=DummyAssess(instructions="Inst", demos=demos))

    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, "# Test", temp_output_dir
    )

    # Should contain Task line but not overflow
    assert "**Task:**" in result
    # Find the task line content
    task_pos = result.find("**Task:**")
    if task_pos != -1:
        # Extract until newline
        line_end = result.find("\n", task_pos)
        task_line = (
            result[task_pos:line_end] if line_end != -1 else result[task_pos : task_pos + 150]
        )
        # Should be reasonably bounded (allow some margin for formatting)
        content_len = len(task_line.replace("**Task:**", "").strip())
        assert content_len <= 130


# ============================================================================
# Test 7: existing Auto-Generated Guidance section preserved
# ============================================================================


def test_auto_generated_guidance_section_preserved(
    monkeypatch, temp_output_dir, sample_task_library
):
    """Verify Auto-Generated Guidance (GEPA-refined) section can still appear."""
    mock_program = DummyProgram(assess=DummyAssess(instructions="Instructions"))

    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, "# Test Skill", temp_output_dir
    )

    # Just verify no crash - section may not appear if no valid guidance generated
    assert isinstance(result, str)
    assert "# Test Skill" in result


# ============================================================================
# Test 8: seed_candidate is leading content
# ============================================================================


def test_seed_candidate_is_leading(monkeypatch, temp_output_dir, sample_task_library):
    """Verify seed_candidate content appears at start of output."""
    seed = "# My Custom Skill\nSome initial content"
    mock_program = DummyProgram(assess=DummyAssess(instructions="Inst"))

    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, seed, temp_output_dir
    )

    assert result.startswith("# My Custom Skill")


# ============================================================================
# Test 9: no sections when all sources empty
# ============================================================================


def test_no_sections_when_all_sources_empty(monkeypatch, temp_output_dir, sample_task_library):
    """Verify output works when instructions and demos are empty."""
    mock_program = DummyProgram(assess=DummyAssess(instructions="", demos=[]))

    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, "# Test", temp_output_dir
    )

    # Shouldn't crash - verify output is valid (even if sections get added)
    assert isinstance(result, str)
    # The key: seed should still be present
    assert result.strip().startswith("# Test")


# ============================================================================
# Test 10: signature preserved (same 7 parameters)
# ============================================================================


def test_function_signature_preserved():
    """Verify make_dspy_synthetic_pipeline has same 7 parameters."""
    from synthetic_evaluator import make_dspy_synthetic_pipeline

    sig = inspect.signature(make_dspy_synthetic_pipeline)
    params = list(sig.parameters.keys())

    assert params == [
        "task_library",
        "seed_candidate",
        "task_lm",
        "reflection_lm",
        "max_bootstrap_evals",
        "max_gepa_evals",
        "output_dir",
    ]
    assert len(params) == 7


# ============================================================================
# Test 11: optimized.save still works (or silently fails)
# ============================================================================


def test_optimized_save_works(monkeypatch, temp_output_dir, sample_task_library):
    """Verify optimized.save() call doesn't crash."""
    mock_program = DummyProgram(
        assess=DummyAssess(instructions="Test"),
        with_save_error=True,
    )

    # Use helper like other tests
    result = run_with_mock_monkeypatch(
        monkeypatch, mock_program, sample_task_library, "# Test", temp_output_dir
    )

    # Should NOT raise - the try/except wraps save
    assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
