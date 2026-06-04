"""
Tests for Task 18.5: tune structural_score() length bounds and specificity density.

Phase 18.5 changes:
- Length sweet spot: 800-2500 -> 600-1800 (Decagon optimum at 1,500 chars)
- Length bloat penalty: 0.04 -> 0.02 (was too lenient)
- Specificity: raw count * 0.015 -> density * 0.05 (hits per 100 words, max 0.20)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path (same pattern as test_evaluator_fixes.py)
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))  # noqa: E402 — test path setup, must precede module import


class TestLengthBounds:
    """structural_score length scoring with the new tighter bounds."""

    def _score(self, n: int) -> float:
        from src.synthetic_evaluator import structural_score

        # Build a skill of exactly n chars using a simple padding pattern
        # Add a section header when n >= 50 so the candidate is realistic (1 section = 0.10)
        candidate = "x" * n if n < 50 else "## Overview\n" + "x" * (n - 12)
        score, breakdown = structural_score(candidate)
        return breakdown["length"]

    def test_sweet_spot_1200_chars(self):
        """1200 chars is in the new sweet spot [600, 1800] -> 0.15"""
        score = self._score(1200)
        assert score == 0.15, f"Expected 0.15 for 1200 chars, got {score}"

    def test_sweet_spot_600_chars(self):
        """600 chars (lower bound of sweet spot) -> 0.15"""
        score = self._score(600)
        assert score == 0.15, f"Expected 0.15 for 600 chars, got {score}"

    def test_sweet_spot_1800_chars(self):
        """1800 chars (upper bound of sweet spot) -> 0.15"""
        score = self._score(1800)
        assert score == 0.15, f"Expected 0.15 for 1800 chars, got {score}"

    def test_acceptable_range_500_chars(self):
        """500 chars (in [400, 600)) -> 0.08"""
        score = self._score(500)
        assert score == 0.08, f"Expected 0.08 for 500 chars, got {score}"

    def test_acceptable_range_2200_chars(self):
        """2200 chars (in (1800, 3000]) -> 0.08"""
        score = self._score(2200)
        assert score == 0.08, f"Expected 0.08 for 2200 chars, got {score}"

    def test_too_short_100_chars(self):
        """100 chars (in [0, 200)) -> 0.0 (too short)"""
        score = self._score(100)
        assert score == 0.0, f"Expected 0.0 for 100 chars, got {score}"

    def test_bloat_penalty_4000_chars(self):
        """4000 chars (>3000) -> 0.02 (stronger bloat penalty than the old 0.04)"""
        score = self._score(4000)
        assert score == 0.02, f"Expected 0.02 for 4000 chars, got {score}"

    def test_bloat_penalty_6000_chars(self):
        """6000 chars (>3000) -> 0.02"""
        score = self._score(6000)
        assert score == 0.02, f"Expected 0.02 for 6000 chars, got {score}"


class TestSpecificityDensity:
    """structural_score specificity scoring with the new density formula."""

    def test_dense_short_skill_scores_higher_than_sparse_long_skill(self):
        """The Decagon study's key finding: a 1,200-char skill with 10 specificity
        hits scores higher than a 4,000-char skill with 12 hits (density wins)."""
        from src.synthetic_evaluator import structural_score

        # Short dense skill: ~1200 chars, 5 inline-code refs spread over ~1200 chars
        short_dense = (
            "Run `uv sync` to install deps. "
            "Use `ruff check` for linting. "
            "Run `pytest` for tests. "
            "Use `mypy` for types. "
            "Use `ty` for fast type checking. "  # 5 inline refs
            + "x" * 1000
        )  # Total ~1100 chars, 5 inline code refs

        # Long sparse skill: 4000 chars, 12 inline-code refs
        long_sparse = (
            "`uv run` " * 12
            + "x" * (4000 - 12 * 8)
        )  # Total ~4000 chars, 12 inline code refs (but most are noise)

        short_score, short_breakdown = structural_score(short_dense)
        long_score, long_breakdown = structural_score(long_sparse)

        # Short dense (5 hits / ~200 words = 2.5 per 100 words = 0.125 score)
        # Long sparse (12 hits / ~800 words = 1.5 per 100 words = 0.075 score)
        # Short dense should score HIGHER (or at least equal)
        assert short_breakdown["specificity"] >= long_breakdown["specificity"], (
            f"Short dense ({short_breakdown['specificity']:.3f}) should score "
            f">= long sparse ({long_breakdown['specificity']:.3f})"
        )

    def test_density_caps_at_0_20(self):
        """Very high density caps at 0.20."""
        from src.synthetic_evaluator import structural_score

        # Build a skill with absurdly high density: 50 inline-code refs in ~200 chars
        high_density = "`uv` " * 50  # 50 hits in ~200 chars
        score, breakdown = structural_score(high_density)
        assert breakdown["specificity"] <= 0.20, (
            f"Specificity score must cap at 0.20, got {breakdown['specificity']}"
        )

    def test_zero_hits_gives_zero_score(self):
        """A skill with no specificity patterns scores 0 on specificity."""
        from src.synthetic_evaluator import structural_score

        # Plain text with no inline code, no tool names, no file paths
        plain = "This is just plain English text with no code references at all. " * 20
        score, breakdown = structural_score(plain)
        assert breakdown["specificity"] == 0.0, (
            f"Expected 0.0 specificity for plain text, got {breakdown['specificity']}"
        )


class TestStructuralScoreBounds:
    """structural_score total score is in [0, 1] for boundary cases."""

    def test_empty_skill(self):
        from src.synthetic_evaluator import structural_score

        score, breakdown = structural_score("")
        assert 0.0 <= score <= 1.0, f"Empty skill score out of bounds: {score}"
        assert breakdown["length"] == 0.0  # empty is < 200 chars

    def test_huge_skill(self):
        from src.synthetic_evaluator import structural_score

        # 10,000 char skill
        huge = "## Overview\n" + "x" * 9988
        score, breakdown = structural_score(huge)
        assert 0.0 <= score <= 1.0, f"Huge skill score out of bounds: {score}"
        assert breakdown["length"] == 0.02  # > 3000 chars

    def test_sweet_spot_skill_typical(self):
        """A typical 1500-char skill with sections + lists + inline code should
        score well (in the upper range)."""
        from src.synthetic_evaluator import structural_score

        typical = """\
## Overview
This is a typical skill for the project. It explains how to use `uv` for
dependency management and `ruff` for linting.

## Commands
1. Run `uv sync` to install dependencies.
2. Run `ruff check .` to lint.
3. Run `pytest` to run tests.

## Pitfalls
- Don't use `pip install`; use `uv` instead.
- Be careful with relative imports.
- The `pyproject.toml` has the dependencies.

## Conventions
- Follow PEP 8.
- Use `pathlib.Path` over `os.path`.
- Type hints required.

## Workflow
1. Read the spec.
2. Make a plan.
3. Implement.
4. Test.
5. Review.
"""
        # ~900 chars (sweet spot)
        score, breakdown = structural_score(typical)
        assert 0.0 <= score <= 1.0
        # All 5 sections present
        for section in ("overview", "commands", "pitfalls", "conventions", "workflow"):
            section_score = breakdown.get(f"section_{section}")
            assert section_score is not None and section_score > 0, (
                f"Section {section} should be present with a positive score, got {section_score}"
            )
