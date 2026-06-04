"""
Tests for Task 16.3: score all components in multi/nested evaluators.

Phase 16.3 changes:
- make_multi_evaluator: score the full concatenated candidate (was: primary only)
- make_nested_evaluator: score the full concatenated candidate (was: root_key only)
- Both: populate side_info["scores"] with per-component scores for
  multi-objective Pareto via frontier_type="hybrid"
"""

from pathlib import Path

from optimize import make_multi_evaluator, make_nested_evaluator


def make_deterministic_base_evaluator():
    """Return a base_evaluator that scores candidates by their LENGTH in chars.

    This gives us a deterministic, LLM-free evaluator that we can use to verify
    the new combined_text and per-component scoring behavior. Score is normalized
    to [0, 1] by dividing by 2000 (a 2000-char skill scores 1.0).
    """
    def base_evaluator(candidate, example):
        text_len = len(candidate) if isinstance(candidate, str) else 0
        # Length-based score: longer is "better" up to 2000 chars
        score = min(1.0, text_len / 2000.0)
        side_info = {
            "score": score,
            "text_length": text_len,
            "example": example,
        }
        return score, side_info
    return base_evaluator


class TestMakeMultiEvaluator:
    """make_multi_evaluator: score the full candidate, not just the primary component."""

    def _make_evaluator(self):
        return make_multi_evaluator(
            base_evaluator=make_deterministic_base_evaluator(),
            skill_dir=Path("/tmp"),
        )

    def test_two_component_candidate_returns_components_metadata(self):
        """A 2-component dict returns side_info['components'] with both keys and lengths."""
        evaluate = self._make_evaluator()
        candidate = {
            "skill_md": "A" * 500,
            "claude_md": "B" * 800,
        }
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)

        # side_info['components'] must have both keys
        assert "components" in side_info
        assert "skill_md" in side_info["components"]
        assert "claude_md" in side_info["components"]
        assert side_info["components"]["skill_md"] == 500
        assert side_info["components"]["claude_md"] == 800
        # n_components is set
        assert side_info["n_components"] == 2

    def test_two_component_candidate_returns_per_component_scores(self):
        """side_info['scores'] is populated with per-component scores for Pareto tracking."""
        evaluate = self._make_evaluator()
        candidate = {
            "skill_md": "A" * 500,  # 500 / 2000 = 0.25
            "claude_md": "B" * 800,  # 800 / 2000 = 0.40
        }
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)

        # side_info['scores'] must have both keys
        assert "scores" in side_info
        assert "skill_md" in side_info["scores"]
        assert "claude_md" in side_info["scores"]
        # The deterministic base_evaluator scores by length/2000
        assert abs(side_info["scores"]["skill_md"] - 0.25) < 0.01
        assert abs(side_info["scores"]["claude_md"] - 0.40) < 0.01

    def test_combined_score_reflects_full_candidate_not_just_primary(self):
        """The returned score reflects the full concatenated candidate (1300 chars),
        not just the primary 'skill_md' (500 chars). 1300/2000 = 0.65 (combined) vs
        500/2000 = 0.25 (primary only)."""
        evaluate = self._make_evaluator()
        candidate = {
            "skill_md": "A" * 500,
            "claude_md": "B" * 800,
        }
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)

        # Combined length: 500 + 800 + overhead for "# key\n" prefixes
        # The base_evaluator sees ~1320 chars → score ~0.66
        assert score > 0.5, (
            f"Expected score > 0.5 (combined length should be > 1000 chars), got {score}. "
            f"This indicates the score is still using only the primary component."
        )

    def test_single_component_candidate_works(self):
        """A 1-component candidate still scores correctly."""
        evaluate = self._make_evaluator()
        candidate = {"skill_md": "X" * 1000}
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)
        # 1000/2000 = 0.5
        assert abs(score - 0.5) < 0.01
        assert side_info["n_components"] == 1
        assert "skill_md" in side_info["scores"]

    def test_empty_candidate_returns_zero_score(self):
        """An empty candidate returns score 0 and an empty components dict."""
        evaluate = self._make_evaluator()
        candidate = {}
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)
        assert score == 0.0
        assert side_info["n_components"] == 0
        assert side_info["components"] == {}
        assert side_info["scores"] == {}


class TestMakeNestedEvaluator:
    """make_nested_evaluator: score the full nested candidate, not just the root_key file."""

    def _make_evaluator(self, root_key="CLAUDE.md"):
        return make_nested_evaluator(
            base_evaluator=make_deterministic_base_evaluator(),
            root_key=root_key,
        )

    def test_nested_candidate_returns_nested_files_metadata(self):
        """A 3-file nested candidate returns side_info['nested_files'] with all 3 files."""
        evaluate = self._make_evaluator()
        candidate = {
            "CLAUDE.md": "X" * 200,
            "src/CLAUDE.md": "Y" * 400,
            "tests/CLAUDE.md": "Z" * 300,
        }
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)

        # side_info['nested_files'] must have all 3 keys
        assert "nested_files" in side_info
        assert side_info["nested_files"]["CLAUDE.md"] == 200
        assert side_info["nested_files"]["src/CLAUDE.md"] == 400
        assert side_info["nested_files"]["tests/CLAUDE.md"] == 300
        assert side_info["n_nested_files"] == 3
        assert set(side_info["nested_file_keys"]) == set(candidate.keys())

    def test_nested_candidate_returns_per_file_scores(self):
        """side_info['scores'] is populated with per-file scores for Pareto tracking."""
        evaluate = self._make_evaluator()
        candidate = {
            "CLAUDE.md": "X" * 200,      # 0.1
            "src/CLAUDE.md": "Y" * 600,  # 0.3
            "tests/CLAUDE.md": "Z" * 1000, # 0.5
        }
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)

        # side_info['scores'] must have all 3 keys with deterministic scores
        assert "scores" in side_info
        assert abs(side_info["scores"]["CLAUDE.md"] - 0.1) < 0.01
        assert abs(side_info["scores"]["src/CLAUDE.md"] - 0.3) < 0.01
        assert abs(side_info["scores"]["tests/CLAUDE.md"] - 0.5) < 0.01

    def test_nested_combined_score_reflects_all_files(self):
        """The returned score reflects the full concatenated nested candidate, not just root_key."""
        evaluate = self._make_evaluator()
        candidate = {
            "CLAUDE.md": "X" * 200,  # root file — small
            "src/CLAUDE.md": "Y" * 1500,  # larger file
        }
        example = {"task_description": "test"}
        score, side_info = evaluate(candidate, example)

        # Combined length: 200 + 1500 + overhead = ~1720 chars → score ~0.86
        # Root-only (legacy): 200 chars → score 0.1
        assert score > 0.5, (
            f"Expected score > 0.5 (combined ~1720 chars), got {score}. "
            f"This indicates the score is still using only the root_key file."
        )

    def test_nested_uses_root_key_deterministically(self):
        """The root_key is used as a hint for the primary entry point in the legacy
        fallback — but the new code uses ALL files. The combined score should be
        the same regardless of which key is the root_key (since all keys are scored)."""
        evaluate_a = self._make_evaluator(root_key="CLAUDE.md")
        evaluate_b = self._make_evaluator(root_key="src/CLAUDE.md")
        candidate = {
            "CLAUDE.md": "X" * 200,
            "src/CLAUDE.md": "Y" * 400,
        }
        example = {"task_description": "test"}
        score_a, _ = evaluate_a(candidate, example)
        score_b, _ = evaluate_b(candidate, example)
        # Both should produce the same combined-text score
        assert abs(score_a - score_b) < 0.01, (
            f"Expected same combined score regardless of root_key, "
            f"got {score_a} vs {score_b}"
        )


class TestCombinedTextContents:
    """The base_evaluator is called with combined_text (all components joined with \\n\\n)."""

    def test_base_evaluator_receives_combined_text(self):
        """Verify the base_evaluator's text_length side_info reflects the combined length."""
        # Custom base_evaluator that records the candidate it received
        received_candidates = []

        def base_evaluator(candidate, example):
            received_candidates.append(candidate)
            return 0.5, {"text": candidate}

        evaluate = make_multi_evaluator(
            base_evaluator=base_evaluator,
            skill_dir=Path("/tmp"),
        )
        candidate = {
            "skill_md": "A" * 100,
            "claude_md": "B" * 200,
        }
        # Evaluate the candidate
        evaluate(candidate, {"task_description": "test"})

        # The base_evaluator was called multiple times:
        # - Once with combined_text (the primary score)
        # - N times with individual component texts (for side_info["scores"])
        # The FIRST call should be the combined text
        first_call = received_candidates[0]
        assert "skill_md" in first_call, "Combined text should include 'skill_md' header"
        assert "claude_md" in first_call, "Combined text should include 'claude_md' header"
        # Length should be ~300 + overhead for "# key\n" prefixes
        assert len(first_call) > 300, f"Combined text should be > 300 chars, got {len(first_call)}"

    def test_per_component_calls_have_individual_texts(self):
        """Verify that the per-component scoring calls receive individual component texts."""
        received_candidates = []

        def base_evaluator(candidate, example):
            received_candidates.append(candidate)
            return 0.5, {"text": candidate}

        evaluate = make_multi_evaluator(
            base_evaluator=base_evaluator,
            skill_dir=Path("/tmp"),
        )
        candidate = {
            "skill_md": "A" * 100,
            "claude_md": "B" * 200,
        }
        evaluate(candidate, {"task_description": "test"})

        # We expect 3 calls: 1 combined + 2 individual
        assert len(received_candidates) >= 3, (
            f"Expected at least 3 base_evaluator calls (1 combined + 2 per-component), "
            f"got {len(received_candidates)}"
        )
        # The 2nd and 3rd calls should be the individual components
        individual_texts = set(received_candidates[1:])
        # The individual texts should be exactly the original component values
        assert "A" * 100 in individual_texts
        assert "B" * 200 in individual_texts


class TestCombinedScoringErrorHandling:
    """Phase 16.3 round 2: combined call's exception is contained."""

    def test_multi_combined_call_failure_returns_zero_score(self):
        """When base_evaluator raises on the combined call, the evaluator returns
        score=0.0 with an error in side_info, NOT a hard crash."""
        from optimize import make_multi_evaluator

        def base(c, ex):
            # Combined text contains "BAD" — fail on it; pass on per-component texts
            if "BAD" in c:
                raise ValueError("simulated base_evaluator failure on combined text")
            return 0.7, {}

        evaluate = make_multi_evaluator(base_evaluator=base, skill_dir=Path("/tmp"))
        candidate = {"skill_md": "OK" * 100, "claude_md": "BAD" * 50}
        example = {"task_description": "test"}

        # Should NOT raise — the try/except catches the combined-call failure
        score, side_info = evaluate(candidate, example)

        assert score == 0.0, f"Expected 0.0 fallback on combined-call failure, got {score}"
        assert "error" in side_info, f"Expected 'error' key in side_info, got keys: {list(side_info.keys())}"
        assert "simulated base_evaluator failure" in side_info["error"]
        # Components metadata must still be populated
        # "OK" * 100 = 200 chars (OK is 2 chars)
        assert side_info["components"]["skill_md"] == 200
        assert side_info["n_components"] == 2
        # Per-component scoring should still happen
        assert "scores" in side_info
        assert side_info["scores"]["skill_md"] == 0.7  # 'OK' is not 'BAD', base succeeds
        assert side_info["scores"]["claude_md"] == 0.0  # per-component try/except catches the same error

    def test_nested_combined_call_failure_returns_zero_score(self):
        """When base_evaluator raises on the combined call in make_nested_evaluator,
        score=0.0 fallback, NOT a crash."""
        from optimize import make_nested_evaluator

        def base(c, ex):
            if "BAD" in c:
                raise ValueError("simulated base_evaluator failure")
            return 0.6, {}

        evaluate = make_nested_evaluator(base_evaluator=base, root_key="CLAUDE.md")
        candidate = {"CLAUDE.md": "OK" * 100, "src/CLAUDE.md": "BAD" * 50}
        example = {"task_description": "test"}

        score, side_info = evaluate(candidate, example)

        assert score == 0.0
        assert "error" in side_info
        assert "simulated base_evaluator failure" in side_info["error"]
        # Nested files metadata should still be set
        # "OK" * 100 = 200 chars (OK is 2 chars)
        assert "nested_files" in side_info
        assert side_info["nested_files"]["CLAUDE.md"] == 200
        assert side_info["n_nested_files"] == 2
        # Per-file scoring
        assert side_info["scores"]["CLAUDE.md"] == 0.6
        assert side_info["scores"]["src/CLAUDE.md"] == 0.0
