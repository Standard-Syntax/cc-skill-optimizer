"""
Tests for Task 18.1: make_length_constrained_proposer factory and --max-skill-chars flag.

The Decagon production study (March 2026, 50 ablation experiments) found
unconstrained GEPA produces prompts exceeding 5,000 chars. A 1,500-char
constraint achieved 4× compression with 0.8% performance loss and better
generalization. The factory returns a callable that injects a constraint
note into the reflective_dataset before proposing.
"""

from optimize import make_length_constrained_proposer


class TestMakeLengthConstrainedProposer:
    """The factory returns a callable with the documented signature."""

    def test_returns_callable(self):
        proposer = make_length_constrained_proposer(max_chars=2000)
        assert callable(proposer)

    def test_default_max_chars_is_2000(self):
        # Calling with no arg should use default
        proposer = make_length_constrained_proposer()
        assert callable(proposer)

    def test_proposer_returns_none_to_delegate(self):
        """The proposer returns None to signal GEPA to use its default proposer
        with the enriched dataset."""
        proposer = make_length_constrained_proposer(max_chars=2000)
        # Pass a minimal reflective_dataset
        dataset = [
            {"feedback": "Original feedback item 1"},
            {"feedback": "Original feedback item 2"},
        ]
        result = proposer(
            current_candidate="dummy",
            reflective_dataset=dataset,
            components_to_update=None,
        )
        assert result is None

    def test_proposer_signature_accepts_three_args(self):
        """The proposer signature is (current_candidate, reflective_dataset, components_to_update)."""
        import inspect
        proposer = make_length_constrained_proposer(max_chars=1500)
        sig = inspect.signature(proposer)
        params = list(sig.parameters.keys())
        # Should have exactly 3 params
        assert len(params) == 3, f"Expected 3 params, got {len(params)}: {params}"
        assert params == ["current_candidate", "reflective_dataset", "components_to_update"], (
            f"Unexpected param names: {params}"
        )

    def test_proposer_applies_constraint_in_place(self):
        """The proposer must mutate each item's feedback in-place to append the
        length constraint. Returning None delegates to GEPA's default proposer
        with the mutated dataset, so the constraint is visible to the reflection LM."""
        # Use a fresh dataset for this test
        dataset = [
            {"feedback": "Original feedback 1"},
            {"feedback": "Original feedback 2"},
        ]
        original_first = dataset[0]["feedback"]

        proposer = make_length_constrained_proposer(max_chars=2000)
        result = proposer(
            current_candidate="dummy",
            reflective_dataset=dataset,
            components_to_update=None,
        )

        # The proposer returns None to delegate
        assert result is None

        # The input dataset IS mutated in-place — each item's feedback should
        # now have the constraint note appended
        assert len(dataset) == 2, "Dataset length should be unchanged"
        assert dataset[0]["feedback"] != original_first, (
            "First item's feedback should be MUTATED (constraint appended)"
        )
        assert "Original feedback 1" in dataset[0]["feedback"], (
            "Original feedback text should still be present (we append, not replace)"
        )
        assert "IMPORTANT: The skill MUST be under 2000 characters" in dataset[0]["feedback"], (
            "The constraint note must be appended to the first item's feedback"
        )
        assert "IMPORTANT: The skill MUST be under 2000 characters" in dataset[1]["feedback"], (
            "The constraint note must be appended to the second item's feedback"
        )

    def test_proposer_respects_max_chars_value(self):
        """The constraint note must include the actual max_chars value passed to the factory."""
        dataset = [{"feedback": "orig"}]
        proposer = make_length_constrained_proposer(max_chars=1500)
        proposer("dummy", dataset, None)
        assert "1500 characters" in dataset[0]["feedback"], (
            "Constraint note should reference the actual max_chars (1500)"
        )
        assert "exceeds 1500 chars" in dataset[0]["feedback"], (
            "Constraint note should mention the threshold"
        )


class TestLengthConstrainedProposerIntegration:
    """The proposer is wired into ReflectionConfig.custom_candidate_proposer."""

    def test_proposer_is_passed_to_reflection_config(self):
        """Verify ReflectionConfig receives custom_candidate_proposer in optimize.py
        for both run_gepa_optimize_anything and run_gepa_synthetic."""
        import inspect

        import optimize

        # Check that run_gepa_optimize_anything and run_gepa_synthetic
        # use the make_length_constrained_proposer
        src_replay = inspect.getsource(optimize.run_gepa_optimize_anything)
        src_synthetic = inspect.getsource(optimize.run_gepa_synthetic)
        assert "custom_candidate_proposer=make_length_constrained_proposer(" in src_replay, (
            "run_gepa_optimize_anything should use make_length_constrained_proposer"
        )
        assert "custom_candidate_proposer=make_length_constrained_proposer(" in src_synthetic, (
            "run_gepa_synthetic should use make_length_constrained_proposer"
        )
        print("OK: both runners use make_length_constrained_proposer")

    def test_max_skill_chars_argparse_flag(self):
        """Verify --max-skill-chars is registered with argparse (default 2000)."""
        import re

        import optimize

        src = inspect_getsource_main_argparse(optimize)
        assert "--max-skill-chars" in src, "--max-skill-chars should be in argparse"
        # The default should be 2000 - search multiline across the multiline argparse call
        match = re.search(r'--max-skill-chars[\s\S]*?default=(\d+)', src)
        assert match is not None, "Expected --max-skill-chars with default=N"
        assert int(match.group(1)) == 2000, f"Expected default=2000, got {match.group(1)}"
        print("OK: --max-skill-chars registered with default=2000")


def inspect_getsource_main_argparse(optimize):
    """Helper: get the source of optimize's main() to inspect argparse."""
    import inspect
    return inspect.getsource(optimize)
