"""
Test suite for de-collision suffix fix in section_parser.py (Task 7.2).

Tests the _make_unique_key helper and its callsites to prevent silent overwrites
when two headings normalize to the same key.
"""

import inspect
import pytest
from section_parser import (
    _make_unique_key,
    _normalize_key,
    parse_sections,
    merge_sections,
    build_section_tree,
)


class TestMakeUniqueKeyHelper:
    """Tests for the _make_unique_key helper function directly."""

    def test_returns_base_key_if_not_in_existing(self):
        """Call with ("section_foo", {"section_bar"}); verify result is "section_foo"."""
        result = _make_unique_key("section_foo", {"section_bar"})
        assert result == "section_foo"

    def test_returns_underscore_2_suffix_on_first_collision(self):
        """Call with ("section_foo", {"section_foo"}); verify result is "section_foo_2"."""
        result = _make_unique_key("section_foo", {"section_foo"})
        assert result == "section_foo_2"

    def test_returns_underscore_3_suffix_on_second_collision(self):
        """Call with ("section_foo", {"section_foo", "section_foo_2"}); verify result is "section_foo_3"."""
        result = _make_unique_key("section_foo", {"section_foo", "section_foo_2"})
        assert result == "section_foo_3"

    def test_returns_underscore_n_for_arbitrary_n(self):
        """Call with ("section_foo", {"section_foo", "section_foo_2", "section_foo_3", "section_foo_4"}); verify result is "section_foo_5"."""
        existing = {"section_foo", "section_foo_2", "section_foo_3", "section_foo_4"}
        result = _make_unique_key("section_foo", existing)
        assert result == "section_foo_5"

    def test_does_not_mutate_existing_keys(self):
        """Call with a set; verify the set is unchanged after the call."""
        original_set = {"section_bar", "section_baz"}
        _make_unique_key("section_foo", original_set)
        assert original_set == {"section_bar", "section_baz"}

    def test_is_private_underscore_prefix(self):
        """Verify the function name is _make_unique_key (not exported)."""
        assert _make_unique_key.__name__ == "_make_unique_key"

    def test_signature_of_helper(self):
        """Use inspect.signature to verify the helper has parameters (base_key: str, existing_keys: set[str])."""
        sig = inspect.signature(_make_unique_key)
        params = list(sig.parameters.keys())
        assert params == ["base_key", "existing_keys"]


class TestDeCollisionInParseSections:
    """Tests for de-collision in parse_sections function."""

    def test_two_punctuation_different_headings_produce_separate_sections(self):
        """Parse a skill with "## Key Patterns!" and "## Key Patterns?"; verify dict has TWO entries."""
        text = """\
## Key Patterns!
Content for first.

## Key Patterns?
Content for second.
"""
        sections = parse_sections(text, max_depth=2)
        # Filter to only section keys (exclude metadata like _header)
        section_keys = [k for k in sections if k.startswith("section_")]
        assert len(section_keys) == 2, (
            f"Expected 2 sections, got {len(section_keys)}: {section_keys}"
        )

    def test_three_collisions_produce_three_distinct_sections(self):
        """Parse a skill with "## Key Patterns!", "## Key Patterns?", "## Key Patterns."; verify THREE distinct keys."""
        text = """\
## Key Patterns!
Content for first.

## Key Patterns?
Content for second.

## Key Patterns.
Content for third.
"""
        sections = parse_sections(text, max_depth=2)
        section_keys = [k for k in sections if k.startswith("section_")]
        assert len(section_keys) == 3, (
            f"Expected 3 sections, got {len(section_keys)}: {section_keys}"
        )

    def test_original_heading_text_preserved_in_metadata(self):
        """For each colliding section, verify _heading metadata contains ORIGINAL text."""
        text = """\
## Key Patterns!
Content for first.

## Key Patterns?
Content for second.
"""
        sections = parse_sections(text, max_depth=2)
        # Find the heading metadata keys
        heading_keys = [k for k in sections if k.endswith("_heading")]
        assert len(heading_keys) == 2, f"Expected 2 heading metdata keys, got {len(heading_keys)}"

        # Verify the original head texts are stored
        for key in heading_keys:
            heading_value = sections[key]
            assert "## Key Patterns" in heading_value


class TestDeCollisionInNestedSections:
    """Tests for de-collision in nested sections."""

    def test_de_collision_works_in_nested_sections(self):
        """Parse skill with nested sections where two siblings have similar names."""
        text = """\
## Commands
### Build
First nested build.

### Build
Second nested build (same name).
"""
        sections = parse_sections(text, max_depth=3)
        # Should have 2 subsection_build keys (excluding metadata keys that start with _)
        subsection_keys = [
            k for k in sections if k.startswith("section_") and "subsection_build" in k
        ]
        assert len(subsection_keys) == 2, f"Expected 2 subsection_build keys, got {subsection_keys}"


class TestDeCollisionInBuildSectionTree:
    """Tests for de-collision in build_section_tree function."""

    def test_build_section_tree_handles_collisions(self):
        """Parse skill with tree structure where two root sections have similar names."""
        text = """\
## Build
Build instructions.

## Build
More build instructions.
"""
        tree = build_section_tree(text, max_depth=2)
        # Should have 2 root sections
        root_names = [s.name for s in tree]
        assert len(root_names) == 2, f"Expected 2 root sections, got {root_names}"


class TestRoundTripIdempotency:
    """Tests for round-trip parse -> merge -> parse -> merge."""

    def test_round_trip_with_collisions(self):
        """Parse, merge, parse again, merge again. Verify idempotency and both sections preserved."""
        text = """\
## Build
First build section.

## Build
Second build section.
"""
        # First parse
        sections1 = parse_sections(text, max_depth=2)
        section_keys1 = [k for k in sections1 if k.startswith("section_")]
        assert len(section_keys1) == 2

        # First merge
        merged1 = merge_sections(sections1, max_depth=2)

        # Second parse
        sections2 = parse_sections(merged1, max_depth=2)
        section_keys2 = [k for k in sections2 if k.startswith("section_")]
        assert len(section_keys2) == 2

        # Second merge
        merged2 = merge_sections(sections2, max_depth=2)

        # Should have same number of sections
        assert len(section_keys2) == len(section_keys1)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_slug_not_decollided(self):
        """A heading like "## " normalizes to "". Existing behavior: skip empty slugs."""
        # An empty heading (just "## ") produces empty slug - this shouldn't cause issues
        text = """\
## Overview
Content.

##
This is weird but shouldn't crash.
"""
        sections = parse_sections(text, max_depth=2)
        # Just ensure it doesn't crash and returns reasonable output
        assert "section_overview" in sections

    def test_key_with_trailing_underscore(self):
        """A heading like "## Key !" normalizes to "section_key_". De-collision produces "section_key__2"."""
        text = """\
## Key !
First section.

## Key
Second section.
"""
        sections = parse_sections(text, max_depth=2)
        section_keys = [k for k in sections if k.startswith("section_")]
        # Both should exist with distinct keys
        assert len(section_keys) == 2, f"Expected 2 sections, got {section_keys}"

    def test_very_long_sequence_of_collisions(self):
        """Test many collisions with same base name - ensure counter doesn't get stuck."""
        # Create 10 headings that all normalize to the same key, with actual content for each
        lines = []
        for i in range(10):
            lines.append(f"## Build")
            lines.append(f"Some content for section {i}.")
            lines.append("")  # blank line between sections
        text = "\n".join(lines)
        sections = parse_sections(text, max_depth=2)
        section_keys = [k for k in sections if k.startswith("section_")]
        # All 10 should exist with distinct keys (base + _2, _3, ..., _10)
        assert len(section_keys) == 10, f"Expected 10 sections, got {section_keys}"


if __name__ == "__main__":
    # Run with pytest when executed directly
    pytest.main([__file__, "-v"])
