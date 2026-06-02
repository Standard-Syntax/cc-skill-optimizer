"""
Test idempotency fix for section_parser.py

Verifies:
1. Round-tripping a document through parse → merge → parse → merge produces the same output
2. Section content does NOT contain the heading line as its first line
3. _header section is NOT affected (it still retains level-1 headings correctly)
"""

import pytest
import re
import sys

sys.path.insert(0, "src")
from section_parser import parse_sections, merge_sections


class TestIdempotency:
    """Test that round-tripping is idempotent."""

    def test_round_trip_produces_same_output(self):
        """parse → merge → parse → merge should produce identical output."""
        original = """\
# My Project Skill

## Overview
This is the overview section with helpful context.

## Commands
Run build and test commands here.

### Build
Build steps for the project.

### Test
Test commands to validate.

## Patterns
Key patterns for the codebase.

## Pitfalls
Common mistakes to avoid.
"""
        # First round-trip
        sections1 = parse_sections(original, max_depth=2)
        merged1 = merge_sections(sections1, max_depth=2)

        # Second round-trip
        sections2 = parse_sections(merged1, max_depth=2)
        merged2 = merge_sections(sections2, max_depth=2)

        # Must be identical
        assert merged1 == merged2, (
            f"Round-trip not idempotent.\nAfter 1st merge:\n{merged1}\nAfter 2nd merge:\n{merged2}"
        )

    def test_round_trip_simple_section(self):
        """Single section document should be idempotent."""
        original = """\
# Document Title

## Overview
Some overview content here.

## Details
Details content here.
"""
        sections1 = parse_sections(original, max_depth=2)
        merged1 = merge_sections(sections1, max_depth=2)
        sections2 = parse_sections(merged1, max_depth=2)
        merged2 = merge_sections(sections2, max_depth=2)

        assert merged1 == merged2

    def test_round_trip_empty_document(self):
        """Empty document should round-trip cleanly."""
        original = ""
        sections1 = parse_sections(original, max_depth=2)
        merged1 = merge_sections(sections1, max_depth=2)
        sections2 = parse_sections(merged1, max_depth=2)
        merged2 = merge_sections(sections2, max_depth=2)

        assert merged1 == merged2


class TestSectionContentNoHeading:
    """Test that section content does NOT contain heading as first line."""

    def test_section_content_strips_heading(self):
        """Section content should NOT have heading line as first line."""
        text = """\
# Header

## Overview
This is overview content.

## Commands
This is commands content.
"""
        sections = parse_sections(text, max_depth=2)

        # Section content should not start with ##
        for key, content in sections.items():
            if key in ("_header", "_section_order"):
                continue
            if content.strip():  # Only check non-empty sections
                first_line = content.split("\n")[0]
                assert not first_line.startswith("##"), (
                    f"Section '{key}' content starts with heading: {first_line!r}"
                )
                assert not first_line.startswith("#"), (
                    f"Section '{key}' content starts with any heading: {first_line!r}"
                )

    def test_subsection_content_strips_heading(self):
        """Subsection content should NOT have heading as first line."""
        text = """\
## Top
Top content.

### Sub
Sub content.
"""
        sections = parse_sections(text, max_depth=2)

        # Find the subsection
        subsection_keys = [k for k in sections if "subsection" in k]
        for key in subsection_keys:
            content = sections[key]
            if content.strip():
                first_line = content.split("\n")[0]
                assert not first_line.startswith("###"), (
                    f"Subsection '{key}' content starts with heading: {first_line!r}"
                )

    def test_content_with_multiple_headings_in_subsection(self):
        """Content that itself contains deeper headings should be preserved."""
        text = """\
## Overview
### Linux
Linux specific steps.

### Windows
Windows specific steps.
"""
        sections = parse_sections(text, max_depth=2)
        overview_content = sections.get("section_overview", "")

        # The overview section should contain the deeper headings
        assert "### Linux" in overview_content
        assert "### Windows" in overview_content


class TestHeaderSection:
    """Test that _header section is NOT affected by the fix."""

    def test_header_retains_level1_heading(self):
        """_header should still retain level-1 headings correctly."""
        text = """\
# Document Title

## Overview
Overview content.
"""
        sections = parse_sections(text, max_depth=2)

        header = sections.get("_header", "")
        # _header should contain the level-1 heading
        assert "# Document Title" in header, (
            f"_header should contain level-1 heading, got: {header!r}"
        )

    def test_header_multiple_lines_before_first_section(self):
        """_header should preserve all content before first section."""
        text = """\
# Project

Some description here.

## Section1
Content.
"""
        sections = parse_sections(text, max_depth=2)
        header = sections.get("_header", "")

        assert "# Project" in header
        assert "Some description" in header

    def test_header_empty_when_no_precontent(self):
        """_header should be empty when document starts with section."""
        text = """\
## Section1
Content.
"""
        sections = parse_sections(text, max_depth=2)
        header = sections.get("_header", "")

        # Header may be empty or contain only whitespace
        assert not header.strip()


class TestMergeReconstruction:
    """Test that merge_sections reconstructs headings correctly."""

    def test_merge_reconstructs_heading_from_key(self):
        """merge_sections should reconstruct heading from key when not in content."""
        sections = {
            "_header": "# My Project\n\n",
            "section_overview": "Some overview text",
            "section_commands": "Run commands here.",
        }
        merged = merge_sections(sections, max_depth=2)

        # Should have reconstructed headings
        assert "## Overview" in merged
        assert "## Commands" in merged

    def test_merge_uses_existing_heading_when_present(self):
        """If content has heading, merge should use it as-is."""
        sections = {
            "_header": "",
            "section_overview": "## Custom Heading\nCustom content.",
        }
        merged = merge_sections(sections, max_depth=2)

        # Should preserve the custom heading
        assert "## Custom Heading" in merged
        assert "Custom content" in merged


class TestEdgeCases:
    """Edge case tests."""

    def test_nested_sections_round_trip(self):
        """Nested sections (h2 > h3) should round-trip correctly."""
        text = """\
## Parent

### Child1
Child1 content.

### Child2
Child2 content.
"""
        sections1 = parse_sections(text, max_depth=2)
        merged1 = merge_sections(sections1, max_depth=2)
        sections2 = parse_sections(merged1, max_depth=2)
        merged2 = merge_sections(sections2, max_depth=2)

        assert merged1 == merged2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
