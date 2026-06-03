"""
Test idempotency fix for section_parser.py

Verifies:
1. Round-tripping a document through parse → merge → parse → merge produces the same output
2. Section content does NOT contain the heading line as its first line
3. _header section is NOT affected (it still retains level-1 headings correctly)
"""

import sys

import pytest

sys.path.insert(0, "src")
from section_parser import (
    build_section_tree,
    find_section,
    load_sections_from_file,
    merge_sections,
    parse_sections,
    save_sections_to_file,
)


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
            # Skip heading metadata keys (e.g. _section_overview_heading)
            if key.startswith("_") and key.endswith("_heading"):
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


class TestHeadingCasingPreservation:
    """Test that non-standard heading casing is preserved through round-trips."""

    def test_non_standard_casing_preserved(self):
        """Headings like '## SQL Server 2016 Quirks' should not become title-cased."""
        text = """\
# My Project

## SQL Server 2016 Quirks
Some content about SQL Server 2016 quirks.

## iOS App Development
Some iOS content.
"""
        sections = parse_sections(text, max_depth=2)
        merged = merge_sections(sections, max_depth=2)

        # SQL Server 2016 Quirks should NOT become "Sql Server 2016 Quirks"
        assert "## SQL Server 2016 Quirks" in merged
        # iOS should NOT become "Ios"
        assert "## iOS App Development" in merged

    def test_all_caps_acronym_preserved(self):
        """All-caps sections like '## API' should not become '## Api'."""
        text = """\
## API
API reference content.

## REST
REST API details.
"""
        sections = parse_sections(text, max_depth=2)
        merged = merge_sections(sections, max_depth=2)

        assert "## API" in merged
        assert "## REST" in merged

    def test_mixed_case_round_trip_idempotent(self):
        """Documents with mixed casing should be fully idempotent."""
        text = """\
## SQL Server 2016 Quirks
Quirk content here.

## iOS App Development
iOS content here.

## API
API reference.
"""
        sections1 = parse_sections(text, max_depth=2)
        merged1 = merge_sections(sections1, max_depth=2)
        sections2 = parse_sections(merged1, max_depth=2)
        merged2 = merge_sections(sections2, max_depth=2)

        assert merged1 == merged2

    def test_stored_heading_metadata_excluded_from_content(self):
        """Stored heading metadata should not appear in content sections."""
        text = """\
## Overview
Overview content.
"""
        sections = parse_sections(text, max_depth=2)

        # Check that heading metadata exists but is NOT in the content
        heading_key = "_section_overview_heading"
        assert heading_key in sections, "Heading metadata should be stored"
        assert sections[heading_key].rstrip() == "## Overview"

        # The actual content should not contain the heading
        assert "## Overview" not in sections.get("section_overview", "")


class TestFindSectionModuleLevel:
    """Test that find_section works at module level."""

    def test_find_section_module_level(self):
        """find_section should be accessible as a module-level function."""
        text = """\
## Overview
Overview content.

## Commands
Commands content.

### Build
Build steps.
"""
        tree = build_section_tree(text, max_depth=2)

        # find_section should work on the tree
        overview_section = find_section(tree, "section_overview")
        assert overview_section is not None
        assert overview_section.name == "section_overview"

        commands_section = find_section(tree, "section_commands")
        assert commands_section is not None

        # find_section should work for subsections too
        build_section = find_section(tree, "section_commands.subsection_build")
        assert build_section is not None
        assert build_section.name == "section_commands.subsection_build"

    def test_find_section_returns_none_for_missing(self):
        """find_section should return None when section is not found."""
        text = """\
## Overview
Overview content.
"""
        tree = build_section_tree(text, max_depth=2)

        result = find_section(tree, "section_nonexistent")
        assert result is None

    def test_find_section_not_closure_bound(self):
        """find_section at module level should not depend on build_section_tree call."""
        # This verifies the function is truly hoisted to module level
        # by calling it without ever calling build_section_tree
        import section_parser

        # find_section should exist as a module-level attribute on the module
        assert hasattr(section_parser, "find_section")
        assert callable(section_parser.find_section)
        assert callable(section_parser.find_section)


class TestFileSaveHeadingPreservation:
    """Verify save_sections_to_file preserves heading casing (file-based round-trip)."""

    def test_sql_server_heading_survives_file_save(self, tmp_path):
        # Write a file with non-standard casing
        original = """# Project

## SQL Server 2016 Quirks

Some content.

## Commands

Run uv commands.
"""
        path = tmp_path / "SKILL.md"
        path.write_text(original, encoding="utf-8")

        # Round-trip through save_sections_to_file
        sections = load_sections_from_file(path, max_depth=2)
        save_sections_to_file(sections, path, max_depth=2)

        # Read back
        result = path.read_text(encoding="utf-8")
        assert "## SQL Server 2016 Quirks" in result
        assert "## Sql Server 2016 Quirks" not in result, f"Casing drift detected:\n{result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
