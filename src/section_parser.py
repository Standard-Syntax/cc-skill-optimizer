"""
section_parser.py
================
Within-file section-level parsing for markdown documents.

Treats each top-level markdown heading (##) as an independent optimizable
component, enabling GEPA to evolve individual sections of CLAUDE.md, AGENTS.md,
and SKILL.md files independently while preserving cross-references between them.

Two modes:
  1. FLAT  — each ## heading is a separate component (default)
  2. NESTED — sections form a tree: h2 → children h3 → children h4 ...

Example:
    # Document
    ## Overview       → key: "section_overview"
    Some content...
    ## Commands       → key: "section_commands"
    ### Build        → key: "section_commands.subsection_build"
    More content...

Usage:
    sections = parse_sections(markdown_text, max_depth=2)
    # {"section_overview": "...", "section_commands": "...", ...}

    merged = merge_sections(sections)
    # Reconstructs the original document structure
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """
    A parsed section with heading level, name, content, and optional children.
    """

    level: int  # 1 = #, 2 = ##, 3 = ###, etc.
    heading: str  # The full heading text (e.g. "## Overview")
    name: str  # Normalized key name (e.g. "overview" or "commands.subsection_build")
    content: str  # Section body text (excludes the heading line)
    children: list[Section] = field(default_factory=list)
    start_line: int = 0  # 1-indexed line number in original file
    end_line: int = 0


# ---------------------------------------------------------------------------
# Heading normalization
# ---------------------------------------------------------------------------


def _normalize_key(heading: str, parent_key: str = "") -> str:
    """
    Convert a heading like "## Build Commands" → "section_build_commands".
    With parent: "### Linux" + parent "section_build" → "section_build.subsection_linux"
    """
    # Strip leading # and whitespace
    text = re.sub(r"^#+\s*", "", heading).strip()
    # Lowercase, replace spaces/hyphens with underscores, strip non-alphanumeric
    slug = re.sub(r"[^a-z0-9_]", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")

    if parent_key:
        return f"{parent_key}.subsection_{slug}" if slug else parent_key
    return f"section_{slug}" if slug else ""


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def parse_sections(
    text: str,
    max_depth: int = 2,
) -> dict[str, str]:
    """
    Parse a markdown document into a flat dict of section_key → section_content.

    Args:
        text:      Raw markdown text
        max_depth: Maximum heading level to treat as a component.
                   1 = only # headings (rare, usually document title)
                   2 = ## headings (default — most useful for skill/claude/agents docs)
                   3 = ### and deeper

    Returns:
        dict[str, str]: section_key → section content.
                        The special key "_header" holds any content before the first heading
                        at level <= max_depth (e.g. the document title in a level-1 heading).

    Example:
        >>> text = "# My Project\\n\\n## Overview\\nSome overview text\\n## Commands\\n### Build\\nBuild steps"
        >>> parse_sections(text, max_depth=2)
        {'_header': '# My Project\\n\\n',
         'section_overview': 'Some overview text',
         'section_commands': '### Build\\nBuild steps'}
    """
    lines = text.splitlines(keepends=True)
    sections: dict[str, str] = {}
    # Track the order of section keys (not _header) for correct document reconstruction.
    section_order: list[str] = []
    section_stack: list[tuple[int, str]] = []  # (level, key)
    current_key = "_header"
    current_lines: list[str] = []
    # Track whether we've encountered any heading at level <= max_depth.
    # Before this, content (including level-1 headings) goes to _header.
    _seen_section = False

    def _flush(target_key: str, lines: list[str]) -> None:
        """Save accumulated lines to sections dict, stripping trailing blank lines."""
        content = "".join(lines).rstrip("\n")
        if content:
            sections[target_key] = content

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            heading = m.group(0)  # keep "## " prefix

            if level > max_depth:
                # Deeper heading — treat as body content
                current_lines.append(line)
                continue

            # Level 1 heading before any section has been seen: treat as header content.
            # This handles "# Document Title\n\n## Section" correctly — the title
            # and any content before ## go to _header, not a section.
            if level == 1 and not _seen_section:
                current_lines.append(line)
                continue

            # This heading is at or above max_depth — it's a real section.
            # Before we've seen any real section, flush accumulated content to _header.
            if not _seen_section:
                _seen_section = True
                _flush("_header", current_lines)
                current_lines = []

            # Flush current section (strip heading line from stored content)
            if current_key:
                flush_lines = current_lines[1:] if current_lines else current_lines
                _flush(current_key, flush_lines)
                # Store raw heading for idempotent round-trip (preserves casing)
                if current_lines:
                    sections[f"_{current_key}_heading"] = current_lines[0]

            # Find parent in stack
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()

            if section_stack:
                parent_key = section_stack[-1][1]
                key = _normalize_key(heading, parent_key)
            else:
                key = _normalize_key(heading)

            current_key = key
            current_lines = [line]  # start new section (include heading line for nested mode)
            section_stack.append((level, key))
            # Track top-level section order for reconstruction
            if level == 2:  # only track top-level (max_depth) sections
                section_order.append(key)
        else:
            current_lines.append(line)

    # Flush final section (strip heading line from stored content)
    if current_key:
        flush_lines = current_lines[1:] if current_lines else current_lines
        _flush(current_key, flush_lines)
        # Store raw heading for idempotent round-trip (preserves casing)
        if current_lines:
            sections[f"_{current_key}_heading"] = current_lines[0]

    # Attach section_order metadata for merge_sections
    sections["_section_order"] = "\t".join(section_order)
    return sections


def merge_sections(
    sections: dict[str, str],
    source_text: str = "",
    max_depth: int = 2,
) -> str:
    """
    Reassemble sections back into a markdown document.

    Args:
        sections:    dict[str, str] of section_key → section content
        source_text: Optional original text — used to preserve sections not in
                     the sections dict (e.g. sections that were pruned).
        max_depth:   Heading level that was used during parsing.

    Returns:
        Reconstructed markdown text with proper heading hierarchy.

    Example:
        >>> sections = {'_header': '# My Project\\n\\n', 'section_overview': 'Overview text'}
        >>> merge_sections(sections)
        '# My Project\\n\\n## Overview\\nOverview text'
    """
    if not sections:
        return source_text

    # Separate header from numbered sections
    header = sections.get("_header", "")
    # Extract section order if available
    section_order_raw = sections.get("_section_order", "")
    section_order = section_order_raw.split("\t") if section_order_raw else []

    # Separate metadata keys (including stored raw headings)
    meta_keys = {"_header", "_section_order"}
    heading_keys = {k for k in sections if k.startswith("_") and k.endswith("_heading")}
    meta_keys |= heading_keys
    content_keys = [k for k in sections if k not in meta_keys]

    parts = [header] if header else []

    # For sections not in the order list (e.g., subsections), sort by key for determinism
    def sort_key(k: str) -> tuple[int, str]:
        # Top-level sections (section_*) come in document order from section_order.
        # Subsections (section_*.subsection_*) come after their parent, sorted by key.
        depth = k.count(".")
        return (depth, k)

    sorted_keys = sorted(content_keys, key=sort_key)

    # Build ordered list of all section keys
    if section_order:
        # Interleave ordered top-level sections with their ordered subsections
        ordered_all: list[str] = []
        for top_key in section_order:
            ordered_all.append(top_key)
            # Append any subsections of this top-level section that aren't in the order list
            prefix = top_key + "."
            for k in sorted_keys:
                if k != top_key and k.startswith(prefix) and k not in ordered_all:
                    ordered_all.append(k)
        # Add any orphaned keys not in section_order
        for k in sorted_keys:
            if k not in ordered_all:
                ordered_all.append(k)
    else:
        ordered_all = sorted_keys

    for key in ordered_all:
        if key in meta_keys:
            continue
        content = sections[key]
        if not content.strip():
            continue

        # Determine heading level from key depth
        depth = key.count(".") + 1  # section_* = h2, section_*.subsection_* = h3
        heading_depth = min(depth + 1, 6)  # +1 because section_* starts at h2

        # Check for stored raw heading (preserves original casing)
        stored_heading = sections.get(f"_{key}_heading")
        if stored_heading:
            parts.append(f"{stored_heading.rstrip()}\n{content}")
        else:
            # Extract heading from content if present (first line is the heading)
            lines = content.splitlines(keepends=True)
            if lines and re.match(r"^#{1,6}\s+", lines[0]):
                # Content already has heading — use as-is
                parts.append(content)
            else:
                # Reconstruct heading from key name
                heading_text = _key_to_heading(key)
                parts.append(f"{'#' * heading_depth} {heading_text}\n{content}")

    result = "\n\n".join(p for p in parts if p.strip())
    return result


def _key_to_heading(key: str) -> str:
    """Convert a section key back to human-readable heading text."""
    # section_overview → Overview
    # section_commands.subsection_build → Build
    parts = key.split(".")
    last = parts[-1]
    if last.startswith("subsection_"):
        last = last[len("subsection_") :]
    elif last.startswith("section_"):
        last = last[len("section_") :]
    # Convert underscores back to spaces, title-case
    return last.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Tree representation (for debugging / visualization)
# ---------------------------------------------------------------------------


def find_section(sections: list[Section], name: str) -> Section | None:
    """
    Recursively find a section by name in a tree of sections.

    Exposed at module level for external use.
    """
    for s in sections:
        if s.name == name:
            return s
        found = find_section(s.children, name)
        if found:
            return found
    return None


def build_section_tree(text: str, max_depth: int = 2) -> list[Section]:
    """
    Build a tree of Section objects from markdown text.

    Useful for visualization and for understanding section hierarchy.
    """
    lines = text.splitlines(keepends=True)
    root: list[Section] = []
    stack: list[Section] = []

    for lineno, line in enumerate(lines, start=1):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if not m:
            continue

        level = len(m.group(1))
        heading = m.group(0)

        # Find parent
        while stack and stack[-1].level >= level:
            stack.pop()

        parent = stack[-1] if stack else None
        parent_list = parent.children if parent else root

        name = _normalize_key(heading, parent.name if parent else "")

        section = Section(
            level=level,
            heading=heading,
            name=name,
            content="",
            start_line=lineno,
        )

        parent_list.append(section)
        stack.append(section)

    # Second pass: collect content between headings
    current: Section | None = None
    content_buffer: list[str] = []

    for lineno, line in enumerate(lines, start=1):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m and len(m.group(1)) <= max_depth:
            if current:
                current.content = "".join(content_buffer).rstrip("\n")
                current.end_line = lineno - 1
            level = len(m.group(1))
            heading = m.group(0)
            while stack and stack[-1].level >= level:
                stack.pop()
            current = stack[-1] if stack else None
            if current:
                current = find_section(
                    root, _normalize_key(heading, stack[-1].name if stack else "")
                )
            content_buffer = []
        else:
            if current is None:
                content_buffer.append(line)

    if current:
        current.content = "".join(content_buffer).rstrip("\n")
        current.end_line = len(lines)

    return root


# ---------------------------------------------------------------------------
# File-level operations
# ---------------------------------------------------------------------------


def load_sections_from_file(path: Path, max_depth: int = 2) -> dict[str, str]:
    """Load and parse a markdown file into sections."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_sections(text, max_depth=max_depth)


def save_sections_to_file(sections: dict[str, str], path: Path, max_depth: int = 2) -> None:
    """Reassemble sections and write back to a file."""
    # Pass all sections to merge_sections — it handles metadata exclusion internally
    original = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    merged = merge_sections(sections, source_text=original, max_depth=max_depth)
    path.write_text(merged, encoding="utf-8")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def section_summary(sections: dict[str, str]) -> str:
    """Human-readable summary of parsed sections."""
    meta_keys = {"_header", "_section_order"}
    meta_keys |= {k for k in sections if k.startswith("_") and k.endswith("_heading")}
    lines = ["Parsed sections:"]
    for key, content in sorted(sections.items()):
        if key in meta_keys:
            continue
        preview = content[:80].replace("\n", " ").strip()
        lines.append(f"  {key}: {preview}...")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (standalone test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """\
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

    sections = parse_sections(sample, max_depth=2)
    print(section_summary(sections))
    print()

    # Test merge
    merged = merge_sections(sections, max_depth=2)
    print("Reconstructed:")
    print(merged[:500])
