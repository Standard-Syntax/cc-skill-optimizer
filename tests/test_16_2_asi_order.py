"""
Tests for Task 16.2: episode_to_asi() section order.

After Phase 16.2 the section order is:
  1. Outcome
  2. Errors
  3. Task
  4. Final assistant message
  5. Duration
  6. Bash commands
  7. Files touched
  8. Context compaction
  9. Token usage
 10. Tool sequence

The high-signal sections (Outcome, Errors, Task, Final assistant message) are
guaranteed to be present in the first ~1500 chars of the ASI output even
when the LLM judge cap (4000 chars after Phase 16.1) is hit.
"""

from src.parse_session import episode_to_asi


def make_full_episode() -> dict:
    """Episode with all optional fields populated, used to test section order."""
    return {
        "session_id": "test",
        "task_prompt": "Write a hello world function",
        "tool_calls": [
            {"tool": "read", "file": "foo.py"},
            {"tool": "write", "file": "bar.py"},
        ],
        "assistant_text": ["Here is the implementation.", "All done."],
        "outcome": "success",
        "neutral_closing": False,
        "error_messages": ["NameError: name 'x' is not defined", "ImportError: no module"],
        "bash_commands": ["ls -la", "pytest", "uv run ruff check"],
        "files_read": ["foo.py", "bar.py"],
        "files_written": ["hello.py"],
        "thinking_blocks": [],
        "compaction_summary": "Context was compacted due to length",
        "token_stats": {"input": 100, "output": 50, "cache_create": 10, "cache_read": 20},
        "duration_s": 45.0,
        "skill_injections": [],
        "raw_lines": 5,
        "source_path": "/tmp/test.jsonl",
        "timestamp": "2026-06-03T00:00:00",
    }


class TestEpisodeToAsiOrder:
    """The 10 sections appear in the documented order."""

    EXPECTED_ORDER = [
        "## Outcome:",
        "## Errors",
        "## Task",
        "## Final assistant message",
        "## Duration:",
        "## Bash commands executed",
        "## Files touched",
        "## Context compaction occurred",
        "## Token usage:",
        "## Tool sequence",
    ]

    def test_sections_appear_in_expected_order(self):
        ep = make_full_episode()
        asi = episode_to_asi(ep)
        # Find the position of each section header in the output
        positions = []
        for header in self.EXPECTED_ORDER:
            idx = asi.find(header)
            assert idx != -1, f"Section '{header}' not found in ASI output"
            positions.append((idx, header))
        # Verify strict increasing order (each header comes AFTER the previous one)
        for i in range(1, len(positions)):
            assert positions[i][0] > positions[i - 1][0], (
                f"Section '{positions[i][1]}' (at {positions[i][0]}) appears BEFORE "
                f"'{positions[i - 1][1]}' (at {positions[i - 1][0]}) — order is wrong"
            )

    def test_outcome_appears_first_among_content_sections(self):
        ep = make_full_episode()
        asi = episode_to_asi(ep)
        # Outcome must be the very first content section (before Task, Errors, etc.)
        outcome_idx = asi.find("## Outcome:")
        task_idx = asi.find("## Task")
        errors_idx = asi.find("## Errors")
        assert outcome_idx < task_idx, "Outcome must appear before Task"
        assert outcome_idx < errors_idx, "Outcome must appear before Errors"

    def test_final_assistant_message_appears_before_low_signal_sections(self):
        ep = make_full_episode()
        asi = episode_to_asi(ep)
        # Final assistant message must appear BEFORE bash commands / files / tool sequence
        # (these are the sections that get truncated when the cap is hit)
        final_idx = asi.find("## Final assistant message")
        bash_idx = asi.find("## Bash commands executed")
        files_idx = asi.find("## Files touched")
        tool_seq_idx = asi.find("## Tool sequence")
        assert final_idx < bash_idx, "Final assistant must appear before Bash commands"
        assert final_idx < files_idx, "Final assistant must appear before Files touched"
        assert final_idx < tool_seq_idx, "Final assistant must appear before Tool sequence"

    def test_high_signal_sections_all_within_first_1500_chars(self):
        """The 4 highest-signal sections (Outcome, Errors, Task, Final assistant)
        must all be present in the first 1500 chars of the ASI output, so they
        survive even a tight cap."""
        ep = make_full_episode()
        asi = episode_to_asi(ep)
        high_signal_headers = [
            "## Outcome:",
            "## Errors",
            "## Task",
            "## Final assistant message",
        ]
        for header in high_signal_headers:
            idx = asi.find(header)
            assert idx != -1, f"High-signal section '{header}' not found"
            assert idx < 1500, (
                f"High-signal section '{header}' starts at {idx} — should be in first 1500 chars. "
                f"After Phase 16.1, cap is 4000 chars; high-signal content must be at the top."
            )


class TestEpisodeToAsiContentPreserved:
    """Reordering must not lose any content — all fields still appear in the output."""

    def test_all_sections_still_present_for_full_episode(self):
        ep = make_full_episode()
        asi = episode_to_asi(ep)
        # All 10 section headers must be present
        for header in [
            "## Outcome:",
            "## Errors",
            "## Task",
            "## Final assistant message",
            "## Duration:",
            "## Bash commands executed",
            "## Files touched",
            "## Context compaction occurred",
            "## Token usage:",
            "## Tool sequence",
        ]:
            assert header in asi, f"Section '{header}' missing from ASI output after reordering"

    def test_files_touched_combines_read_and_written(self):
        ep = make_full_episode()
        asi = episode_to_asi(ep)
        # The new "Files touched" section must mention both read and written files
        files_section_start = asi.find("## Files touched")
        files_section_end = asi.find("\n\n", files_section_start)
        files_section = asi[files_section_start:files_section_end]
        assert "foo.py" in files_section, "Files read should appear in 'Files touched' section"
        assert "hello.py" in files_section, "Files written should appear in 'Files touched' section"
        # And the format prefix should distinguish them
        assert "read:" in files_section, "Read files should be prefixed with 'read:'"
        assert "wrote:" in files_section, "Written files should be prefixed with 'wrote:'"

    def test_minimal_episode_omits_optional_sections(self):
        """An episode with no errors, no bash commands, no files — the optional
        sections are skipped, but the high-signal sections are still present."""
        minimal_ep = {
            "task_prompt": "Say hi",
            "tool_calls": [],
            "assistant_text": ["Hi"],
            "outcome": "success",
            "error_messages": [],
            "bash_commands": [],
            "files_read": [],
            "files_written": [],
            "compaction_summary": None,
            "token_stats": {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0},
            "duration_s": None,
        }
        asi = episode_to_asi(minimal_ep)
        # High-signal sections must be present
        assert "## Outcome:" in asi
        assert "## Task" in asi
        assert "## Final assistant message" in asi
        # Optional sections must be absent
        assert "## Errors" not in asi
        assert "## Bash commands" not in asi
        assert "## Files touched" not in asi
        assert "## Duration" not in asi
        assert "## Context compaction" not in asi
        # Tool sequence is always present (even if "(0 total)")
        assert "## Tool sequence" in asi
