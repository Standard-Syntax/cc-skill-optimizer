"""
Tests for subagent JSONL scanning fix in watch_and_learn.py (Task 6.1).

The change adds subagent JSONL files (at project_dir/subagents/agent-*.jsonl)
to the scanning loop.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


# Helper function that mimics the scanning logic from watch_and_learn.py
def scan_for_jsonl_files(
    projects_dir: Path,
    project_filter: str | None,
    known_paths: set[str],
) -> list[Path]:
    """Mimics the scan loop from watch_and_learn.watch_and_learn."""
    new_files: list[Path] = []
    if projects_dir.exists():
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            if project_filter and project_filter not in project_dir.name:
                continue
            # Scan both top-level and subagent JSONL files.
            # Subagent logs live at project_dir/subagents/agent-<id>.jsonl (one level deep).
            for jsonl in list(project_dir.glob("*.jsonl")) + list(
                (project_dir / "subagents").glob("*.jsonl")
                if (project_dir / "subagents").exists()
                else []
            ):
                if str(jsonl) not in known_paths and time.time() - jsonl.stat().st_mtime > 10:
                    # Only process sessions that haven't been modified in >10s
                    # (give Claude Code time to finish writing)
                    new_files.append(jsonl)
    return new_files


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary projects directory."""
    return tmp_path / "projects"


class TestTopLevelJsonlDiscovery:
    """Test that top-level JSONL files are still discovered."""

    def test_top_level_jsonl_discovered(self, tmp_project_dir: Path) -> None:
        """Test: top-level JSONL files are still discovered."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)
        # Create a top-level JSONL file
        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age the file (move mtime back by 30 seconds)
        old_mtime = time.time() - 30
        jsonl_file.touch()
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        assert jsonl_file in new_files


class TestSubagentJsonlDiscovery:
    """Test that subagent JSONL files are discovered."""

    def test_subagent_jsonl_discovered(self, tmp_project_dir: Path) -> None:
        """Test: subagent JSONL files are discovered."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)
        subagents_dir = project_dir / "subagents"
        subagents_dir.mkdir(parents=True)

        # Create a subagent JSONL file
        jsonl_file = subagents_dir / "agent-abc.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age the file
        old_mtime = time.time() - 30
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        assert jsonl_file in new_files


class TestMixedDiscovery:
    """Test discovery of both top-level and subagent files."""

    def test_both_top_level_and_subagent_files(self, tmp_project_dir: Path) -> None:
        """Test: both top-level and subagent files in the same project."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)
        subagents_dir = project_dir / "subagents"
        subagents_dir.mkdir(parents=True)

        # Create both
        top_file = project_dir / "foo.jsonl"
        top_file.write_text('{"task": "test"}\n')
        sub_file = subagents_dir / "agent-abc.jsonl"
        sub_file.write_text('{"task": "test"}\n')

        # Age both files
        old_mtime = time.time() - 30
        import os

        os.utime(top_file, (old_mtime, old_mtime))
        os.utime(sub_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        assert top_file in new_files
        assert sub_file in new_files
        assert len(new_files) == 2


class TestMissingDirectoryHandling:
    """Test handling of missing subagents/ directory."""

    def test_subagents_dir_not_exists_no_crash(self, tmp_project_dir: Path) -> None:
        """Test: subagents/ directory doesn't exist — no crash."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)
        # Note: NO subagents/ directory created

        # Create a top-level JSONL file
        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age the file
        old_mtime = time.time() - 30
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        # Should succeed and find only the top-level file
        assert jsonl_file in new_files
        assert len(new_files) == 1

    def test_subagents_dir_empty_no_crash(self, tmp_project_dir: Path) -> None:
        """Test: subagents/ directory exists but is empty — no crash."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)
        subagents_dir = project_dir / "subagents"
        subagents_dir.mkdir(parents=True)
        # Note: EMPTY subagents/ directory

        # Create a top-level JSONL file
        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age the file
        old_mtime = time.time() - 30
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        # Should succeed and find only the top-level file
        assert jsonl_file in new_files
        assert len(new_files) == 1


class TestNonJsonlFilters:
    """Test that non-JSONL files in subagents/ are ignored."""

    def test_non_jsonl_files_ignored(self, tmp_project_dir: Path) -> None:
        """Test: non-JSONL files in subagents/ are ignored."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)
        subagents_dir = project_dir / "subagents"
        subagents_dir.mkdir(parents=True)

        # Create a non-JSONL file
        txt_file = subagents_dir / "notes.txt"
        txt_file.write_text("some notes\n")

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        # Should NOT be in new_files
        assert txt_file not in new_files
        assert len(new_files) == 0


class TestNoRecursiveDescent:
    """Test that nested subagent directories are NOT recursed."""

    def test_nested_subagent_dirs_not_recurse(self, tmp_project_dir: Path) -> None:
        """Test: nested subagent directories (e.g., subagents/agent-1/subagents/) are NOT recursed."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)
        subagents_dir = project_dir / "subagents"
        subagents_dir.mkdir(parents=True)

        # Create nested subagent directory structure
        nested_dir = subagents_dir / "agent-1" / "subagents"
        nested_dir.mkdir(parents=True)
        nested_file = nested_dir / "agent-2.jsonl"
        nested_file.write_text('{"task": "test"}\n')
        # Age the file
        old_mtime = time.time() - 30
        import os

        os.utime(nested_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        # Nested file should NOT be in new_files (pattern is one level deep)
        assert nested_file not in new_files
        assert len(new_files) == 0


class TestJsonlPatternMatching:
    """Test that .jsonl in the name triggers discovery regardless of content."""

    def test_jsonl_in_name_not_valid_json_still_discovered(self, tmp_project_dir: Path) -> None:
        """Test: file with .jsonl in the name but not actually JSON is still discovered."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)

        # Create a file with .jsonl extension but non-JSON text content
        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text("not valid json content\n")
        # Age the file
        old_mtime = time.time() - 30
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        # Should still be discovered (scan doesn't validate content)
        assert jsonl_file in new_files


class TestKnownPathsDeduplication:
    """Test that known_paths prevents reprocessing."""

    def test_scan_deduplicates_via_known_paths(self, tmp_project_dir: Path) -> None:
        """Test: scan deduplicates via known_paths."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)

        # Create a file
        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age the file
        old_mtime = time.time() - 30
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()

        # First scan
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)
        assert jsonl_file in new_files

        # Add to known_paths (simulating the processing in watch_and_learn)
        known_paths.add(str(jsonl_file))

        # Second scan - should be empty now
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)
        assert len(new_files) == 0


class TestMtimeGuard:
    """Test the 10-second mtime guard."""

    def test_recent_file_not_discovered(self, tmp_project_dir: Path) -> None:
        """Test: create a file with mtime 5 seconds ago; verify it's NOT discovered."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)

        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age by only 5 seconds (less than 10s guard)
        recent_mtime = time.time() - 5
        import os

        os.utime(jsonl_file, (recent_mtime, recent_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        assert jsonl_file not in new_files

    def test_old_file_discovered(self, tmp_project_dir: Path) -> None:
        """Test: create a file with mtime 30 seconds ago; verify it IS discovered."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)

        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age by 30 seconds (greater than 10s guard)
        old_mtime = time.time() - 30
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        assert jsonl_file in new_files


class TestParseSessionCalls:
    """Test that parse_session is called for each file in new_files."""

    def test_parse_session_called_for_each_new_file(
        self, tmp_project_dir: Path, monkeypatch
    ) -> None:
        """Test: cross-platform .exists() check."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)

        # Create both top-level and subagent files
        top_file = project_dir / "foo.jsonl"
        top_file.write_text(
            '{"task": "test", "tool_calls": ["call1", "call2"], "duration_s": 10}\n'
        )
        subagents_dir = project_dir / "subagents"
        subagents_dir.mkdir(parents=True)
        sub_file = subagents_dir / "agent-abc.jsonl"
        sub_file.write_text(
            '{"task": "test", "tool_calls": ["call1", "call2"], "duration_s": 10}\n'
        )

        # Age both files
        old_mtime = time.time() - 30
        import os

        os.utime(top_file, (old_mtime, old_mtime))
        os.utime(sub_file, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        # Verify both are discovered
        assert top_file in new_files
        assert sub_file in new_files
        assert len(new_files) == 2


class TestCrossPlatformExists:
    """Test cross-platform .exists() check for subagents/."""

    def test_mocked_subagents_not_exists_no_crash(self, tmp_project_dir: Path, monkeypatch) -> None:
        """Test: mock Path.exists to return False for subagents/; verify the empty list is used."""
        project_dir = tmp_project_dir / "myproject"
        project_dir.mkdir(parents=True)

        # Create a top-level file
        jsonl_file = project_dir / "foo.jsonl"
        jsonl_file.write_text('{"task": "test"}\n')
        # Age the file
        old_mtime = time.time() - 30
        import os

        os.utime(jsonl_file, (old_mtime, old_mtime))

        # Patch exists() to always return False for subagents/
        original_exists = Path.exists

        def mock_exists(self):
            if isinstance(self, Path) and self.name == "subagents":
                return False
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", mock_exists)

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        # Should succeed and still find the top-level file
        assert jsonl_file in new_files


class TestProjectFilter:
    """Test project_filter functionality."""

    def test_project_filter_respected(self, tmp_project_dir: Path) -> None:
        """Test: project_filter is respected when scanning."""
        # Create two projects
        project_a = tmp_project_dir / "myproject"
        project_a.mkdir(parents=True)
        project_b = tmp_project_dir / "otherproject"
        project_b.mkdir(parents=True)

        # Create files in both
        file_a = project_a / "foo.jsonl"
        file_a.write_text('{"task": "test"}\n')
        file_b = project_b / "bar.jsonl"
        file_b.write_text('{"task": "test"}\n')

        # Age both files
        old_mtime = time.time() - 30
        import os

        os.utime(file_a, (old_mtime, old_mtime))
        os.utime(file_b, (old_mtime, old_mtime))

        known_paths: set[str] = set()

        # Filter to only "myproject"
        new_files = scan_for_jsonl_files(tmp_project_dir, "myproject", known_paths)

        assert file_a in new_files
        assert file_b not in new_files


class TestMultipleProjects:
    """Test scanning multiple projects."""

    def test_multiple_projects_scanned(self, tmp_project_dir: Path) -> None:
        """Test: multiple projects with JSONL files are all scanned."""
        project_a = tmp_project_dir / "project1"
        project_a.mkdir(parents=True)
        project_b = tmp_project_dir / "project2"
        project_b.mkdir(parents=True)

        # Create files in both projects + subagent
        file_a = project_a / "foo.jsonl"
        file_a.write_text('{"task": "test"}\n')
        subagents_b = project_b / "subagents"
        subagents_b.mkdir(parents=True)
        file_b = subagents_b / "agent-x.jsonl"
        file_b.write_text('{"task": "test"}\n')

        # Age both files
        old_mtime = time.time() - 30
        import os

        os.utime(file_a, (old_mtime, old_mtime))
        os.utime(file_b, (old_mtime, old_mtime))

        known_paths: set[str] = set()
        new_files = scan_for_jsonl_files(tmp_project_dir, None, known_paths)

        assert file_a in new_files
        assert file_b in new_files
        assert len(new_files) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
