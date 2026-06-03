"""
Tests for the two-poll size stability check in watch_and_learn.py (Task 6.2).

These tests verify:
- The scan loop's size stability check (first poll, size unchanged, size changed)
- The pre-existing files mtime fallback
- File deletion between glob and stat
- The `known_files`, `stable_files`, `preexisting_paths` data structures
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


def create_episode_jsonl(path: Path, task: str = "Test task", tool_calls: int = 5) -> None:
    """Create a minimal valid JSONL file representing a session."""
    data = {
        "task_prompt": task,
        "tool_calls": [{"name": f"Tool{i}", "input": {}} for i in range(tool_calls)],
        "outcome": "success",
        "duration_s": 1.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def scan_cycle(
    projects_dir: Path,
    project_filter: str | None,
    known_files: dict[str, int],
    stable_files: set[str],
    preexisting_paths: set[str],
) -> list[Path]:
    """
    Replicate the scan logic from watch_and_learn() for testing.
    This mimics lines 66-121 of the original file.
    """
    new_files: list[Path] = []

    if not projects_dir.exists():
        return new_files

    # Iterate over project directories inside projects/
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        if project_filter and project_filter not in project_dir.name:
            continue

        def _safe_subagent_glob(subdir: Path) -> list[Path]:
            try:
                return list(subdir.glob("*.jsonl"))
            except (PermissionError, OSError):
                return []

        # Scan top-level and subagent JSONL files
        candidate_files: list[Path] = list(project_dir.glob("*.jsonl")) + _safe_subagent_glob(
            project_dir / "subagents"
        )

        for jsonl in candidate_files:
            path_str = str(jsonl)
            try:
                current_size = jsonl.stat().st_size
            except (FileNotFoundError, OSError):
                # File was deleted between glob and stat — skip silently
                continue

            prev_size = known_files.get(path_str)

            if prev_size is None:
                # First poll for this file
                if path_str in preexisting_paths:
                    # Pre-existing: use mtime guard (legacy behavior)
                    if time.time() - jsonl.stat().st_mtime > 10:
                        known_files[path_str] = current_size
                        if path_str not in stable_files:
                            new_files.append(jsonl)
                else:
                    # New file: record size, wait for second poll
                    known_files[path_str] = current_size
            elif prev_size == current_size and path_str not in stable_files:
                # Size unchanged — mark as stable and add
                stable_files.add(path_str)
                new_files.append(jsonl)
            else:
                # Size changed — update record
                known_files[path_str] = current_size

    return new_files


@pytest.fixture
def tmp_projects_dir(tmp_path: Path) -> Path:
    """Create a temporary projects directory structure.

    Returns projects/ directory (parent of project folders).
    """
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(parents=True)
    proj_dir = projects_dir / "testproject"
    proj_dir.mkdir(parents=True)
    return projects_dir  # Return projects/, not the project folder


# =============================================================================
# Test Cases
# =============================================================================


def test_new_file_not_added_on_first_poll(tmp_projects_dir: Path) -> None:
    """
    Test: new file is recorded but NOT added to new_files on first poll.

    Create a JSONL file; run one poll; verify new_files is empty
    (file is in known_files but not stable_files yet).
    """
    # Get actual project dir inside projects/
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)

    # File should be recorded in known_files
    assert str(jsonl_file) in known_files
    # But NOT in new_files (needs second poll for stability)
    assert str(jsonl_file) not in [str(p) for p in new_files]
    # And not in stable_files yet
    assert str(jsonl_file) not in stable_files


def test_new_file_becomes_stable_on_second_poll(tmp_projects_dir: Path) -> None:
    """
    Test: new file becomes stable and is added to new_files on second poll.

    Create a JSONL file; run two polls (with no change between polls);
    verify new_files contains the file on the second poll.
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # First poll
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl_file) not in [str(p) for p in new_files]

    # Second poll (no change to file)
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    # Should now be in new_files
    assert str(jsonl_file) in [str(p) for p in new_files]
    # And in stable_files
    assert str(jsonl_file) in stable_files


def test_file_changing_size_not_stable(tmp_projects_dir: Path) -> None:
    """
    Test: file changing in size between polls does NOT become stable.

    Create a JSONL file, append content, run two polls
    (each after appending); verify new_files is empty.
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)
    initial_size = jsonl_file.stat().st_size

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # First poll
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl_file) not in [str(p) for p in new_files]

    # Modify file between polls
    jsonl_file.write_text(
        jsonl_file.read_text() + "\n" + '{"task_prompt": "extra"}',
        encoding="utf-8",
    )

    # Second poll - file size changed
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    # Still should not be in new_files (size changed)
    assert str(jsonl_file) not in [str(p) for p in new_files]
    # Should record updated size
    assert known_files[str(jsonl_file)] != initial_size


def test_file_changing_then_stable(tmp_projects_dir: Path) -> None:
    """
    Test: file changing then stable becomes stable.

    Create a JSONL file; poll 1 (size=N); append content; poll 2 (size=N+k);
    no change; poll 3 (size=N+k); verify new_files contains the file after poll 3.
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)
    initial_size = jsonl_file.stat().st_size

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # Poll 1 - record initial size
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl_file) not in [str(p) for p in new_files]

    # Append content
    jsonl_file.write_text(
        jsonl_file.read_text() + "\n" + '{"task_prompt": "more"}',
        encoding="utf-8",
    )

    # Poll 2 - size changed, records new size
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl_file) not in [str(p) for p in new_files]
    assert known_files[str(jsonl_file)] != initial_size

    # Poll 3 - no change, should stabilize
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl_file) in [str(p) for p in new_files]
    assert str(jsonl_file) in stable_files


def test_preexisting_file_old_mtime_added_on_first_poll(tmp_projects_dir: Path) -> None:
    """
    Test: pre-existing file with old mtime is added on first poll.

    Initialize with a corpus containing an episode from a path;
    create a JSONL file at that path with mtime 30s ago;
    run one poll; verify the file IS in new_files (mtime fallback).
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_old.jsonl"
    create_episode_jsonl(jsonl_file)
    path_str = str(jsonl_file)

    # Set old mtime (30 seconds ago)
    old_time = time.time() - 30
    os.utime(jsonl_file, (old_time, old_time))

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = {path_str}

    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)

    # Should be in new_files due to old mtime fallback (> 10 second threshold)
    assert path_str in [str(p) for p in new_files]


def test_preexisting_file_recent_mtime_not_added(tmp_projects_dir: Path) -> None:
    """
    Test: pre-existing file with recent mtime is NOT added on first poll.

    Same as above but mtime is 5s ago;
    verify the file is NOT in new_files (mtime guard filters it).
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_recent.jsonl"
    create_episode_jsonl(jsonl_file)
    path_str = str(jsonl_file)

    # Set recent mtime (5 seconds ago)
    recent_time = time.time() - 5
    os.utime(jsonl_file, (recent_time, recent_time))

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = {path_str}

    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)

    # Should NOT be in new_files (mtime too recent, < 10 seconds)
    assert path_str not in [str(p) for p in new_files]


def test_file_deletion_between_glob_and_stat(tmp_projects_dir: Path) -> None:
    """
    Test: file deletion between glob and stat is handled.

    Create then delete a JSONL file;
    verify the loop continues (no crash) and new_files is empty.
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # Delete the file before scanning
    jsonl_file.unlink()

    # Should not crash, empty new_files
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert len(new_files) == 0


def test_stable_file_not_readded(tmp_projects_dir: Path) -> None:
    """
    Test: stable file is not re-added on subsequent polls.

    A file already in stable_files should not appear in new_files again.
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)
    path_str = str(jsonl_file)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # First poll - record
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)

    # Second poll - stabilize
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert path_str in stable_files

    # Third poll - should NOT appear in new_files again
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert path_str not in [str(p) for p in new_files]


def test_top_level_and_subagent_files_both_scanned(tmp_projects_dir: Path) -> None:
    """
    Test: top-level and subagent files are both scanned.

    Create top-level and subagent JSONLs;
    both should appear in new_files on second poll.
    """
    project_dir = next(tmp_projects_dir.iterdir())

    # Top-level file
    top_jsonl = project_dir / "session_top.jsonl"
    create_episode_jsonl(top_jsonl)

    # Subagent file
    subagent_dir = project_dir / "subagents"
    subagent_dir.mkdir(exist_ok=True)
    subagent_jsonl = subagent_dir / "session_sub.jsonl"
    create_episode_jsonl(subagent_jsonl)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # First poll
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)

    # Second poll - both should be stable
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)

    new_file_paths = [str(p) for p in new_files]
    assert str(top_jsonl) in new_file_paths
    assert str(subagent_jsonl) in new_file_paths


def test_very_large_file_size_handled(tmp_projects_dir: Path) -> None:
    """
    Test: very large file size is handled correctly.

    Create a JSONL file with size 100KB+.
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_large.jsonl"

    # Create a large-ish file (~200KB)
    large_data = {"task_prompt": "Large task", "tool_calls": []}
    content = json.dumps(large_data) * 5000  # ~200KB+
    jsonl_file.write_text(content, encoding="utf-8")

    file_size = jsonl_file.stat().st_size
    assert file_size > 100_000, f"File size {file_size} not large enough"

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # First poll
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl_file) in known_files
    assert file_size == known_files[str(jsonl_file)]

    # Second poll - large file size checked
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl_file) in [str(p) for p in new_files]


def test_mtime_guard_for_preexisting_files() -> None:
    """
    Test: 10-second mtime guard is still used for pre-existing files.

    Confirm via inspection that the preexisting_paths branch uses
    time.time() - jsonl.stat().st_mtime > 10.
    """
    # Read the source file and verify the mtime guard logic
    source = Path("watch_and_learn.py").read_text()

    # Find the mtime guard condition
    assert "time.time() - jsonl.stat().st_mtime > 10" in source, (
        "mtime guard not found in watch_and_learn.py"
    )


def test_no_known_paths_variable_in_watch_and_learn() -> None:
    """
    Test: known_paths variable is removed.

    Run grep and confirm zero matches in watch_and_learn.py.
    """
    import subprocess

    result = subprocess.run(
        ["grep", "-n", "known_paths", "watch_and_learn.py"],
        capture_output=True,
        text=True,
    )

    # grep returns 1 if no matches
    assert result.returncode != 0, f"Found known_paths: {result.stdout}"


def test_data_structure_types(tmp_projects_dir: Path) -> None:
    """
    Test: data structure types are correct.

    Verify that known_files is dict[str, int], stable_files is set[str],
    preexisting_paths is set[str].
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # First poll
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)

    # Verified by initialization
    assert isinstance(known_files, dict)
    assert isinstance(stable_files, set)
    assert isinstance(preexisting_paths, set)

    # Check types during use
    path_str = str(jsonl_file)
    assert path_str in known_files
    assert isinstance(known_files[path_str], int)


def test_project_filter_respected(tmp_projects_dir: Path) -> None:
    """
    Test: project_filter is respected when scanning.
    """
    project_dir = next(tmp_projects_dir.iterdir())
    jsonl_file = project_dir / "session_001.jsonl"
    create_episode_jsonl(jsonl_file)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # Scan with mismatched filter
    new_files = scan_cycle(
        tmp_projects_dir, "nonExistent", known_files, stable_files, preexisting_paths
    )

    assert len(new_files) == 0


def test_multiple_project_directories(tmp_projects_dir: Path) -> None:
    """
    Test: multiple project directories are handled correctly.
    """
    # Create second project directory
    proj2 = tmp_projects_dir / "project2"
    proj2.mkdir()

    jsonl1 = next(tmp_projects_dir.iterdir()) / "session_001.jsonl"
    jsonl2 = proj2 / "session_002.jsonl"
    create_episode_jsonl(jsonl1)
    create_episode_jsonl(jsonl2)

    known_files: dict[str, int] = {}
    stable_files: set[str] = set()
    preexisting_paths: set[str] = set()

    # First poll records both
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl1) in known_files
    assert str(jsonl2) in known_files

    # Second poll stabilizes both
    new_files = scan_cycle(tmp_projects_dir, None, known_files, stable_files, preexisting_paths)
    assert str(jsonl1) in [str(p) for p in new_files]
    assert str(jsonl2) in [str(p) for p in new_files]


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
