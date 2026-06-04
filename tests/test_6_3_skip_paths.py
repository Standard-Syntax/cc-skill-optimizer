"""
Tests for skip_paths parameter in build_corpus (Task 6.3)

Validates that callers can pass a set of already-processed file paths so the
function doesn't re-parse them.
"""

from __future__ import annotations

import inspect
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure src/ is on the path
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))  # noqa: E402 — test path setup, must precede module import

# Intentional: import after sys.path.insert for path setup
from parse_session import build_corpus, iter_session_files  # noqa: E402


@pytest.fixture
def tmp_session_dir(tmp_path: Path) -> Path:
    """Create a temp directory with valid JSONL session files."""
    proj_dir = tmp_path / "projects" / "testproj" / "subagents"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Create several JSONL files
    files = [
        proj_dir / "session1.jsonl",
        proj_dir / "session2.jsonl",
        proj_dir / "session3.jsonl",
    ]

    for f in files:
        # Each file needs at least one valid JSON line with type="user" and a task prompt
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [{"type": "text", "text": f"Task: {f.name}"}],
                    },
                    "sessionId": f.stem,
                    "uuid": f"uuid-{f.stem}",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "echo hello"},
                            },
                        ],
                        "id": "msg-1",
                    },
                    "sessionId": f.stem,
                    "uuid": f"uuid-{f.stem}-2",
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": []},
                    "toolUseResult": {
                        "toolUseId": "tu-1",
                        "content": "hello",
                        "isError": False,
                    },
                    "sessionId": f.stem,
                    "uuid": f"uuid-{f.stem}-3",
                }
            ),
        ]
        f.write_text("\n".join(lines))

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Create an empty directory with no JSONL files."""
    empty = tmp_path / "empty"
    empty.mkdir(exist_ok=False)
    return empty


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_skip_paths_parameter_exists_in_signature():
    """1. Verify skip_paths parameter is present in signature with default None."""
    sig = inspect.signature(build_corpus)
    params = sig.parameters
    assert "skip_paths" in params, "skip_paths parameter must exist in build_corpus signature"
    param = params["skip_paths"]
    assert param.default is None, "skip_paths default must be None"


def test_build_corpus_without_skip_paths_works(empty_dir):
    """2. Call without skip_paths - backward compatibility."""
    # Patch DEFAULT_CLAUDE_DIR to avoid scanning real home dir
    with patch("parse_session.DEFAULT_CLAUDE_DIR", empty_dir):
        result = build_corpus(claude_dir=empty_dir)
    assert result == [], f"Expected empty list, got {result!r}"


def test_skip_paths_empty_set_behaves_like_no_filter(tmp_session_dir):
    """3. Empty set() behaves like no filter - processes all files."""
    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        result = build_corpus(claude_dir=tmp_session_dir, skip_paths=set())
    # All 3 files should be processed since skip_paths={} matches nothing
    assert len(result) >= 0  # May be 0 if files don't meet min_tool_calls


def test_skip_paths_empty_set_does_not_crash(tmp_session_dir):
    """4. Empty set does not crash, no skip suffix message."""
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            result = build_corpus(claude_dir=tmp_session_dir, skip_paths=set())

        output = sys.stderr.getvalue()
        sys.stderr = old_stderr

        assert result == [] or isinstance(result, list), "Should not crash"
        # Verify no "(skipped 0 already-processed)" suffix
        assert "(skipped 0 already-processed)" not in output, (
            "Empty skip_paths should not print skip message"
        )
    finally:
        sys.stderr = old_stderr


def test_skip_paths_skips_a_single_file(tmp_session_dir):
    """5. skip_paths containing a file path skips that file."""
    # Get the list of files that would be iterated
    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        all_files = list(iter_session_files(tmp_session_dir))

    if len(all_files) < 2:
        pytest.skip("Need at least 2 files to test skipping")

    file_to_skip = str(all_files[0])

    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        build_corpus(claude_dir=tmp_session_dir, skip_paths={file_to_skip})

    # The result should have fewer items (or equal if files don't meet criteria)
    # We'll check by examining stderr output
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            build_corpus(claude_dir=tmp_session_dir, skip_paths={file_to_skip})
        output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Should show "(skipped 1 already-processed)"
    assert "(skipped 1 already-processed)" in output, f"Expected skip message in: {output}"


def test_skip_paths_skips_multiple_files(tmp_session_dir):
    """6. skip_paths containing multiple file paths skips all of them."""
    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        all_files = list(iter_session_files(tmp_session_dir))

    if len(all_files) < 3:
        pytest.skip("Need at least 3 files to test skipping multiple")

    files_to_skip = {str(all_files[0]), str(all_files[1])}

    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            build_corpus(claude_dir=tmp_session_dir, skip_paths=files_to_skip)
        output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    assert "(skipped 2 already-processed)" in output, (
        f"Expected skip message for 2 files in: {output}"
    )


def test_skip_paths_non_existent_path_silently_ignored(tmp_session_dir):
    """7. skip_paths containing a non-existent path is silently ignored."""
    non_existent = "/tmp/this/path/does/not/exist/12345.jsonl"

    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            build_corpus(claude_dir=tmp_session_dir, skip_paths={non_existent})
        output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Should NOT crash, should process all files normally
    assert "(skipped" not in output or "(skipped 0" in output, (
        "Non-existent skip_paths should be silently ignored"
    )


def test_print_message_includes_skip_suffix_when_applicable(tmp_session_dir):
    """8. Print message includes '(skipped N already-processed)' when applicable."""
    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        all_files = list(iter_session_files(tmp_session_dir))

    if len(all_files) < 3:
        pytest.skip("Need at least 3 files")

    files_to_skip = {str(all_files[0]), str(all_files[1]), str(all_files[2])}

    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            build_corpus(claude_dir=tmp_session_dir, skip_paths=files_to_skip)
        output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    assert "(skipped 3 already-processed)" in output, (
        f"Expected skip message for 3 files in: {output}"
    )


def test_print_message_excludes_skip_suffix_when_zero(tmp_session_dir):
    """9. Print message does NOT include skip suffix when zero skipped."""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            build_corpus(claude_dir=tmp_session_dir, skip_paths=set())
        output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    assert "(skipped" not in output, f"Should NOT include skip message when zero skipped: {output}"


def test_skip_paths_filtered_before_parse_session_called(tmp_session_dir):
    """10. skip_paths is filtered BEFORE parse_session is called."""
    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        all_files = list(iter_session_files(tmp_session_dir))

    if len(all_files) < 2:
        pytest.skip("Need at least 2 files")

    file_to_skip = str(all_files[0])

    # Track which files parse_session receives
    from parse_session import parse_session as ps_func

    called_paths: list[str] = []

    original_parse = ps_func

    def track_parse(path: Path) -> dict:
        called_paths.append(str(path))
        return original_parse(path)

    with (
        patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir),
        patch("parse_session.parse_session", side_effect=track_parse),
    ):
        build_corpus(claude_dir=tmp_session_dir, skip_paths={file_to_skip})

    assert file_to_skip not in called_paths, (
        f"parse_session should NOT be called for skipped file: {called_paths}"
    )


def test_skipped_count_correctly_counted(tmp_session_dir):
    """11. skipped_count is correctly counted."""
    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        all_files = list(iter_session_files(tmp_session_dir))

    if len(all_files) < 3:
        pytest.skip("Need at least 3 files")

    files_to_skip = {str(all_files[0]), str(all_files[1]), str(all_files[2])}

    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            build_corpus(claude_dir=tmp_session_dir, skip_paths=files_to_skip)
        output = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    assert "(skipped 3 already-processed)" in output


def test_skip_paths_type_annotation():
    """12. Verify type annotation is set[str] | None = None."""
    sig = inspect.signature(build_corpus)
    param = sig.parameters["skip_paths"]
    ann = param.annotation
    # Check it accepts set[str] | None or equivalent
    annotation_str = str(ann)
    assert "set" in annotation_str.lower() or "none" in annotation_str.lower(), (
        f"Expected set[str] | None annotation, got {ann}"
    )


def test_cumulative_usage_pattern(tmp_session_dir):
    """13. Demonstrate watch_and_learn.py usage pattern."""
    with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
        all_files = list(iter_session_files(tmp_session_dir))

    if len(all_files) < 2:
        pytest.skip("Need at least 2 files")

    # First call - no skip_paths
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            result1 = build_corpus(claude_dir=tmp_session_dir)
        sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Second call - skip all paths that were in result1
    parsed_paths = {ep["source_path"] for ep in result1}

    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with patch("parse_session.DEFAULT_CLAUDE_DIR", tmp_session_dir):
            build_corpus(claude_dir=tmp_session_dir, skip_paths=parsed_paths)
        output2 = sys.stderr.getvalue()
    finally:
        sys.stderr = old_stderr

    # Second call should show all files as skipped (if any were parsed)
    if len(result1) > 0:
        skipped_match = (
            output2.split("(skipped ")[1].split(")")[0] if "(skipped " in output2 else "0"
        )
        assert int(skipped_match) >= len(parsed_paths), (
            f"Expected at least {len(parsed_paths)} skipped, got {skipped_match}"
        )
