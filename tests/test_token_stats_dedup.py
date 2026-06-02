"""
Test parse_session token_stats deduplication.

Verifies that when Claude Code 2.1+ splits a single logical assistant turn
across multiple JSONL lines (same message.id, different uuid), token_stats
are NOT double-counted. Also verifies normal single-entry turns work.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure src/ is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parse_session import parse_session


def _write_jsonl(entries: list[dict]) -> Path:
    """Write entries to a temp JSONL file and return the path."""
    fp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    for e in entries:
        fp.write(json.dumps(e) + "\n")
    fp.close()
    return Path(fp.name)


def _make_assistant_entry(
    msg_id: str,
    entry_uuid: str,
    content: list[dict],
    usage: dict | None = None,
) -> dict:
    """Make a minimal assistant JSONL entry."""
    return {
        "type": "assistant",
        "uuid": entry_uuid,
        "sessionId": "test-session",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {
            "id": msg_id,
            "content": content,
            "usage": usage
            or {
                "input_tokens": 100,
                "output_tokens": 200,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 25,
            },
        },
    }


def _make_user_entry(entry_uuid: str, text: str) -> dict:
    """Make a minimal user JSONL entry."""
    return {
        "type": "user",
        "uuid": entry_uuid,
        "sessionId": "test-session",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {
            "content": text,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_split_turn_deduplicates_token_stats():
    """
    Claude Code 2.1+ splits one logical assistant turn into multiple entries
    that share the same message.id but have different entry.uuids.
    token_stats must count the usage tokens ONLY ONCE.
    """
    # First user message to establish task_prompt
    user1 = _make_user_entry("user-1", "Fix the bug in login")

    # Assistant entry with thinking block
    assistant1 = _make_assistant_entry(
        msg_id="msg-abc",
        entry_uuid="entry-1",
        content=[{"type": "thinking", "thinking": "I should check the code..."}],
    )

    # Same message.id but different entry.uuid (tool_use entry split from same turn)
    assistant2 = _make_assistant_entry(
        msg_id="msg-abc",
        entry_uuid="entry-2",
        content=[
            {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "login.py"}}
        ],
    )

    # Another split entry for the tool result
    user2 = _make_user_entry("user-2", "Here is the file content...")
    user2["toolUseResult"] = {
        "toolUseId": "tool-1",
        "content": "file content here",
        "isError": False,
    }

    # Third assistant entry sharing same message.id (continuation of same turn)
    assistant3 = _make_assistant_entry(
        msg_id="msg-abc",
        entry_uuid="entry-3",
        content=[{"type": "text", "text": "I found the issue."}],
    )

    # Final assistant entry with different message.id (new turn)
    assistant4 = _make_assistant_entry(
        msg_id="msg-def",
        entry_uuid="entry-4",
        content=[{"type": "text", "text": "Done fixing the bug."}],
    )

    fp = _write_jsonl([user1, assistant1, assistant2, user2, assistant3, assistant4])
    try:
        ep = parse_session(fp)

        # msg-abc appeared 3 times but token_stats must count it only once
        # Each entry's usage: input=100, output=200, cache_create=50, cache_read=25
        # msg-abc (counted once): input=100, output=200, cache_create=50, cache_read=25
        # msg-def (counted once): input=100, output=200, cache_create=50, cache_read=25
        # Total expected: input=200, output=400, cache_create=100, cache_read=50
        assert ep["token_stats"]["input"] == 200, (
            f"input tokens: expected 200, got {ep['token_stats']['input']}"
        )
        assert ep["token_stats"]["output"] == 400, (
            f"output tokens: expected 400, got {ep['token_stats']['output']}"
        )
        assert ep["token_stats"]["cache_create"] == 100, (
            f"cache_create: expected 100, got {ep['token_stats']['cache_create']}"
        )
        assert ep["token_stats"]["cache_read"] == 50, (
            f"cache_read: expected 50, got {ep['token_stats']['cache_read']}"
        )
    finally:
        fp.unlink()


def test_single_entry_turn_counts_tokens():
    """
    Normal case: one assistant entry with one message.id → tokens counted once.
    """
    user1 = _make_user_entry("user-1", "Write a test")

    assistant1 = _make_assistant_entry(
        msg_id="msg-xyz",
        entry_uuid="entry-1",
        content=[{"type": "text", "text": "Here is the test."}],
        usage={
            "input_tokens": 50,
            "output_tokens": 75,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
        },
    )

    fp = _write_jsonl([user1, assistant1])
    try:
        ep = parse_session(fp)

        assert ep["token_stats"]["input"] == 50
        assert ep["token_stats"]["output"] == 75
        assert ep["token_stats"]["cache_create"] == 10
        assert ep["token_stats"]["cache_read"] == 5
    finally:
        fp.unlink()


def test_multiple_distinct_turns_count_separately():
    """
    Two distinct assistant turns (different message.ids) → both counted.
    """
    user1 = _make_user_entry("user-1", "First task")
    assistant1 = _make_assistant_entry(
        msg_id="msg-1",
        entry_uuid="entry-1",
        content=[{"type": "text", "text": "First response"}],
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 2,
        },
    )

    user2 = _make_user_entry("user-2", "Second task")
    assistant2 = _make_assistant_entry(
        msg_id="msg-2",
        entry_uuid="entry-2",
        content=[{"type": "text", "text": "Second response"}],
        usage={
            "input_tokens": 30,
            "output_tokens": 40,
            "cache_creation_input_tokens": 15,
            "cache_read_input_tokens": 8,
        },
    )

    fp = _write_jsonl([user1, assistant1, user2, assistant2])
    try:
        ep = parse_session(fp)

        assert ep["token_stats"]["input"] == 40  # 10 + 30
        assert ep["token_stats"]["output"] == 60  # 20 + 40
        assert ep["token_stats"]["cache_create"] == 20  # 5 + 15
        assert ep["token_stats"]["cache_read"] == 10  # 2 + 8
    finally:
        fp.unlink()


def test_assistant_without_message_id_does_not_crash():
    """
    Edge case: assistant entry with no message.id field → must not crash,
    and must still accumulate tokens (using empty string as key).
    """
    user1 = _make_user_entry("user-1", "Hello")
    assistant1 = _make_user_entry("user-2", "oops")  # reuse helper but type should be assistant
    assistant1["type"] = "assistant"
    assistant1["message"]["id"] = None  # Explicitly no id
    assistant1["message"]["usage"] = {
        "input_tokens": 99,
        "output_tokens": 88,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    fp = _write_jsonl([user1, assistant1])
    try:
        ep = parse_session(fp)
        # With no msg_id (None → ""), each such entry is counted
        assert ep["token_stats"]["input"] == 99
    finally:
        fp.unlink()


def test_empty_session():
    """Empty JSONL file → returns empty episode with zero token_stats."""
    fp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    fp.close()
    fp_path = Path(fp.name)
    try:
        ep = parse_session(fp_path)
        assert ep["token_stats"]["input"] == 0
        assert ep["token_stats"]["output"] == 0
        assert ep["token_stats"]["cache_create"] == 0
        assert ep["token_stats"]["cache_read"] == 0
    finally:
        fp_path.unlink(missing_ok=True)
