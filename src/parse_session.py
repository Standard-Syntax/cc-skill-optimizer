"""
parse_session.py
================
Reads Claude Code JSONL session logs from ~/.claude/projects/**/*.jsonl
and extracts structured task episodes for use as GEPA training/validation data.

Each episode = one user task request + the full agent trace + outcome signal.

Output schema (list of dicts):
    {
        "session_id": str,
        "task_prompt":  str,          # first user message initiating the task
        "tool_calls":   list[dict],   # [{tool, input, result, success}, ...]
        "assistant_text": list[str],  # all assistant text blocks
        "outcome":      str,          # "success" | "error" | "interrupted" | "unknown"
        "neutral_closing": bool,      # True if primary outcome was "unknown" but secondary
                                      # heuristic (no errors + files written) fired.
                                      # Evaluators should map this to a higher score (~0.7)
                                      # than plain "unknown" (0.5).
        "error_messages": list[str],  # stderr / error fields from tool_results
        "bash_commands": list[str],   # every bash command executed
        "files_read":   list[str],    # every file path read
        "files_written": list[str],   # every file path written/edited
        "thinking_blocks": list[str], # extended thinking content (if present)
        "compaction_summary": str | None,  # context compaction summary if hit
        "token_stats":  dict,         # input/output/cache tokens
        "duration_s":   float | None, # wall-clock seconds for the episode
        "skill_injections": list[str],# skill files injected at session start
        "raw_lines":    int,          # total JSONL lines in session
    }
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_DIR = Path.home() / ".claude"

# Positive completion signals for conservative outcome inference
POSITIVE_COMPLETION_SIGNALS = (
    "done",
    "complete",
    "completed",
    "finished",
    "success",
    "succeeded",
    "✓",
    "implemented",
    "created",
    "added",
    "fixed",
    "wrote",
    "updated",
    "all set",
    "shipped",
    "ready",
    "passed",
    "deployed",
)


def iter_session_files(
    claude_dir: Path = DEFAULT_CLAUDE_DIR,
    project_filter: str | None = None,
) -> Iterator[Path]:
    """Yield every *.jsonl session file under claude_dir/projects/."""
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        if project_filter and project_filter not in project_dir.name:
            continue
        for jsonl in sorted(project_dir.glob("*.jsonl")):
            yield jsonl
        # subagents
        subagents = project_dir / "subagents"
        if subagents.exists():
            for jsonl in sorted(subagents.glob("*.jsonl")):
                yield jsonl


# ---------------------------------------------------------------------------
# Low-level JSONL reading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    lines = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                lines.append(json.loads(raw))
    return lines


# ---------------------------------------------------------------------------
# Content-block helpers
# ---------------------------------------------------------------------------


def _text_from_content(content: Any) -> str:
    """Extract plain text from a message content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("type", "")
                if t == "text":
                    parts.append(block.get("text", ""))
                elif t == "thinking":
                    parts.append(f"<thinking>{block.get('thinking', '')}</thinking>")
        return "\n".join(p for p in parts if p)
    return ""


def _tool_uses_from_content(content: Any) -> list[dict]:
    """Return list of tool_use blocks from content."""
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _tool_results_from_content(content: Any) -> list[dict]:
    """Return list of tool_result blocks from user messages."""
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def _thinking_from_content(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        b.get("thinking", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "thinking" and b.get("thinking")
    ]


# ---------------------------------------------------------------------------
# Session-level parsing
# ---------------------------------------------------------------------------


def parse_session(path: Path) -> dict:
    entries = read_jsonl(path)
    if not entries:
        return _empty_episode(str(path.stem), path)

    # -----------------------------------------------------------------------
    # Deduplication — Claude Code 2.1+ writes each tool call as its own
    # assistant entry, all sharing the same message.id (one logical turn split
    # across multiple JSONL lines).  The old dedup-by-message-id approach kept
    # only the first entry (typically the thinking block) and discarded all the
    # tool_use entries that followed.
    #
    # New approach: deduplicate only on entry.uuid (the per-line identifier),
    # not on message.id.  Two entries are truly duplicate only when they have
    # the same uuid.  Multiple entries sharing a message.id are intentional
    # and must all be kept.
    # -----------------------------------------------------------------------
    seen_uuids: set[str] = set()
    deduped: list[dict] = []
    for entry in entries:
        uid = entry.get("uuid")
        if uid:
            if uid in seen_uuids:
                continue
            seen_uuids.add(uid)
        deduped.append(entry)
    entries = deduped

    session_id = entries[0].get("sessionId", path.stem)

    # Collect data across all entries
    task_prompt: str = ""
    assistant_texts: list[str] = []
    thinking_blocks: list[str] = []
    tool_calls: list[dict] = []
    error_messages: list[str] = []
    bash_commands: list[str] = []
    files_read: list[str] = []
    files_written: list[str] = []
    skill_injections: list[str] = []
    compaction_summary: str | None = None
    token_stats: dict = {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0}
    seen_message_ids: set[str] = set()
    timestamps: list[datetime] = []
    outcome = "unknown"

    # -----------------------------------------------------------------------
    # Build tool_use_id -> result map.
    #
    # Claude Code 2.1+ stores tool results in TWO possible places:
    #   (a) Legacy: nested inside user message.content[] as tool_result blocks
    #   (b) New:    top-level entry.toolUseResult dict on the user entry
    # We handle both.
    # -----------------------------------------------------------------------
    tool_results_by_id: dict[str, dict] = {}

    for entry in entries:
        if entry.get("type") != "user":
            continue

        # (a) Legacy nested tool_result blocks in message.content
        content = entry.get("message", {}).get("content", [])
        for tr in _tool_results_from_content(content):
            tid = tr.get("tool_use_id", "")
            if tid:
                tool_results_by_id[tid] = tr

        # (b) New top-level toolUseResult field
        tur = entry.get("toolUseResult")
        if isinstance(tur, dict):
            # toolUseResult shape: {"toolUseId": "...", "content": [...], "isError": bool}
            tid = tur.get("toolUseId", "")
            if tid:
                # Normalise to the same shape as a legacy tool_result block
                tool_results_by_id[tid] = {
                    "tool_use_id": tid,
                    "content": tur.get("content", ""),
                    "is_error": tur.get("isError", False),
                }

    # Second pass: main parse
    first_user_seen = False
    for entry in entries:
        ts_str = entry.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamps.append(ts)
            except ValueError:
                pass

        etype = entry.get("type", "")
        msg = entry.get("message", {})
        content = msg.get("content", [])

        # ---- USER turn ----
        if etype == "user":
            text = _text_from_content(content)

            # Detect skill injections (system-injected, not real user messages)
            if "Base directory for this skill:" in text:
                skill_injections.append(text[:200])
                continue

            # Detect context compaction
            if "has been summarized to save context" in text.lower() or (
                isinstance(content, str) and "compacted" in content.lower()
            ):
                compaction_summary = text[:500]
                continue

            # First real user message = task prompt
            # Skip slash-command wrappers like <command-name>/plan</command-name>
            if not first_user_seen and text.strip():
                if not text.strip().startswith("<command-"):
                    task_prompt = text.strip()
                    first_user_seen = True
                elif not task_prompt:
                    # Fall back to storing it if it's the only thing we have
                    task_prompt = text.strip()

            # Check for interruption
            if "[Request interrupted" in text:
                outcome = "interrupted"

        # ---- ASSISTANT turn ----
        elif etype == "assistant":
            msg_id = msg.get("id", "")
            if msg_id not in seen_message_ids:
                seen_message_ids.add(msg_id)
                usage = msg.get("usage", {})
                token_stats["input"] += usage.get("input_tokens", 0)
                token_stats["output"] += usage.get("output_tokens", 0)
                token_stats["cache_create"] += usage.get("cache_creation_input_tokens", 0)
                token_stats["cache_read"] += usage.get("cache_read_input_tokens", 0)

            # Text blocks
            text = _text_from_content(content)
            if text.strip():
                assistant_texts.append(text.strip())

            # Thinking
            thinking_blocks.extend(_thinking_from_content(content))

            # Tool calls — each assistant entry now has exactly one content
            # block which may be thinking, text, or tool_use
            for tu in _tool_uses_from_content(content):
                tool_name = tu.get("name", "")
                tool_input = tu.get("input", {})
                tool_id = tu.get("id", "")

                # Get paired result
                result = tool_results_by_id.get(tool_id, {})
                result_content = result.get("content", "")
                result_is_error = result.get("is_error", False)

                # Extract result text — content may be list of blocks or plain string
                if isinstance(result_content, list):
                    result_text = " ".join(
                        b.get("text", "")
                        for b in result_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    result_text = str(result_content or "")

                # Error detection
                if result_is_error or "error" in result_text.lower()[:100]:
                    error_messages.append(result_text[:500])

                # Specialised extraction
                if tool_name in ("Bash", "bash"):
                    cmd = tool_input.get("command", "")
                    if cmd:
                        bash_commands.append(cmd)
                        if (
                            result_is_error or "error" in result_text.lower()[:80]
                        ) and outcome == "unknown":
                            outcome = "error"

                elif tool_name in ("Read", "read"):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        files_read.append(fp)

                elif tool_name in ("Write", "write") or tool_name in ("Edit", "MultiEdit", "edit"):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        files_written.append(fp)

                tool_calls.append(
                    {
                        "tool": tool_name,
                        "input": tool_input,
                        "result": result_text[:1000],
                        "success": not result_is_error,
                        "id": tool_id,
                    }
                )

    # Secondary heuristic flag: if primary inference returns "unknown" but the
    # session shows strong completion signals (no errors, files written), tag
    # it as a "neutral closing" so the evaluator can map to ~0.7 instead of 0.5.
    neutral_closing: bool = False

    # Infer outcome if not already set (conservative — require positive completion signal)
    if outcome == "unknown":
        if error_messages:
            outcome = "error"
        elif assistant_texts:
            last_msg = assistant_texts[-1].lower()
            if any(signal in last_msg for signal in POSITIVE_COMPLETION_SIGNALS):
                outcome = "success"
            elif any(w in last_msg for w in ("error", "failed", "cannot", "sorry")):
                outcome = "error"

    # Secondary heuristic: if outcome is still "unknown" but the session shows
    # strong completion signals (no errors, files written), tag as neutral_closing.
    # The evaluator maps "unknown" + neutral_closing=True to ~0.7 (vs 0.5 for plain unknown).
    if outcome == "unknown" and not error_messages and len(files_written) > 0:
        neutral_closing = True

    duration_s: float | None = None
    if len(timestamps) >= 2:
        duration_s = (timestamps[-1] - timestamps[0]).total_seconds()

    # Store first timestamp as ISO string for chronological sorting (task 10.4)
    episode_timestamp: str | None = timestamps[0].isoformat() if timestamps else None

    return {
        "session_id": session_id,
        "task_prompt": task_prompt,
        "tool_calls": tool_calls,
        "assistant_text": assistant_texts,
        "outcome": outcome,
        "neutral_closing": neutral_closing,  # True if secondary heuristic fired
        "error_messages": error_messages,
        "bash_commands": bash_commands,
        "files_read": files_read,
        "files_written": files_written,
        "thinking_blocks": thinking_blocks,
        "compaction_summary": compaction_summary,
        "token_stats": token_stats,
        "duration_s": duration_s,
        "skill_injections": skill_injections,
        "raw_lines": len(entries),
        "source_path": str(path),
        "timestamp": episode_timestamp,
    }


def _empty_episode(session_id: str, path: Path) -> dict:
    return {
        "session_id": session_id,
        "task_prompt": "",
        "tool_calls": [],
        "assistant_text": [],
        "outcome": "unknown",
        "neutral_closing": False,
        "error_messages": [],
        "bash_commands": [],
        "files_read": [],
        "files_written": [],
        "thinking_blocks": [],
        "compaction_summary": None,
        "token_stats": {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0},
        "duration_s": None,
        "skill_injections": [],
        "raw_lines": 0,
        "source_path": str(path),
        "timestamp": None,
    }


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------


def build_corpus(
    claude_dir: Path = DEFAULT_CLAUDE_DIR,
    project_filter: str | None = None,
    min_tool_calls: int = 2,
    skip_empty_prompts: bool = True,
    skip_paths: set[str] | None = None,
    sort_by_time: bool = False,
) -> list[dict]:
    """
    Load all sessions matching filters and return the parsed episode list.

    Args:
        claude_dir:        Root of .claude/ directory.
        project_filter:    Optional substring to filter project directory names.
        min_tool_calls:    Discard sessions with fewer tool calls (too trivial).
        skip_empty_prompts: Discard sessions with no recoverable task prompt.
        skip_paths:        Optional set of absolute path strings to skip. Use this
                          to avoid re-parsing files that have already been processed
                          (e.g. by watch_and_learn.py's cumulative known_files set).
                          Callers should pass the cumulative set across all
                          watch sessions to ensure files are parsed exactly once.
        sort_by_time:      If True, sort episodes chronologically by timestamp
                          before returning (task 10.4). Episodes without a
                          timestamp are sorted to the end.
    """
    episodes: list[dict] = []
    skipped_count = 0
    for path in iter_session_files(claude_dir, project_filter):
        if skip_paths is not None and str(path) in skip_paths:
            skipped_count += 1
            continue
        ep = parse_session(path)
        if skip_empty_prompts and not ep["task_prompt"]:
            continue
        if len(ep["tool_calls"]) < min_tool_calls:
            continue
        episodes.append(ep)

    skip_msg = f" (skipped {skipped_count} already-processed)" if skipped_count else ""
    print(
        f"[parse_session] Loaded {len(episodes)} episodes from {claude_dir}{skip_msg}",
        file=sys.stderr,
    )

    if sort_by_time:
        episodes = sort_episodes_by_timestamp(episodes)

    return episodes


def sort_episodes_by_timestamp(corpus: list[dict]) -> list[dict]:
    """
    Sort episodes chronologically by timestamp (task 10.4).

    Episodes without a timestamp are sorted to the end.
    Returns a new list (does not mutate the input).

    Args:
        corpus: List of episode dicts, each with an optional "timestamp" field
                (ISO format string or None).
    """
    with_ts = [e for e in corpus if e.get("timestamp")]
    without_ts = [e for e in corpus if not e.get("timestamp")]
    with_ts.sort(key=lambda e: e["timestamp"])
    return with_ts + without_ts


# ---------------------------------------------------------------------------
# ASI builder — converts an episode into diagnostic text for GEPA reflection
# ---------------------------------------------------------------------------


def episode_to_asi(ep: dict) -> str:
    """
    Build Actionable Side Information from a parsed episode.
    This is what the GEPA reflection LM reads to diagnose failures.

    Section order is optimized for truncation safety (after Phase 16.1 the
    episode cap is 4000 chars; the LLM judge still slices with [:4000], so
    content at the end is at risk of being cut). Highest-signal sections
    come first to ensure they're always preserved.

    Order (top → bottom):
      1. Outcome           — single line, fastest failure-mode classifier
      2. Errors            — concrete error messages, prevent recurrence
      3. Task              — the actual user request being optimized for
      4. Final assistant   — the most diagnostic single message
      5. Duration          — session-length context
      6. Bash commands     — what the agent actually ran
      7. Files touched     — read + written, combined for brevity
      8. Context compaction — verbose; only present when it happened
      9. Token usage       — diagnostic for cost/context pressure
     10. Tool sequence     — bulkier; safe to truncate to the last 30 tools
    """
    parts: list[str] = []

    # 1. Outcome — always present; fastest signal
    parts.append(f"## Outcome: {ep['outcome'].upper()}")

    # 2. Errors — concrete, actionable, prevents same failure next iteration
    if ep["error_messages"]:
        parts.append("## Errors\n" + "\n".join(f"- {e[:300]}" for e in ep["error_messages"][:5]))

    # 3. Task — the user request
    parts.append(f"## Task\n{ep['task_prompt'][:600]}")

    # 4. Final assistant message — most diagnostic single message
    if ep["assistant_text"]:
        last = ep["assistant_text"][-1][:800]
        parts.append(f"## Final assistant message\n{last}")

    # 5. Duration — one line, useful for efficiency tuning
    if ep["duration_s"] is not None:
        parts.append(f"## Duration: {ep['duration_s']:.1f}s")

    # 6. Bash commands — what the agent ran
    if ep["bash_commands"]:
        parts.append(
            "## Bash commands executed\n"
            + "\n".join(f"  $ {c[:200]}" for c in ep["bash_commands"][:15])
        )

    # 7. Files touched — read + written combined for brevity
    files_touched: list[str] = []
    if ep["files_read"]:
        files_touched.extend(f"read: {f}" for f in ep["files_read"][:10])
    if ep["files_written"]:
        files_touched.extend(f"wrote: {f}" for f in ep["files_written"][:10])
    if files_touched:
        parts.append("## Files touched\n" + "\n".join(files_touched))

    # 8. Context compaction — verbose; only when it happened
    if ep["compaction_summary"]:
        parts.append(f"## Context compaction occurred\n{ep['compaction_summary'][:300]}")

    # 9. Token usage
    tok = ep["token_stats"]
    parts.append(
        f"## Token usage: input={tok['input']} output={tok['output']} "
        f"cache_create={tok['cache_create']} cache_read={tok['cache_read']}"
    )

    # 10. Tool sequence — bulkier; safe to truncate to the last 30 tools
    tool_seq = [tc["tool"] for tc in ep["tool_calls"][:30]]
    parts.append(f"## Tool sequence ({len(ep['tool_calls'])} total)\n{' → '.join(tool_seq)}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Parse Claude Code session logs")
    ap.add_argument("--claude-dir", default=str(DEFAULT_CLAUDE_DIR))
    ap.add_argument("--project-filter", default=None)
    ap.add_argument("--min-tool-calls", type=int, default=2)
    ap.add_argument("--output", default="-", help="Output path (- for stdout)")
    args = ap.parse_args()

    corpus = build_corpus(
        Path(args.claude_dir),
        args.project_filter,
        args.min_tool_calls,
    )

    if args.output == "-":
        for ep in corpus:
            # Strip heavy fields for CLI preview
            preview = {k: v for k, v in ep.items() if k not in ("tool_calls", "assistant_text")}
            sys.stdout.write(_json.dumps(preview) + "\n")
    else:
        with open(args.output, "w") as out:
            for ep in corpus:
                # Strip heavy fields for CLI preview
                preview = {k: v for k, v in ep.items() if k not in ("tool_calls", "assistant_text")}
                out.write(_json.dumps(preview) + "\n")
        print(f"Wrote {len(corpus)} episodes to {args.output}")
