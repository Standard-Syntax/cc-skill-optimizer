"""
Drop this into ~/Repo/cc-skill-optimizer and run:
    uv run python inspect_jsonl.py <path-to-jsonl>

It shows every unique entry structure found in the file so we can
see exactly where tool calls are stored.
"""

import collections
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

print(f"File: {path.name}  ({len(lines)} lines)\n")

# Group by type and show one full example of each type
by_type = collections.defaultdict(list)
for entry in lines:
    by_type[entry.get("type", "MISSING")].append(entry)

for t, entries in sorted(by_type.items(), key=lambda x: -len(x[1])):
    print(f"{'=' * 60}")
    print(f"type={t!r}  ({len(entries)} entries)")
    print(f"{'=' * 60}")

    # Show the keys present across all entries of this type
    all_keys = set()
    for e in entries:
        all_keys.update(e.keys())
    print(f"  Top-level keys: {sorted(all_keys)}")

    # Show one full example (truncated)
    ex = entries[0]

    # Special handling: show content blocks if present
    msg = ex.get("message", {})
    if msg:
        print(f"  message keys: {sorted(msg.keys())}")
        content = msg.get("content", [])
        if isinstance(content, list):
            print(f"  content blocks ({len(content)}):")
            for i, block in enumerate(content[:5]):
                if isinstance(block, dict):
                    btype = block.get("type", "?")
                    bkeys = sorted(block.keys())
                    # For tool_use blocks, show name and input keys
                    if btype == "tool_use":
                        print(
                            f"    [{i}] type=tool_use  name={block.get('name')!r}  id={block.get('id', '')[:12]}..."
                        )
                        inp = block.get("input", {})
                        print(
                            f"         input keys: {sorted(inp.keys()) if isinstance(inp, dict) else type(inp)}"
                        )
                    else:
                        preview = str(block.get("text", block))[:80]
                        print(f"    [{i}] type={btype!r}  keys={bkeys}  preview={preview!r}")
        elif isinstance(content, str):
            print(f"  content (str): {content[:120]!r}")

    # For non-message entries, show raw (truncated)
    non_msg_keys = {
        k: v
        for k, v in ex.items()
        if k not in ("type", "timestamp", "sessionId", "message", "uuid")
    }
    if non_msg_keys:
        raw = json.dumps(non_msg_keys, default=str)[:300]
        print(f"  other fields: {raw}")

    print()

# Summary: look for tool_use anywhere in the whole file
print("=" * 60)
print("TOOL_USE SEARCH — scanning every field of every entry")
print("=" * 60)
found = []


def find_tool_use(obj, path=""):
    if isinstance(obj, dict):
        if obj.get("type") == "tool_use":
            found.append((path, obj.get("name"), obj.get("id", "")))
        for k, v in obj.items():
            find_tool_use(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_tool_use(v, f"{path}[{i}]")


for entry in lines:
    find_tool_use(entry)

if found:
    print(f"Found {len(found)} tool_use blocks:")
    for path, name, tid in found[:10]:
        print(f"  path={path}  name={name!r}  id={tid[:12]}")
else:
    print("NO tool_use blocks found anywhere in the file.")
    print()
    print("Checking for 'tool' keyword anywhere...")
    tool_mentions = []

    def find_tool_mentions(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "tool" in k.lower():
                    tool_mentions.append((f"{path}.{k}", str(v)[:100]))
                find_tool_mentions(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                find_tool_mentions(v, f"{path}[{i}]")
        elif isinstance(obj, str) and "tool_use" in obj:
            tool_mentions.append((path, obj[:100]))

    for entry in lines:
        find_tool_mentions(entry)

    if tool_mentions:
        print(f"  Found {len(tool_mentions)} 'tool' references:")
        seen = set()
        for path, val in tool_mentions[:15]:
            key = path.split(".")[-1]
            if key not in seen:
                seen.add(key)
                print(f"    {path}: {val!r}")
    else:
        print("  No 'tool' references anywhere. These may be pure chat sessions.")
