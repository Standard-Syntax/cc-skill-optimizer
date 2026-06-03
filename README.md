# cc-skill-optimizer

**Optimize Claude Code `SKILL.md`, `CLAUDE.md`, and `AGENTS.md` files using GEPA and DSPy — powered by your own session logs.**

```
Claude Code sessions (~/.claude/projects/**/*.jsonl)
        ↓  parse_session.py
        ↓
    Structured episode corpus (task, tool calls, outcome, ASI)
        ↓
GEPA optimize_anything  ←→  reflection LM reads traces, proposes fixes
        ↓
Optimized SKILL.md / CLAUDE.md / AGENTS.md
        ↓  write to .claude/skills/<repo>/SKILL.md
        ↓
Better future sessions → more data → continuous loop
```

GEPA (Genetic-Pareto) uses Reflective Text Evolution: instead of just knowing *that* a session failed, it reads the full execution trace — error messages, tool sequences, duration, compaction events — to diagnose *why*, then proposes targeted fixes. **55% → 82% resolve rate** on Jinja, **79.3% → 100%** on Bleve with Claude Code Haiku (per the gskill paper).

---

## Installation

```bash
# Clone this repo next to your project
cd ~/code
git clone [email protected]:Standard-Syntax/cc-skill-optimizer.git
cd cc-skill-optimizer

# Install with uv
uv sync

# Or with pip
pip install gepa dspy-ai litellm anthropic rich
```

---

## Quick Start

### 1. One-shot optimization (recommended first run)

```bash
# Optimize your banking analytics SKILL.md from real session logs
uv run python optimize.py \
    --target skill \
    --seed-file skills/banking-analytics-seed.md \
    --project-filter banking \
    --max-evals 150 \
    --phase 1 \
    --reflection-lm minimax/minimax-m3 \
    --output-dir outputs/
```

**Phases:**
- `--phase 1` (default): Synthetic exploration mode, 100 evals, 4-thread parallel evaluation
- `--phase 2`: Session-backed refinement, 60 evals, requires existing session logs

The optimized file lands at `outputs/skill/best_candidate.md`.
Copy it to `.claude/skills/banking/SKILL.md` in your project.

### 2. Continuous improvement daemon

```bash
# Watches for new sessions every 15s, re-optimizes every 25 new episodes
uv run python watch_and_learn.py \
    --skill-file /path/to/project/.claude/skills/banking/SKILL.md \
    --project-filter banking \
    --optimize-every 25 \
    --reflection-lm minimax/minimax-m3

# Run in background on your Debian/niri workstation:
nohup uv run python watch_and_learn.py \
    --skill-file ~/.claude/skills/banking/SKILL.md \
    --project-filter banking &
```

### 3. With DSPy (more powerful program optimization)

```bash
uv run python optimize.py \
    --target skill \
    --use-dspy \
    --seed-file skills/banking-analytics-seed.md \
    --project-filter banking \
    --max-evals 200 \
    --task-lm minimax/minimax-m2.7-highspeed \
    --reflection-lm minimax/minimax-m3
```

### 4. Optimize CLAUDE.md (global project instructions)

```bash
uv run python optimize.py \
    --target claude \
    --seed-file CLAUDE.md \
    --max-evals 100
```

### 5. Optimize AGENTS.md (multi-agent orchestration)

```bash
uv run python optimize.py \
    --target agent \
    --seed-file AGENTS.md \
    --max-evals 80
```

### 6. Optimize WITHIN-FILE SECTIONS (each ## heading = separate component)

```bash
# Optimize sections of CLAUDE.md — each ## heading evolved independently:
uv run python optimize.py \
    --target sections \
    --seed-file CLAUDE.md \
    --project-filter banking \
    --section-depth 2 \
    --max-evals 200

# Include nested subsections (### headings as children of ## sections):
uv run python optimize.py \
    --target sections \
    --seed-file .claude/skills/banking/SKILL.md \
    --section-depth 3 \
    --no-sessions \
    --domain banking \
    --max-evals 150
```

**How section optimization works:**
- Parses the file into sections by `## ` heading (configurable with `--section-depth`)
- Each section becomes its own GEPA component — evolved independently
- Subsections (`###`) nest under their parent `##` heading
- Pareto selection ensures improvements in one section don't break others
- Reconstructed in document order after optimization

### 7. Optimize NESTED files at different directory levels

```bash
# Discover and optimize CLAUDE.md/AGENTS.md files across the project tree:
uv run python optimize.py \
    --target nested \
    --nested-root . \
    --nested-depth 3 \
    --project-filter myproject \
    --max-evals 150

# Zero-session optimization with custom file patterns:
uv run python optimize.py \
    --target nested \
    --nested-root . \
    --nested-patterns CLAUDE.md,AGENTS.md \
    --nested-depth 2 \
    --no-sessions \
    --domain banking \
    --max-evals 100
```

**How nested file optimization works:**
- Discovers all `CLAUDE.md`, `AGENTS.md`, and `SKILL.md` files recursively from `--nested-root`
- Each file becomes an independent GEPA component (e.g., `src/CLAUDE.md`, `test/CLAUDE.md`)
- Root file is optimized for project-wide context (architecture, global conventions)
- Subdirectory files are optimized for area-specific guidance (commands, patterns, pitfalls)
- Pareto selection ensures improvements in one file don't break cross-references
- Directory structure is preserved in output (`outputs/nested/CLAUDE.md`, `outputs/nested/src/CLAUDE.md`)

**Why nested files?**
- Claude Code loads the root `CLAUDE.md` always
- When working in `src/`, it also loads `src/CLAUDE.md` for focused guidance
- This reduces context waste — agents only see relevant rules
- Each file can be concise and focused rather than one massive file

---

## How It Works

### Session Log Parsing (`src/parse_session.py`)

Claude Code writes every session to:
```
~/.claude/projects/<project-slug>/<session-id>.jsonl
~/.claude/projects/<project-slug>/subagents/<agent-id>.jsonl
```

Each JSONL line is one of:
- `{"type": "user", ...}` — your message or tool results
- `{"type": "assistant", ...}` — Claude's response + tool calls + token usage

`parse_session.py` extracts per-episode:
- **task_prompt**: the first real user message
- **tool_calls**: every Bash/Read/Write/Edit call with input+result+success
- **error_messages**: stderr/error fields from failed tool results
- **bash_commands**: every shell command executed
- **outcome**: inferred success/error/interrupted/unknown
- **neutral_closing**: true if outcome=unknown + no errors + files written (~0.7 score)
- **token_stats**: input/output/cache tokens (for efficiency scoring)
- **duration_s**: wall-clock time
- **thinking_blocks**: extended thinking content
- **compaction_summary**: whether context was compacted (session too long)
- **skip_paths**: cumulative set of already-processed files for dedup

### GEPA Evaluator (`src/evaluator.py`)

Two scoring modes:

**Heuristic scoring (default, free)**:
- Base: outcome signal (success=1.0, error=0.0, unknown=0.5)
- neutral_closing bonus: unknown outcome + no errors + files written = ~0.7 (higher-confidence completion)
- Efficiency bonus: fewer tool calls + shorter duration = +0.15 max

**LLM judge scoring (--use-llm-judge)**:
- An LLM reads the candidate skill + episode trace
- Scores 0-1 how much the skill would have helped
- Weighted 65% LLM + 35% heuristic by default (was 40%/60%)
- Full context: 8000 chars (raised from 3000)

### GEPA Optimization (`optimize.py`)

Uses `gepa.optimize_anything` (Generalization mode):
1. **Select** candidate from Pareto frontier (best at different episode subsets)
2. **Execute** evaluator on a minibatch → get (score, ASI)
3. **Reflect** — reflection LM reads the ASI traces and diagnoses why sessions failed
4. **Mutate** — generate improved SKILL.md informed by all accumulated lessons
5. **Accept** — update Pareto front if improved

**Parallel evaluation** is enabled by default (4 threads for GEPA, 4 workers for evaluator batch). Phase 1 uses synthetic exploration, Phase 2 uses session-backed refinement.

The ASI passed to the reflection LM includes:
- Task prompt
- Outcome + duration
- All error messages
- Full bash command sequence
- Tool call sequence
- Final assistant message
- Token stats + compaction events

### Actionable Side Information (ASI)

The key GEPA concept: instead of just a score, the evaluator returns diagnostic text that tells the reflection LM *why* a session failed. For Claude Code:
- "EACCES: permission denied" → skill should warn about permission issues
- "ModuleNotFoundError: litellm" → skill should specify `uv run`
- Context compaction triggered → skill is too verbose, sessions run long
- 47 tool calls for a simple edit → skill missing repo navigation guidance

---

## Configuration

### API Keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# For reflection LM (can be different provider):
export OPENAI_API_KEY="sk-..."      # if using OpenAI models
export GEMINI_API_KEY="..."          # if using Gemini models
```

**Important**: `llm_config.configure()` must be called explicitly before using the optimizer. The module no longer runs as a side effect on import.

```python
import llm_config
llm_config.configure()  # Required: validates API keys
# Now safe to use litellm/gepa
```

litellm handles routing — use any model string it supports.

### Recommended LM combinations

| Use case | task_lm | reflection_lm | cost |
|----------|---------|---------------|------|
| Fast iteration | `minimax/minimax-m2.7-highspeed` | `minimax/minimax-m3` | Low |
| High quality | `minimax/minimax-m3` | `minimax/minimax-m3` | High |
| Cheap reflection | `minimax/minimax-m2.7-highspeed` | `minimax/minimax-m2.7-highspeed` | Very low |
| Skill generation (M2.7) | `minimax/minimax-m2.7` | `minimax/minimax-m3` | Very low ($0.30/M) |

> **MiniMax M2.7** excels at skill generation tasks requiring high adherence (97%) with low hallucination (34%), but is verbose (4× average tokens) and slow (~50 tps). Use M2.7-highspeed for eval/judge calls — never route high-throughput eval traffic through M2.7.

### GEPA budget guidance

| Corpus size | Recommended --max-evals | Expected runtime |
|-------------|------------------------|------------------|
| 20–50 episodes | 60–100 | 10–20 min |
| 50–150 episodes | 100–200 | 20–45 min |
| 150+ episodes | 200–400 | 1–2 hr |

---

## File Structure

```
cc-skill-optimizer/
├── optimize.py              # Main optimization runner
├── watch_and_learn.py       # Continuous improvement daemon
├── pyproject.toml           # uv/pip dependencies
├── src/
│   ├── parse_session.py     # Claude Code JSONL parser
│   ├── section_parser.py    # Within-file section parser for nested doc optimization
│   ├── synthetic_evaluator.py  # GEPA + DSPy synthetic evaluation
│   ├── evaluator.py         # GEPA-compatible scoring functions
│   ├── utils.py            # Shared helpers (_parse_llm_json)
│   └── llm_config.py      # MiniMax LLM configuration
├── skills/
│   └── banking-analytics-seed.md  # Derek's seed SKILL.md
└── outputs/                 # Optimized artifacts land here
    └── skill/
        ├── best_candidate.md
        └── gepa_result.json
```

---

## Tips for Your Stack

**Banking analytics project filter**: `--project-filter banking` or whatever substring
your `~/.claude/projects/` directory for that repo contains.

**RPIV pipeline**: Optimize `AGENTS.md` with `--target agent` to improve the
Research→Plan→Implement→Validate handoffs. The corpus from multi-agent sessions
includes subagent JSONL files automatically.

**Nested context files**: Use `--target nested` to optimize CLAUDE.md files at multiple
directory levels (root + src/ + test/ + etc.). Each file is evolved independently,
reducing context waste while keeping guidance focused.

**opencode plugins**: The session logs capture all tool calls including MCP tool
calls (`mcp__server__tool` names). These appear in the tool sequence ASI.

**Incremental approach**: Start with 1 optimization run, deploy the skill, run
Claude Code on real tasks for a week, then re-run optimization with the new corpus.
The gskill paper showed gains compound: skills learned on cheap models transfer
directly to production agents.

---

## Extending: Custom Evaluators

```python
# Custom evaluator for your specific domain
def my_evaluate(candidate: str, example: dict) -> tuple[float, dict]:
    import gepa.optimize_anything as oa

    # Run your domain-specific scoring
    score = my_domain_scorer(candidate, example)

    # Log diagnostic info as ASI
    oa.log(f"Task: {example['task_prompt'][:100]}")
    oa.log(f"Outcome: {example['outcome']}")
    oa.log(f"Errors: {example['error_messages']}")

    return score, {
        "score": score,
        "task": example["task_prompt"][:200],
        # ... any other diagnostic fields
    }
```

---

## Changelog

### Phase 8 (2026-06-03) — Final QA Verification

- **Full QA gate pass**: lint, reviewer, test_engineer, drift verification — all gates approved
- No regressions in Phase 1-7 features

### Phase 4-7 Improvements (2026-06-03)

**Phase 4: GEPA signal-quality fixes**
- `oa.log()` wired as primary ASI channel — 6 diagnostic calls (outcome, duration, errors, tool calls, compaction, judge score) flow to reflection LM
- `--max-evals` CLI override now works (was silently ignored)
- M2.7 judge detection (case-insensitive substring dispatch when `judge_lm` contains "m2.7")
- LLM judge context raised 3000→8000 chars (full skill visible to judge)
- Thinking model guard prevents temperature+thinking conflicts

**Phase 5: Scoring rubric alignment**
- `_cache_bonus` removed from score formula → exposed via `side_info["cache_ratio"]` (indicates session input structure, not skill quality)
- DSPy MIPROv2 instruction extraction — optimized instructions + few-shot demos injected into output
- Judge weight standardized to 0.65 in both `make_replay_evaluator` and `make_synthetic_evaluator`

**Phase 6: Data ingestion reliability**
- Subagent JSONL scanning — now scans both `project_dir/*.jsonl` and `project_dir/subagents/agent-*.jsonl`
- Two-poll size stability — mtime replaced with size tracking across polls (handles long thinking pauses)
- `skip_paths` parameter for cumulative dedup across watch sessions

**Phase 7: Module hygiene**
- Section parser whitespace stripping — `lstrip('\n')` prevents round-trip accumulation
- Heading de-collision — `_make_unique_key()` adds `_2`, `_3` suffixes for collisions
- Explicit `llm_config.configure()` — module no longer runs as side effect
- Nested evaluator priority fallback — prefers root keys (no `/`) over nested, then alphabetical

### Phase 1 (2026-06-03) — Code Review Remediation

- **Added `--phase` flag** to `optimize.py`:
  - `--phase 1` (default): Synthetic exploration, 100 evals, 4-thread parallel evaluation
  - `--phase 2`: Session-backed refinement, 60 evals (requires session logs)
- **Performance improvements**: `EngineConfig parallel=True` reduces wall time 4-8x
- **New `src/utils.py`**: Shared `_parse_llm_json` helper function
- **GEPA reflection feedback**: Evaluators now populate `side_info["feedback"]` with natural-language diagnostic strings (not just numeric scores)
- **Conservative outcome inference**: Ambiguous sessions now score 0.5 (was 1.0). Uses `POSITIVE_COMPLETION_SIGNALS`
- **Configurable tool-call thresholds**: Evaluators accept `tool_call_thresholds` kwarg
- **Section parser idempotency**: `_heading_<key>` metadata enables round-trip across GEPA iterations

**Bug fixes (27 issues across 7 source files):**
- token_stats double-counting deduplicated by message.id
- multi-evaluator temp dir scoping corrected
- NESTED_ROOT now fetched from CLI arg (was hardcoded)
- DSPy extraction warning when no signatures found
- Regex pre-compilation in synthetic_evaluator.py
- Backup file rotation (max 5 backups)

**All tests pass**: 102/102, zero regressions

### Earlier Versions

If you use this in research:

```bibtex
@misc{agrawal2025gepa,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A and others},
  year={2025},
  eprint={2507.19457},
  archivePrefix={arXiv},
}
```
