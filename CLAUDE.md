# cc-skill-optimizer

Optimizes Claude Code `SKILL.md`, `CLAUDE.md`, and `AGENTS.md` files using GEPA
(genetic-Pareto optimization) and DSPy — powered by real session logs from
`~/.claude/projects/**/*.jsonl`.

## Stack

- **Python 3.13+** (not 3.14 — some `gepa` dependencies have not been updated for 3.14)
- **uv** for package management (NOT pip)
- **gepa** for genetic-Pareto optimization
- **dspy** for program-level optimization
- **litellm** for LLM routing (anthropic-compatible endpoint)
- **anthropic** SDK for direct API fallback

## Commands

| Task | Command |
|------|---------|
| Run optimization | `uv run python optimize.py --target <skill\|claude\|agent\|multi\|sections\|nested> [args]` |
| Continuous daemon | `uv run python watch_and_learn.py --skill-file <path> --project-filter <filter>` |
| Inspect JSONL | `uv run python inspect_jsonl.py <jsonl_file>` |
| Run diagnostics | `uv run python diagnose.py <jsonl_file>` |
| Lint | `uv run ruff check . && uv run ruff format --check .` |
| Type check | `uv run ty check .` |

⚠️ **Do NOT run `pip` directly** — always use `uv run` or `uv add`.

## Hard Constraints

- **Always use `uv run`** for any Python script execution — bare `python` will miss venv dependencies
- **GEPA requires (score, side_info) tuple** from evaluator — returning a flat float breaks ASI (Adaptive Strategy Integration) reflection loop
- **Always call `llm_config.configure()` explicitly** before any litellm calls — the module no longer runs as a side effect on import
- **Never pass `temperature` or `top_k` with extended thinking enabled** — these parameters break Anthropic extended thinking; use `top_p` (0.95–1.0) instead
- **Always pass back `thinking` AND `redacted_thinking` blocks** in tool-use loops — filtering on `block.type == "thinking"` silently drops redacted blocks and breaks reasoning continuity

## LLM Configuration

This project routes all LLM calls through litellm to the MiniMax Anthropic-compatible endpoint (`https://api.minimax.io/anthropic/v1`). The endpoint accepts standard Anthropic request formats with `anthropic/` model prefix.

### Model Constants (src/llm_config.py)

| Constant | Value | Best Use |
|----------|-------|----------|
| `DEFAULT_MODEL` | `minimax/minimax-m2.7-highspeed` | Fast eval + judge calls (cheap, fast) |
| `REFLECTION_MODEL` | `minimax/minimax-m3` | Deep reflection / GEPA mutation proposals |
| `MINIMAX_M2_7` | `minimax/minimax-m2.7` | Skill generation, coding tasks needing 97% skill adherence |
| `MINIMAX_M2_7_HIGHSPEED` | `minimax/minimax-m2.7-highspeed` | Faster M2.7 variant for iterative coding tasks |

### M2.7 Profile

**Strengths:** 97% skill adherence, 34% hallucination rate (low — best-in-class), SWE-Pro 56.22%, $0.30/M input tokens.

**Weaknesses:**
- **4x verbosity** — generates ~4× more tokens than average models
- **Slow throughput** — ~50 tokens/second
- **Tunnel-vision debugging** — repeats the same failed approach instead of trying alternatives
- **"Read everything first"** — reads more files than needed before responding, causing timeouts

### M2.7 Mitigations (bundled in src/)

All mitigations are already implemented in the codebase. Use these when configuring M2.7 as your task or reflection LM:

| Mitigation | Constant / Function | What It Does |
|------------|-------------------|--------------|
| Concise system prefix | `M2_7_CONCISE_PREFIX` | Enforces concise output ("Target 50% fewer tokens"). Prepended to system prompts for M2.7 calls. |
| Low-effort thinking | `THINKING_CONFIG_M2_7` | Adaptive thinking with `effort: "low"` — prevents over-thinking simple tasks. |
| Strict token budgets | `M2_7_REFLECTION_MAX_TOKENS = 4096` | Caps reflection output at 4K tokens (half the Sonnet budget). |
| Fast eval budget | `M2_7_EVAL_MAX_TOKENS = 256` | Very low cap for judge calls — JSON score response needs ~50 tokens. |
| M2.7-optimized judge | `_JUDGE_SYSTEM_M2_7` + `judge_score_task_m2_7()` | Modified rubric that rewards skills with multiple approaches, error recovery, and conciseness (under 150 words). |
| Tunnel-vision directives | Documented in `optimize.py` at line ~1220 | Guidelines like "If the first approach fails, explicitly try a fundamentally different strategy." |

### Thinking Configuration

- **Never pass `temperature` or `top_k`** with extended thinking enabled — they are incompatible and raise API errors.
- **top_p** is allowed at 0.95–1.0 when thinking is enabled.
- **Always pass back `thinking` AND `redacted_thinking` blocks** in tool-use loops (see Hard Constraints above).
- `THINKING_CONFIG_REFLECTION` — adaptive, `effort: "medium"` — for GEPA reflection calls.
- `THINKING_CONFIG_EVAL` — `type: "disabled"` — for judge/eval calls (speed + cost).
- `THINKING_CONFIG_M2_7` — adaptive, `effort: "low"` — mitigates M2.7 verbosity.

## Architectural Ripples

- `src/parse_session.py` → `src/evaluator.py`: episodes parsed by parse_session have a strict schema (task_prompt, tool_calls, outcome, etc.) that evaluator.py's scoring functions depend on — do not change the episode schema without updating _outcome_score() and _efficiency_bonus()
- `src/llm_config.py` → all LLM calls: import llm_config first to set ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL before any litellm calls; module sets these as side-effect
- `optimize.py` → GEPA: GEPA's dict[str,str] candidate mode (for multi-component targets) requires all component keys to be stable across iterations — changing section key names mid-evolution invalidates the population

## Domain Terms

- **episode**: one user task request + full agent trace + outcome signal; the atomic unit of evaluation
- **ASI**: Adaptive Strategy Integration — GEPA's mechanism where side_info from evaluator drives the reflection loop
- **REPLAY mode**: evaluation against previously-parsed episodes (fast, no Claude Code needed)
- **LIVE mode**: evaluation by invoking Claude Code on a real task (ground-truth but expensive)
- **side_info**: dict returned alongside score from evaluator; GEPA reads this to adapt strategy; must be serializable to JSON

## Do Not Read or Modify

- `outputs/` — GEPA run artifacts; generated fresh on each run
- `.venv/` — uv virtual environment; managed by uv, never edit manually
- `skills/python-313-modern-syntax/` and `skills/python-modern-design-patterns/` — bundled reference skills, not part of this project
- `.ruff_cache/`, `__pycache__/` — tool output

## Active State (as of 2026-05)

- This project is a research prototype — API surfaces (CLI args, internal function signatures) may change between sessions
- The `nested` target is experimental — section key stability across file boundaries is not yet fully validated
