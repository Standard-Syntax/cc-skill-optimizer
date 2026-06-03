# cc-skill-optimizer — Code Review

**Reviewed:** 2026-06-03  
**Codebase:** `cc-skill-optimizer-main` (GEPA + DSPy skill optimizer)  
**Reviewer:** Claude Sonnet 4.6  
**Scope:** Efficiency, accuracy, configuration, and strategic improvement

---

## Architecture overview

The pipeline is well-structured. `parse_session.py` extracts episodes from JSONL logs → `evaluator.py` scores candidates → `optimize.py` orchestrates GEPA/DSPy → `watch_and_learn.py` closes the continuous loop. The six optimization targets (`skill`, `claude`, `agent`, `multi`, `sections`, `nested`) are coherent, and the `dict[str, str]` multi-component mode is a genuinely good architectural choice that maps cleanly to GEPA's native multi-parameter support.

---

## Module-by-module findings

### `parse_session.py` — strong overall

The UUID-based dedup fix (replacing the earlier message.id approach) is correct given Claude Code 2.1+ behavior, where one logical turn is split across multiple JSONL lines that share a message.id but have distinct UUIDs. The dual tool-result extraction — handling both legacy nested `tool_result` blocks and the new top-level `toolUseResult` field — is exactly right.

**`build_corpus` has no caching across calls.** Every invocation re-reads all matching JSONL files from disk. For large corpora (150+ sessions) this is acceptable, but `watch_and_learn.py` already maintains `known_paths` to avoid re-parsing — that dedup logic should be factored back into `build_corpus` as an optional `skip_paths: set[str]` parameter rather than duplicated at the call site.

**Outcome inference is conservative (intentionally) but asymmetric.** The `POSITIVE_COMPLETION_SIGNALS` list requires a positive keyword in the last assistant message to declare success. If the agent finishes correctly but closes with a neutral summary like "Here is the updated file," the episode scores `unknown` (0.5) rather than `success` (1.0). This is intentional but compresses training signal. Consider a secondary heuristic: if `len(error_messages) == 0` and `len(files_written) > 0` and outcome is `unknown`, score at 0.7 rather than 0.5.

**`_thinking_from_content` drops `redacted_thinking` blocks.** The parser filters on `block.get("type") == "thinking"` and silently discards `redacted_thinking` type blocks. `llm_config.py` correctly warns about this in its multi-turn protocol notes, but the parser itself doesn't handle it. For scoring purposes this is benign — you don't need the thinking content to measure outcome or efficiency — but it's worth a comment in the code noting the deliberate omission.

---

### `evaluator.py` — correct but scoring signal is thin

**`_outcome_score` produces a compressed distribution.** With `success=1.0`, `unknown=0.5`, `error=0.0`, and many correctly-completed sessions scoring `unknown` due to conservative outcome inference, the heuristic score for a typical corpus clusters around 0.5–0.65. GEPA's Pareto frontier accumulates candidates that are "best at something" — if most scores fall within 0.1 of each other, the frontier fills with near-duplicate candidates rather than genuinely specialized ones. The `_efficiency_bonus` (±0.15) helps, but the effective scoring range is roughly 0.35–0.85. Widening the outcome inference (see `parse_session.py` note above) is the most direct fix.

**`_cache_bonus` measures the wrong thing.** Cache hit ratio reflects the input token structure of a session, not the quality of the skill being optimized. A skill identical to the previous iteration inherits its predecessor's cache state and receives a bonus regardless of quality. This can reward unchanged candidates over genuinely improved ones. Move cache ratio out of the score formula and into `side_info` only, so the reflection LM can use it qualitatively as a verbosity signal.

**The LLM judge truncates the skill to 3,000 characters.** A SKILL.md optimized to the 2,000-token target is approximately 8,000–12,000 characters. The judge in `llm_judge_score` truncates `candidate_skill[:3000]`, meaning it never evaluates the second half of the skill. Increase the truncation limit to at least 8,000 characters, or split the judge into two calls covering first and second halves and average the scores.

**Inconsistent `judge_weight` defaults across evaluators.** `make_replay_evaluator` defaults to `judge_weight=0.4` (40% LLM judge, 60% heuristic); `make_synthetic_evaluator` defaults to `judge_weight=0.65` (65% LLM judge). There is no comment explaining the difference. Since the LLM judge is the only source of semantic signal about whether a skill would have helped an agent, 40% in replay mode seems low — especially given how compressed the heuristic score distribution is.

---

### `synthetic_evaluator.py` — creative but two significant gaps

**`oa.log()` is never called inside either evaluator.** This is the highest-leverage gap in the codebase. `oa.log()` is GEPA's primary ASI injection mechanism: output written with it is captured per-evaluator-call in a thread-safe context variable and injected directly into the reflection prompt. The codebase instead builds `side_info["feedback"]` manually and returns it in the `(score, dict)` tuple. This works — `optimize_anything` does consume the returned dict — but it bypasses the intended channel. The reflection LM would benefit significantly from explicit `oa.log()` calls for the rich diagnostic fields:

```python
import gepa.optimize_anything as oa

# inside the evaluate() closure:
oa.log(f"Outcome: {ep['outcome']}")
oa.log(f"Duration: {ep.get('duration_s', '?'):.0f}s")
oa.log(f"Errors: {ep['error_messages'][:2]}")
oa.log(f"Tool sequence: {' → '.join(tc['tool'] for tc in ep['tool_calls'][:10])}")
if ep.get('compaction_summary'):
    oa.log("Context compaction hit — skill may be too verbose")
```

**`judge_score_task_m2_7` is dead code.** The M2.7-tuned judge function is defined and exported from `synthetic_evaluator.py` but is never called from `make_synthetic_evaluator`. The default `judge_score_task` is used regardless of which model is configured as the judge. The M2.7-specific rubric — rewarding error-recovery strategies, rewarding multiple solution approaches, penalizing verbose skills over 150 words — is the right scoring policy when routing through `minimax/minimax-m2.7-highspeed`, but it never fires. Wire it in:

```python
# in make_synthetic_evaluator, detect M2.7 and dispatch accordingly
_judge_fn = judge_score_task_m2_7 if "m2.7" in judge_lm else judge_score_task

def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
    ...
    if use_judge:
        j_score, j_info = _judge_fn(candidate, example, judge_lm)
```

**`_REQUIRED_SECTIONS` regex for `commands` is too broad.** The pattern `r"##?\s*(command|run|build|test|install|usage)"` matches `## Installation Guide` — which is usually developer setup, not runtime commands the agent needs. This inflates structural scores for skills with setup documentation but no actionable runtime guidance. Tighten the pattern or add a negative lookahead for `install` when not followed by whitespace and end-of-line.

**The DSPy two-stage pipeline does not extract the optimized instructions.** `make_dspy_synthetic_pipeline` assembles its output by running the optimized program on a few tasks and concatenating raw `improved_guidance` fields into a `## Auto-Generated Guidance (GEPA-refined)` section. This discards whatever instruction optimization MIPROv2 discovered. The correct approach is to extract `optimized.predictor.signature.instructions` after compilation — the same extraction that `run_dspy_gepa` in `optimize.py` attempts (with a fallback warning) but that `make_dspy_synthetic_pipeline` skips entirely.

---

### `optimize.py` — well-structured, one significant bug

**`--max-evals` is effectively ignored.** In both `run_gepa_optimize_anything` and `run_gepa_synthetic`, the call passes:

```python
max_metric_calls=max_evals_override if max_evals_override is not None else max_metric_calls,
```

`max_evals_override` is set to `_gepa_default_max_evals` (100 for phase 1, 60 for phase 2) in `main()` unconditionally — it is never `None`. The `--max-evals` CLI argument therefore never controls the optimization budget. The intent was clearly for `--phase` to set sensible defaults while still allowing user override. Fix by using `args.max_evals` as the primary value and applying phase defaults only when the user did not explicitly pass `--max-evals`:

```python
# in main():
_phase_default = {1: 100, 2: 60}[args.phase]
_effective_max_evals = args.max_evals if args.max_evals != 100 else _phase_default
# (or detect via argparse default sentinel with ap.set_defaults)
```

**`frontier_type` default for phase 1 (`"instance"`) could be `"hybrid"`.** With `"instance"` tracking, GEPA maintains candidates that are best on individual episodes. With a small corpus (20–50 episodes), many candidates can land on the frontier through random variation rather than genuine specialization. `"hybrid"` tracks both per-instance and per-objective frontiers, which with a multi-component score (outcome + efficiency + cache) gives the frontier more discriminating structure. This is a low-risk experiment worth trying on a phase 1 run.

**`run_dspy_gepa` passes `temperature=0.7` and `temperature=1.0` to DSPy LM objects.** `llm_config.py` correctly documents that temperature is incompatible with Anthropic extended thinking. If someone passes a thinking-enabled model string to `--task-lm` or `--reflection-lm` for the DSPy path, the call will fail. Add a guard or at least a prominent comment at the DSPy runner entry point.

**`make_nested_evaluator` fallback is fragile.** When the `CLAUDE.md` key is absent from the candidate dict, the code falls back to `min(candidate, key=lambda k: len(candidate[k]))` — the shortest file by character count. The shortest file is likely a minimal subsection config or a stub, not the primary context file. The fallback should prefer any key without a `/` separator (root-level file), then fall back to alphabetical order.

---

### `section_parser.py` — mostly correct, one round-trip issue

The idempotency fix — storing `_{key}_heading` metadata to preserve original heading casing across GEPA iterations — is the right approach. However, `merge_sections` reconstructs sections as:

```python
f"{stored_heading.rstrip()}\n{content}"
```

If `content` begins with a blank line (common when section bodies start with a paragraph break), each GEPA round-trip adds one extra blank line between the heading and the body. After ten or more iterations this produces skills with progressively growing inter-section whitespace. Fix by stripping the leading newline from `content` before reassembly:

```python
f"{stored_heading.rstrip()}\n{content.lstrip(chr(10))}"
```

**`_normalize_key` produces silent key collisions** for headings that differ only in punctuation: `## Key Patterns!` and `## Key Patterns?` both normalize to `section_key_patterns_`. In practice this is rare, but if two sections collide the second silently overwrites the first in the `sections` dict. A de-collision suffix (append `_2`, `_3`, and so on) would be safer.

---

### `llm_config.py` — accurate, one architectural concern

**`configure()` runs as a module-level side effect at import time.** Importing `llm_config` anywhere — including in tests — immediately mutates `ANTHROPIC_BASE_URL` and reroutes all subsequent litellm calls to MiniMax's endpoint. This makes it impossible to import the module in a context where you want the real Anthropic endpoint without also accepting the base URL override. Move the `configure()` call to an explicit opt-in, called from `main()` only:

```python
# llm_config.py — remove the bare configure() call at module bottom
# optimize.py main():
from src.llm_config import configure
configure()
```

---

### `watch_and_learn.py` — structurally sound, two reliability gaps

**The 10-second mtime guard does not reliably detect session completion.** Claude Code writes JSONL lines incrementally throughout a session. A 10-second gap in writes may simply be a long thinking step rather than a closed session. Track file size across polls instead: only process a file once its size has been stable for two consecutive poll cycles.

```python
# replace known_paths: set[str] with:
known_files: dict[str, int] = {}  # path → size at last poll
stable_files: set[str] = set()    # paths stable across two polls

# in the scan loop:
current_size = jsonl.stat().st_size
prev_size = known_files.get(str(jsonl))
if prev_size is None:
    known_files[str(jsonl)] = current_size
elif prev_size == current_size and str(jsonl) not in stable_files:
    stable_files.add(str(jsonl))
    new_files.append(jsonl)  # stable for two polls → safe to parse
else:
    known_files[str(jsonl)] = current_size
```

**The watcher does not scan `subagents/` subdirectories.** `parse_session.iter_session_files` handles subagent logs correctly, but the polling loop in `watch_and_learn.py` only globs `project_dir/*.jsonl` — not `project_dir/subagents/*.jsonl`. Multi-agent RPIV sessions produce subagent logs that are completely invisible to the daemon. Add the subagents glob:

```python
for jsonl in list(project_dir.glob("*.jsonl")) + list(
    (project_dir / "subagents").glob("*.jsonl")
    if (project_dir / "subagents").exists() else []
):
```

---

## Strategic recommendations

The following are ranked by expected impact on optimization quality.

### 1. Wire `oa.log()` as the primary ASI channel (high impact)

This is the single highest-leverage change. The reflection LM currently receives feedback only through `side_info["feedback"]` — a single string constructed in the evaluator. `oa.log()` is captured per-evaluator-call in a thread-safe context variable and injected directly into the reflection prompt in the format GEPA was designed to consume. Switching the rich diagnostic fields (error sequences, tool sequences, compaction events, token stats) to `oa.log()` calls, while keeping `side_info` for structured metadata, gives the reflection LM fuller context with no additional API cost.

### 2. Fix the `--max-evals` override bug (high impact)

The CLI flag is effectively ignored because `_gepa_default_max_evals` is always non-None. This means every optimization run uses the phase default (100 or 60), regardless of what the user specifies. Until this is fixed, budget control requires editing source code rather than CLI flags.

### 3. Wire the M2.7-specific judge (high impact for your stack)

`judge_score_task_m2_7` encodes the right rubric for your default judge model — rewarding error recovery, rewarding multiple solution approaches, penalizing verbosity over 150 words. It is dead code today. Dispatching to it when the judge model string contains `"m2.7"` requires three lines of change and immediately improves scoring alignment with M2.7's known behavior profile.

### 4. Fix LLM judge skill truncation (medium impact)

The 3,000-character truncation means the judge scores on roughly the first third of a fully-optimized skill. This introduces a systematic bias toward shorter candidates — not because they perform better, but because the judge sees all of them. Increase to 8,000 characters or implement a two-pass scoring strategy.

### 5. Replace `_cache_bonus` with ASI-only metadata (medium impact)

Cache hit ratio is a session property, not a skill quality signal. Removing it from the score formula eliminates a noise source that can reward unchanged candidates. Keeping it in `side_info` lets the reflection LM interpret it qualitatively — "cache ratio was 0.09 and compaction hit; skill may be generating excessively verbose sessions."

### 6. Fix the `watch_and_learn.py` subagent gap (medium impact for RPIV)

RPIV sessions produce the richest training signal — multi-step tool sequences, clear success/failure boundaries, and explicit error escalation. None of that data reaches the daemon today because subagent JSONL files are not scanned. For continuous optimization of AGENTS.md or multi-agent skills, this is a blocking gap.

### 7. Consider seedless mode for first-run bootstrapping (low impact, low effort)

When no prior SKILL.md exists (or only the generic `SEED_SKILL_MD` placeholder would be used), `seed_candidate=None` lets the reflection LM generate the first draft from `objective` and `background` alone. The objective and background strings in `optimize.py` are detailed enough that seedless mode would likely produce a stronger starting point than the generic seed — and it costs one additional reflection call.

### 8. Enable `use_merge` if exposed through `EngineConfig` (low impact, worth checking)

System-aware merge — combining instructions from two Pareto-frontier candidates that excel on different episodes — is one of GEPA's key differentiators from basic evolutionary search. Check whether the current `optimize_anything` API exposes merge controls through `EngineConfig` or as a direct `optimize_anything` parameter, and enable it. It is especially valuable for `--target multi` where different components may naturally specialize on different episode types.

---

## Summary table

| Module | Finding | Severity | Status |
|--------|---------|----------|--------|
| `synthetic_evaluator.py` | `oa.log()` never called; ASI routed through dict only | High | Open |
| `optimize.py` | `--max-evals` CLI flag ignored; phase default always wins | High | Open |
| `synthetic_evaluator.py` | `judge_score_task_m2_7` defined but never called | High | Open |
| `evaluator.py` | LLM judge truncates skill at 3,000 chars (target is ~10,000) | Medium | Open |
| `evaluator.py` | `_cache_bonus` in score formula rewards session structure, not skill quality | Medium | Open |
| `evaluator.py` | `judge_weight` defaults inconsistent across evaluators (0.4 vs 0.65) | Medium | Open |
| `watch_and_learn.py` | Subagent JSONL files not scanned; RPIV sessions invisible | Medium | Open |
| `watch_and_learn.py` | mtime guard unreliable for detecting session completion | Medium | Open |
| `synthetic_evaluator.py` | DSPy pipeline discards MIPROv2 instruction output; uses raw model outputs | Medium | Open |
| `parse_session.py` | `build_corpus` has no skip-paths caching; re-reads all files each call | Low | Open |
| `parse_session.py` | Completed sessions with neutral closing message score `unknown` (0.5) | Low | Open |
| `section_parser.py` | Round-trip accumulates extra blank lines across GEPA iterations | Low | Open |
| `section_parser.py` | `_normalize_key` produces silent collisions for punctuation-only heading differences | Low | Open |
| `llm_config.py` | `configure()` runs as module-level side effect; mutates global env at import | Low | Open |
| `optimize.py` | `make_nested_evaluator` fallback selects shortest file, not root-level file | Low | Open |
| `optimize.py` | DSPy path uses `temperature` — incompatible with thinking-enabled models | Low | Open |
