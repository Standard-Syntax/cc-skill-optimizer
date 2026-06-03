# `cc-skill-optimizer` — Code Review

> GEPA + DSPy optimization pipeline for Claude Code skill files  
> Reviewed: June 2026 · Confidence: 8/10

-----

## Overview

The architecture is sound. Session JSONL parsing → heuristic + LLM-judge scoring → GEPA reflection loop, with a clean synthetic fallback for zero-session runs. The code is well-documented and the layered evaluation strategy (structural scorer → LLM judge → heuristic replay) is a good design. What follows are the gaps between how GEPA is being used and how it is designed to work at full strength — plus a handful of accuracy issues in the scoring logic.

-----

## 🔴 Critical

### 1. Evaluator returns `side_info` dict — GEPA reads `feedback` string

This is the highest-leverage issue. From the GEPA docs:

> GEPA reads the metric’s `feedback` field directly into the reflection prompt. A plain-float metric still works, but the proposer sees only a generic “This trajectory got a score of {n}” caption — concrete failure modes never reach it. **GEPA doesn’t error on a float; it just gives you a much weaker version of itself.**

Your replay evaluator’s `side_info` dict is rich diagnostic data — error messages, tool call counts, compaction events — but `gepa.optimize_anything` passes it through as ASI without synthesizing it into the natural-language `feedback` string the reflection LM actually reads. The insight “agent hit 3 timeout errors after trying to `grep` across 80k files” never reaches the mutation proposal.

**Fix — add a `feedback` key to `side_info`:**

```python
# In evaluator.py
def _build_feedback(episode: dict, side_info: dict) -> str:
    parts = []
    if episode.get("outcome") == "error":
        errs = side_info["error_messages"][:2]
        parts.append(f"Agent hit errors: {'; '.join(errs)}")
    if side_info["n_tool_calls"] >= 25:
        parts.append(
            f"Excessive tool calls ({side_info['n_tool_calls']}); "
            "skill may lack shortcuts or preferred commands."
        )
    if episode.get("compaction_summary"):
        parts.append(
            "Context compaction hit — skill may be too verbose "
            "or omits key file pointers."
        )
    if not parts:
        parts.append(
            f"Outcome: {episode.get('outcome', 'unknown')}. "
            "No specific failure detected."
        )
    return " ".join(parts)

# In make_replay_evaluator → evaluate():
side_info["feedback"] = _build_feedback(example, side_info)
```

The same applies in `make_synthetic_evaluator`. The judge already returns a `gaps` list — that is perfect material for `feedback`:

```python
side_info["feedback"] = (
    f"Score {final:.2f}. Structural: {struct_score:.2f}. "
    f"Gaps: {'; '.join(j_info.get('gaps', []))[:300]}. "
    f"Judge: {j_info.get('reasoning', '')}"
)
```

-----

### 2. `parallel=False` in `EngineConfig` kills throughput

```python
config=GEPAConfig(
    engine=EngineConfig(
        max_metric_calls=max_metric_calls,
        cache_evaluation=True,
        parallel=False,        # ← bottleneck
        frontier_type="instance",
    ),
)
```

The replay evaluator is stateless — it reads from the closed-over `episodes` list and makes independent API calls. Parallelism is safe here. With `parallel=False`, every metric call is sequential, meaning a 150-eval run that could finish in 15 minutes takes over an hour.

**Fix:**

```python
engine=EngineConfig(
    max_metric_calls=max_metric_calls,
    cache_evaluation=True,
    parallel=True,
    frontier_type="instance",
),
```

Start conservatively (4 threads) to avoid MiniMax rate limits. For the synthetic path, add `num_threads=4` to `GEPAConfig` — the structural scorer is pure Python and judge calls are independent.

-----

## 🟠 High

### 3. Outcome inference is too optimistic

```python
else:
    outcome = "success"  # conservative: if no errors, assume success
```

The comment says “conservative” but the logic is the opposite. An agent that read 20 files and stopped with “I could not find the configuration” scores as `success`. This corrupts the training signal by rewarding dead-end sessions.

**Fix — require a positive completion signal:**

```python
positive_signals = (
    "done", "complete", "finished", "success", "✓",
    "implemented", "created", "added", "fixed",
)
if any(w in last for w in positive_signals):
    outcome = "success"
else:
    outcome = "unknown"  # stays at 0.5 — honest for ambiguous sessions
```

-----

### 4. `reflection_minibatch_size=3` is too small for domain complexity

```python
reflection=ReflectionConfig(
    reflection_lm=reflection_lm,
    reflection_minibatch_size=3,
),
```

With `max_evals=100–200` and a 50+ episode corpus, a minibatch of 3 means the reflection LM sees very few failure examples per mutation proposal. Three examples rarely span the failure-mode space for a domain like banking analytics — DAX errors, T-SQL compatibility, TMDL path issues, and uv/pip confusion are four distinct failure classes that won’t all appear in three samples.

**Recommendation:** Use 5–8 for session-backed runs with 50+ episodes. For synthetic mode with 20 tasks, 4 is reasonable.

-----

## 🟡 Medium

### 5. DSPy pipeline: `MIPROv2` used where `dspy.GEPA` is documented

The `make_dspy_synthetic_pipeline` docstring describes Stage 2 as “dspy.GEPA on top of the bootstrapped program” but the implementation uses `MIPROv2`:

```python
mipro_optimizer = MIPROv2(
    metric=metric,
    prompt_model=reflect_lm_obj,
    num_threads=2,
    auto="medium",
)
```

This is not wrong — MIPROv2 is a valid choice — but it loses GEPA’s Pareto frontier and feedback-driven reflection. The metric in this pipeline also returns a bare float, which would miss GEPA’s feedback channel even if you switched.

**Options:**

- **Option A (minimal):** Update the docstring to say “MIPROv2” and be done with it.
- **Option B (full):** Switch to `dspy.GEPA` with a feedback-returning metric:

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    pitfalls = [p[2:] for p in getattr(gold, "task_pitfalls_str", "").split("\n") if p.startswith("- ")]
    coverage = sum(
        any(w.lower() in assessment.lower() for w in p.split()[:3])
        for p in pitfalls
    ) / max(1, len(pitfalls))
    has_code = bool(re.search(r"`[^`]+`|--\w+|\$ \w+", guidance))
    score = min(1.0, coverage * 0.85 + (0.15 if has_code else 0.0))
    feedback = (
        f"Pitfall coverage: {coverage:.0%}. "
        f"{'Has concrete code examples.' if has_code else 'Missing code examples — add commands or snippets.'} "
        f"Uncovered pitfalls: {[p for p in pitfalls if not any(w.lower() in assessment.lower() for w in p.split()[:3])][:2]}"
    )
    return dspy.Prediction(score=score, feedback=feedback)
```

-----

### 6. `_efficiency_bonus` thresholds penalize legitimate complex tasks

```python
if n_tools <= 5:
    bonus += 0.08
elif n_tools <= 12:
    bonus += 0.04
elif n_tools >= 40:
    bonus -= 0.10
elif n_tools >= 25:
    bonus -= 0.05
```

For banking analytics, 12 tool calls is often a good session — reading 3 files, writing a DAX measure, running a linter, verifying output. A complex TMDL scaffold can legitimately need 30+. The ≥25 penalty will downgrade successful complex tasks and push GEPA toward optimizing the skill for trivial ones.

**Fix — make thresholds configurable:**

```python
def make_replay_evaluator(
    episodes: list[dict],
    use_llm_judge: bool = False,
    judge_lm: str = DEFAULT_MODEL,
    judge_weight: float = 0.4,
    tool_call_thresholds: dict | None = None,   # ← add this
):
    thresholds = tool_call_thresholds or {
        "low": 5, "mid": 12, "high": 25, "very_high": 40
    }
    ...
```

Also: `_cache_bonus` caps at `0.05` and contributes negligibly. A cache ratio above 0.70 strongly correlates with well-structured sessions; consider raising the ceiling to `0.10`.

-----

### 7. `merge_sections` silently drifts heading casing across iterations

In `parse_sections`, the heading line is stripped when flushing a section:

```python
flush_lines = current_lines[1:] if current_lines else current_lines
```

Then in `merge_sections`, the reconstruction path runs (since the heading was stripped):

```python
heading_text = _key_to_heading(key)
parts.append(f"{'#' * heading_depth} {heading_text}\n{content}")
```

`_key_to_heading` title-cases the slug. Any heading with non-standard casing — `## SQL Server 2016 Quirks`, `## DAX CALCULATE Patterns` — round-trips to `## Sql Server 2016 Quirks` and `## Dax Calculate Patterns`. After enough GEPA iterations this silently degrades section headings. The idempotency tests likely catch basic cases, but acronym-casing drift is a subtle failure mode that only surfaces after several optimization rounds.

**Fix:** Store the original heading line during parsing:

```python
# In parse_sections, before flushing:
sections[f"_heading_{key}"] = heading  # store verbatim heading line

# In merge_sections:
stored_heading = sections.get(f"_heading_{key}")
if stored_heading:
    parts.append(f"{stored_heading}\n{content}")
else:
    heading_text = _key_to_heading(key)
    parts.append(f"{'#' * heading_depth} {heading_text}\n{content}")
```

-----

## 🟢 Low

### 8. `_parse_llm_json` is duplicated across modules

`evaluator.py` and `synthetic_evaluator.py` both define identical implementations. Extract to a shared `src/utils.py`:

```python
# src/utils.py
import json, re

def parse_llm_json(raw: str, default: dict | list) -> dict | list:
    """Strip markdown code fences and parse LLM JSON response."""
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default
```

-----

### 9. `find_section` defined inside a for loop

In `build_section_tree`, the recursive `find_section` helper is redefined on each loop iteration:

```python
for lineno, line in enumerate(lines, start=1):
    ...
    def find_section(sections: list[Section], name: str) -> Section | None:
        ...
    current = find_section(root, ...)
```

Move it to module level.

-----

## Strategy: Two-Phase Optimization

The README mentions a phase-2 workflow (synthetic → refine with real sessions), which is the right approach. But the GEPA config doesn’t vary between phases — `max_evals=100` and `reflection_minibatch_size=3` regardless of which phase you are in.

Phase 1 (synthetic, broad exploration) benefits from:

- `frontier_type="instance"` — explore many distinct candidates
- Higher `reflection_minibatch_size` (5–8)
- `candidate_selection_strategy="pareto"` (default)

Phase 2 (session-backed, refinement from a Phase 1 winner) benefits from:

- `candidate_selection_strategy="current_best"` — converge on the known good seed
- Lower `reflection_minibatch_size` (3–4) focused on real failure modes

Consider adding a `--phase` flag that configures these automatically.

-----

## Summary

|Priority  |Issue                                                |File                                    |Impact                         |
|----------|-----------------------------------------------------|----------------------------------------|-------------------------------|
|🔴 Critical|Evaluator missing `feedback` string for reflection LM|`evaluator.py`, `synthetic_evaluator.py`|GEPA runs at ~50% effectiveness|
|🔴 Critical|`parallel=False` in `EngineConfig`                   |`optimize.py`                           |4–8× throughput loss           |
|🟠 High    |`outcome = "success"` when no errors detected        |`parse_session.py`                      |Corrupt training signal        |
|🟠 High    |`reflection_minibatch_size=3` too small              |`optimize.py`                           |Weak reflection proposals      |
|🟡 Medium  |MIPROv2 used where docs say dspy.GEPA                |`synthetic_evaluator.py`                |Missed Pareto exploration      |
|🟡 Medium  |Tool-call thresholds penalize complex tasks          |`evaluator.py`                          |Bias toward trivial sessions   |
|🟡 Medium  |Heading casing drifts across merge/parse cycles      |`section_parser.py`                     |Idempotency failure at scale   |
|🟢 Low     |`_parse_llm_json` duplicated                         |`evaluator.py`, `synthetic_evaluator.py`|Maintenance debt               |
|🟢 Low     |`find_section` defined inside for loop               |`section_parser.py`                     |Cosmetic performance issue     |

The highest-leverage change is item 1: threading natural-language feedback strings into the GEPA reflection loop. That is the core mechanic currently left unused. Fix that, enable parallelism, and correct the outcome inference — those three changes together will meaningfully improve optimization quality before touching anything else.