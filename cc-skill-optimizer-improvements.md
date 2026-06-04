# `cc-skill-optimizer` — Improvement Recommendations

> Excludes model/provider changes. Ordered by expected impact.

-----

## 1. Wire `neutral_closing` into `_outcome_score()`

**File:** `src/evaluator.py`  
**Severity:** 🔴 Bug — silent accuracy regression

`parse_session.py` sets `neutral_closing=True` on episodes where the primary outcome is `"unknown"` but secondary signals are strong (no errors, files written). The field is documented as mapping to ~0.7 instead of 0.5. `_outcome_score()` never reads it, so the signal is computed and discarded.

**Current code:**

```python
def _outcome_score(episode: dict) -> float:
    mapping = {
        "success": 1.0,
        "unknown": 0.5,   # neutral_closing ignored here
        "interrupted": 0.3,
        "error": 0.0,
    }
    return mapping.get(episode.get("outcome", "unknown"), 0.5)
```

**Fix:**

```python
def _outcome_score(episode: dict) -> float:
    outcome = episode.get("outcome", "unknown")
    if outcome == "unknown" and episode.get("neutral_closing"):
        return 0.7
    return {
        "success": 1.0,
        "unknown": 0.5,
        "interrupted": 0.3,
        "error": 0.0,
    }.get(outcome, 0.5)
```

-----

## 2. Fix the train/val split for small corpora

**File:** `optimize.py` → `split_corpus()` and `main()`  
**Severity:** 🔴 Accuracy — misaligned with GEPA’s own guidance

The GEPA FAQ prescribes: “80/20 when total datapoints exceed 200; 50/50 when fewer than 200.” The current 70/20/10 split discards a 10% test slice that `main()` never uses. On a 50-session corpus this means six episodes burned for nothing.

**Current:**

```python
def split_corpus(
    episodes: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.20,
    ...
```

**Fix:**

```python
def split_corpus(
    episodes: list[dict],
    train_frac: float = 0.80,   # GEPA recommends maximizing train; val stays 20%
    val_frac: float = 0.20,
    ...
```

Also expose `--test-frac 0` as a CLI flag, or remove the unused test split entirely. The synthetic path already uses 75/25 — closer to GEPA guidance — and can stay as-is.

> **Reference:** GEPA FAQ — “We typically recommend calling GEPA with at least 15–30× len(valset) metric calls to allow it to propose and evaluate up to 15 new candidates.”

-----

## 3. Raise and reorder the ASI episode truncation

**File:** `src/evaluator.py` → `llm_judge_score()`  
**Severity:** 🟠 Quality — high-signal content cut first

The candidate gets 8,000 chars; the ASI episode gets 2,000. `episode_to_asi()` places errors and outcome near the top but appends tool sequence and the final assistant message last — the sections most likely to be truncated. The GEPA guide states the reflection LM “benefits most from full tool_calls and complete test output.”

**Current:**

```python
user_msg = (
    f"SKILL.md:\n{candidate_skill[:8000]}\n\n"
    f"Episode:\n{asi[:2000]}\n\n"
```

**Fix — two changes:**

1. Raise the ASI cap:

```python
user_msg = (
    f"SKILL.md:\n{candidate_skill[:6000]}\n\n"
    f"Episode:\n{asi[:4000]}\n\n"
```

1. Reorder `episode_to_asi()` so highest-signal content comes first:

```python
def episode_to_asi(ep: dict) -> str:
    parts: list[str] = []
    parts.append(f"## Outcome: {ep['outcome'].upper()}")          # was 2nd
    if ep["error_messages"]:
        parts.append("## Errors\n" + ...)                          # was 4th
    parts.append(f"## Task\n{ep['task_prompt'][:400]}")           # was 1st
    if ep["assistant_text"]:
        last = ep["assistant_text"][-1][:800]
        parts.append(f"## Final assistant message\n{last}")       # was last
    # ... bash commands, files, token stats, tool sequence after
```

-----

## 4. Score all components in `make_multi_evaluator` and `make_nested_evaluator`

**File:** `optimize.py`  
**Severity:** 🟠 Accuracy — multi-component Pareto is effectively single-component

Both wrappers extract one “primary text” and score only that. Components tracked in `side_info["components"]` don’t affect the score. GEPA’s Pareto frontier can’t surface trade-offs between files it has no scores for.

**Current:**

```python
def evaluate(candidate: dict[str, str], example) -> tuple[float, dict]:
    skill_text = candidate.get("skill_md") or candidate.get("claude_md") or ""
    if not skill_text:
        skill_text = "\n\n".join(f"# {k}\n{v}" for k, v in candidate.items())
    score, side_info = base_evaluator(skill_text, example)
    ...
```

**Fix — concatenate all components for scoring:**

```python
def evaluate(candidate: dict[str, str], example) -> tuple[float, dict]:
    # Score the full candidate, not just the primary component.
    # Concatenate all components so the judge and heuristics see the whole package.
    combined_text = "\n\n".join(f"# {k}\n{v}" for k, v in candidate.items())
    score, side_info = base_evaluator(combined_text, example)
    side_info["components"] = {k: len(v) for k, v in candidate.items()}
    side_info["n_components"] = len(candidate)
    return score, side_info
```

For true multi-objective tracking, add a `scores` dict to `side_info` keyed by component name:

```python
# Score each component individually and expose for hybrid Pareto
component_scores = {}
for key, text in candidate.items():
    s, _ = base_evaluator(text, example)
    component_scores[key] = s
side_info["scores"] = component_scores  # gepa reads side_info["scores"] for frontier_type="hybrid"
```

-----

## 5. Extract shared DSPy infrastructure; document the DSPy path limitation

**File:** `optimize.py` → `run_dspy_gepa()` and `run_dspy_native_gepa()`  
**Severity:** 🟠 Maintainability — duplicated code that has already diverged

Both runners define identical `SkillGuidedTask`, `SkillProgram`, `ep_to_example()`, and `_ideal_completion_from_episode()`. The metric return types have already diverged between the two (one returns `float`, one returns `dspy.Prediction`).

**Fix — create `src/dspy_shared.py`:**

```python
# src/dspy_shared.py
"""Shared DSPy infrastructure for run_dspy_gepa and run_dspy_native_gepa."""

import dspy


class SkillGuidedTask(dspy.Signature):
    """Apply repository skills to complete a software engineering task."""
    skill_instructions: str = dspy.InputField(desc="SKILL.md content guiding the agent")
    task_prompt: str = dspy.InputField(desc="The software engineering task to complete")
    error_context: str = dspy.InputField(
        desc="Prior errors and context from the session", default=""
    )
    completion: str = dspy.OutputField(
        desc="How the agent should approach and complete this task"
    )


class SkillProgram(dspy.Module):
    def __init__(self, skill_content: str):
        self.skill_content = skill_content
        self.predictor = dspy.Predict(SkillGuidedTask)

    def forward(self, task_prompt: str, error_context: str = "") -> dspy.Prediction:
        return self.predictor(
            skill_instructions=self.skill_content,
            task_prompt=task_prompt,
            error_context=error_context,
        )


def ep_to_example(ep: dict) -> dspy.Example:
    errors = "; ".join(ep.get("error_messages", [])[:2])
    return dspy.Example(
        task_prompt=ep.get("task_prompt", ""),
        error_context=errors,
        completion=_ideal_completion_from_episode(ep),
    ).with_inputs("task_prompt", "error_context")


def _ideal_completion_from_episode(ep: dict) -> str:
    parts = []
    outcome = ep.get("outcome", "unknown")
    if outcome == "success":
        parts.append("Successfully completed the task with minimal tool calls.")
    elif outcome == "error":
        parts.append(
            "Task encountered errors. The key issues were: "
            + "; ".join(ep.get("error_messages", ["unknown"])[:2])
        )
    cmds = ep.get("bash_commands", [])[:3]
    if cmds:
        parts.append("Key commands: " + "; ".join(cmds))
    return " ".join(parts) or "Task completed."
```

Also add a prominent docstring to `run_dspy_gepa()` noting that it extracts DSPy’s internal `signature.instructions` field, not a SKILL.md — so the output format differs from GEPA’s `optimize_anything` path.

-----

## 6. Add a length-constrained custom proposer

**File:** `optimize.py` → `run_gepa_optimize_anything()` / `run_gepa_synthetic()`  
**Severity:** 🟡 Quality — prevents overfitting at higher eval counts

The Decagon production study (50 ablation experiments) found unconstrained GEPA produces prompts exceeding 5,000 chars. A 1,500-char constraint achieved 4× compression with only 0.8% performance loss and better generalization. GEPA supports a `custom_candidate_proposer` hook in `ReflectionConfig`.

**Add a length-aware proposer factory to `optimize.py`:**

```python
def make_length_constrained_proposer(max_chars: int = 2000):
    """
    Returns a custom GEPA proposer that injects a character-limit constraint
    into the reflection prompt. Prevents prompt bloat from accumulating across
    iterations — the single largest overfitting risk in long optimization runs.
    
    Reference: Decagon ablation study (March 2026) — 1,500-char constraint
    achieved 4× compression, -0.8% performance, +generalization.
    """
    from gepa.optimize_anything import ProposalFn  # import lazily

    def proposer(current_candidate, reflective_dataset, components_to_update):
        # The default proposer's output is string | dict[str, str].
        # Inject the constraint into the reflection context by wrapping.
        constraint_note = (
            f"\n\nIMPORTANT: The skill MUST be under {max_chars} characters. "
            f"Prefer concise, specific bullets over verbose paragraphs. "
            f"If the current draft exceeds {max_chars} chars, aggressively prune "
            f"generic advice and keep only repo-specific, actionable guidance."
        )
        # Append the constraint to each example's feedback before proposing
        enriched = []
        for item in reflective_dataset:
            enriched.append({**item, "feedback": item.get("feedback", "") + constraint_note})
        # Delegate to GEPA's default proposer with enriched data
        return None  # returning None tells GEPA to use its default proposer with enriched data

    return proposer
```

Wire it in via `ReflectionConfig(custom_candidate_proposer=make_length_constrained_proposer())` and expose `--max-skill-chars` as a CLI flag (default: `2000`, matching the 2000-token SKILL.md target).

-----

## 7. Add warm-restart seeding to `watch_and_learn.py`

**File:** `watch_and_learn.py`  
**Severity:** 🟡 Efficiency — full re-optimization on every new session

The watcher accumulates new episodes and re-runs optimization from the original seed each time. GEPA has no built-in checkpoint resume, but the previous best candidate is always written to `outputs/<target>/best_candidate.md`. Seeding the next run from that file approximates warm-restarting without any framework changes.

**Pattern to add in the watch loop:**

```python
def _get_warm_seed(output_dir: Path, target: str, original_seed: str) -> str:
    """
    Use the previous run's best candidate as the seed for the next run.
    Falls back to the original seed if no prior output exists.
    """
    best_path = output_dir / target / "best_candidate.md"
    if best_path.exists():
        content = best_path.read_text(encoding="utf-8")
        print(f"[watch] Warm-restarting from previous best ({len(content)} chars)")
        return content
    return original_seed

# In the watch loop, before each optimize call:
seed = _get_warm_seed(output_dir, args.target, original_seed)
# Pass seed as --seed-file to the subprocess or inline call
```

This is especially valuable when corpus growth is slow (1–2 new sessions per day) and the marginal improvement per run is small. Starting from the previous best means GEPA refines rather than rediscovers.

-----

## 8. Fix `generate_tasks_for_domain()` token budget

**File:** `src/synthetic_evaluator.py`  
**Severity:** 🟡 Correctness — silent truncation above ~15 tasks

`EVAL_MAX_TOKENS = 512`. Task generation uses `EVAL_MAX_TOKENS * 8 = 4096`. Each task dict (description, context, pitfalls, criteria) consumes roughly 300–400 output tokens. At `--generate-tasks 15` the output is near the ceiling; at 20+ it silently truncates mid-JSON, `_parse_llm_json()` returns `[]`, and the run falls back to the built-in library without warning.

**Fix:**

```python
# In llm_config.py — add a dedicated constant
TASK_GEN_MAX_TOKENS = 8192   # sufficient for 20 tasks at ~400 tokens each
```

```python
# In synthetic_evaluator.py — replace the multiplier
resp = litellm.completion(
    model=judge_lm,
    messages=[...],
    max_tokens=TASK_GEN_MAX_TOKENS,   # was: EVAL_MAX_TOKENS * 8
    **INFERENCE_PARAMS,
)
```

Also add a warning when `_parse_llm_json()` returns fewer tasks than requested:

```python
tasks = _parse_llm_json(raw, [])
if isinstance(tasks, list) and len(tasks) < n:
    print(
        f"[synthetic] WARNING: requested {n} tasks but got {len(tasks)}. "
        f"Response may have been truncated. Consider reducing --generate-tasks."
    )
```

-----

## 9. Enrich `_JUDGE_SYSTEM` with SKILL.md format constraints

**File:** `src/evaluator.py`  
**Severity:** 🟡 Quality — judge lacks the structural rubric the structural scorer uses

The judge system prompt scores “how helpful” a skill would be but has no rubric for skill format. A 10,000-char sprawling document scores the same structurally as a concise 1,500-char one. Adding one sentence aligns the judge with the constraints the structural scorer already enforces.

**Current:**

```python
_JUDGE_SYSTEM = (
    f"{DIRECT_OUTPUT_PREFIX}\n\n"
    "You are a Claude Code skill evaluator. You will receive:\n"
    "- A candidate SKILL.md\n"
    "- A session episode (tool calls, errors, outcome)\n\n"
    "Score 0.0–1.0: how much would this skill have helped the agent?\n"
    "Consider: error prevention, tool efficiency, context window usage.\n\n"
    ...
)
```

**Fix — add one sentence after the scoring criteria:**

```python
"Consider: error prevention, tool efficiency, context window usage.\n"
"Format quality also matters: a well-formed skill is under 2000 tokens, "
"uses markdown headers and numbered lists, and contains repo-specific commands "
"rather than generic advice. Penalise skills that are vague, overly long, or "
"repeat information the agent would already know.\n\n"
```

-----

## 10. Tune `structural_score()` length bounds and specificity density

**File:** `src/synthetic_evaluator.py`  
**Severity:** 🟡 Accuracy — length heuristic penalizes the optimal range

The Decagon study found 1,500-char skills with high specificity outperform 5,000-char ones. The current length sweet spot (`800 <= n <= 2500 → 0.15`) doesn’t penalize verbose candidates enough and doesn’t reward concise-but-dense ones.

Also, specificity is counted as raw matches rather than density — a 4,000-char skill with 12 code references scores higher than a 1,200-char skill with 10, even if the shorter one is more actionable per word.

**Suggested changes to `structural_score()`:**

```python
# 1. Tighten the length sweet spot toward 600–1800 chars
if 600 <= n <= 1800:
    length_score = 0.15
elif 400 <= n < 600 or 1800 < n <= 3000:
    length_score = 0.08
elif n < 200:
    length_score = 0.0
else:
    length_score = 0.02   # was 0.04 — stronger penalty for bloat

# 2. Switch specificity from raw count to density
word_count = max(1, len(candidate.split()))
specificity_density = specificity_hits / word_count * 100   # hits per 100 words
specificity_score = min(0.20, specificity_density * 0.05)   # tune the multiplier
```

-----

## Summary

|# |Change                                             |File                                             |Impact           |
|--|---------------------------------------------------|-------------------------------------------------|-----------------|
|1 |Wire `neutral_closing` into `_outcome_score()`     |`src/evaluator.py`                               |🔴 Bug fix        |
|2 |Change default train/val split to 80/20            |`optimize.py`                                    |🔴 Accuracy       |
|3 |Raise ASI cap to 4,000; reorder `episode_to_asi()` |`src/evaluator.py`, `src/parse_session.py`       |🟠 Quality        |
|4 |Score all components in multi/nested evaluators    |`optimize.py`                                    |🟠 Accuracy       |
|5 |Extract shared DSPy code; document DSPy path limits|`optimize.py`, new `src/dspy_shared.py`          |🟠 Maintainability|
|6 |Add length-constrained custom proposer             |`optimize.py`                                    |🟡 Quality        |
|7 |Warm-restart seeding in `watch_and_learn.py`       |`watch_and_learn.py`                             |🟡 Efficiency     |
|8 |Dedicate `TASK_GEN_MAX_TOKENS`; warn on truncation |`src/synthetic_evaluator.py`, `src/llm_config.py`|🟡 Correctness    |
|9 |Add format rubric to `_JUDGE_SYSTEM`               |`src/evaluator.py`                               |🟡 Quality        |
|10|Tune `structural_score()` length bounds and density|`src/synthetic_evaluator.py`                     |🟡 Accuracy       |