# cc-skill-optimizer — Research Report

## GEPA 0.1.1 + DSPy 3.3.0b1 Improvements

*Researched June 2026 — Sources: GEPA docs, DSPy 3.3.0b1 release notes, gskill/optimize_anything papers, Hermes Agent self-evolution, Arize prompt-learning blog*

-----

## Executive Summary

The GEPA and DSPy ecosystems have evolved significantly since this project was built. GEPA 0.1.1 (March 2026) and DSPy 3.3.0b1 (May 2026) introduce concrete APIs that the cc-skill-optimizer currently doesn’t use: multi-objective Pareto tracking, per-predictor tool-description optimization, candidate lineage visualization, top-K Pareto selection, multi-objective `"scores"` side-info dict, evaluation caching, `frontier_type="hybrid"`, and the `gskill`/`optimize_anything` paper’s validated SI (Side Information) design patterns. Each section below maps directly to a file or function in the current codebase.

-----

## 1. GEPA 0.1.1 — New APIs and Breaking Changes

### 1.1 `DspyGEPAResult` Shape Changed

The upstream `gepa[dspy]==0.1.1` changed result structures that the current project may not account for:

|Old (pre-0.1.1)                         |New (0.1.1)                                                               |
|----------------------------------------|--------------------------------------------------------------------------|
|`candidates` = list of instruction dicts|`candidates` = list of compiled DSPy modules                              |
|`best_candidate` = dict                 |`best_candidate` = compiled DSPy module                                   |
|`val_subscores` keyed by int            |`val_subscores` = `list[dict[Any, float]]` keyed by validation instance ID|
|`per_val_instance_best_candidates`      |now `dict[Any, set[int]]`                                                 |
|`best_outputs_valset`                   |now `dict[Any, list[tuple[int, Prediction]]]`                             |
|`highest_score_achieved_per_val_task`   |now a dict keyed by validation instance ID                                |

**Action for `optimize.py`:** Audit any code that unpacks `result.candidates` or `result.detailed_results` to use the new shapes. The `run_dspy_gepa()` function likely breaks on 0.1.1 when it tries to extract instructions from candidate dicts.

### 1.2 Reflection Template Placeholders Renamed

GEPA 0.1.1 renamed default placeholders from `<curr_param>` / `<side_info>` to new names. In `dspy.GEPA`, passing `reflection_prompt_template` through `gepa_kwargs` now raises a `ValueError`; use `instruction_proposer` for custom proposal behavior.

**Action:** If any custom reflection templates exist in `optimize.py`, update placeholder names or switch to `instruction_proposer`.

### 1.3 New `top_k_pareto` Candidate Selector

GEPA 0.1.1 ships `candidate_selection_strategy="top_k_pareto"` (default K=5), which limits parent selection to the top-K candidates by aggregate score. This focuses mutation effort on the most promising programs rather than sampling from the entire Pareto front — useful when the corpus is small (20–50 episodes) and you want faster convergence.

**Action in `optimize.py`:** Expose `--candidate-strategy` CLI flag. Current default is `"pareto"`. For small banking analytics corpora, `"top_k_pareto"` or `"epsilon_greedy"` (0.1 random exploration) may produce better results.

### 1.4 Candidate Tree Visualization

GEPA 0.1.1 auto-generates an interactive HTML lineage tree of all explored candidates, color-coded by role (best, Pareto front, seed) with hover previews. Logged to WandB and MLflow at each step. This is free if `use_wandb=True` or `use_mlflow=True` is passed.

**Action:** Add `--wandb` / `--mlflow` flags to `optimize.py` as a one-liner passthrough to `GEPAConfig(engine=EngineConfig(use_wandb=True))`.

-----

## 2. Multi-Objective Pareto Tracking (Missing in Current Evaluator)

### 2.1 The `"scores"` Side-Info Key

GEPA 0.1.1’s `optimize_anything` API supports multi-objective Pareto tracking when the evaluator returns a `"scores"` dict inside `side_info`:

```python
return score, {
    "score": score,
    "scores": {          # ← NEW: enables multi-objective Pareto frontier
        "outcome": outcome_score,
        "efficiency": eff_score,
        "compactness": 1.0 - compaction_penalty,
        "error_recovery": error_recovery_score,
    },
    ...
}
```

All `"scores"` values must follow higher-is-better. GEPA maintains per-objective frontiers automatically. This means a candidate that’s excellent at reducing tool calls but mediocre overall stays on the frontier, rather than being pruned.

**Current gap:** `evaluator.py`’s `score_episode()` and `make_replay_evaluator()` return a single blended scalar. The components (`outcome_score`, `efficiency_bonus`) exist but are blended before GEPA sees them.

**Action in `src/evaluator.py`:**

```python
def score_episode(episode, thresholds=None):
    base = _outcome_score(episode)
    eff = _efficiency_bonus(episode, thresholds)
    score = max(0.0, min(1.0, base + eff))
    cache_ratio = _compute_cache_ratio(episode)

    side_info = {
        "score": score,
        "scores": {                         # ← add this block
            "outcome": base,
            "efficiency": (eff + 0.15) / 0.30,   # normalize to [0,1]
            "cache_efficiency": cache_ratio,
            "low_error_rate": 1.0 if not episode.get("error_messages") else 0.0,
        },
        ...
    }
    return score, side_info
```

Then set `frontier_type="hybrid"` in the `GEPAConfig` to exploit both instance-level and objective-level diversity.

### 2.2 `frontier_type` Selection Guide

|Scenario                                         |Recommended                   |Why                               |
|-------------------------------------------------|------------------------------|----------------------------------|
|Single metric, small corpus (20–50 episodes)     |`"instance"` (current default)|Per-episode specialization        |
|Multi-objective scoring (after adding `"scores"`)|`"hybrid"`                    |Instance + objective keys together|
|Very small corpus (<15 episodes)                 |`"objective"`                 |Avoid front explosion             |
|Large corpus (150+ episodes)                     |`"cartesian"`                 |Maximum diversity                 |

-----

## 3. gskill / `optimize_anything` Paper — Validated SI Design Patterns

The February 2026 gskill blog and the May 2026 `optimize_anything` paper (arXiv:2605.19633) provide the only peer-validated SI design for the exact use case — learning skills for coding agents from session traces. Key findings:

### 3.1 What Goes in Side Information

The gskill evaluator returns:

```python
side_info = {
    "task_description": task["task_description"],
    "agent_trace": {
        "tool_calls": episode["tool_calls"][:20],     # full tool sequence
        "code_edits": episode["files_written"],
        "errors": episode["error_messages"],
    },
    "test_outcome": {
        "passed": test_pass,
        "output": test_output[:1000],                  # actual test stderr
    },
    "resolution_time_s": episode.get("duration_s"),
    "scores": {
        "pass_rate": float(test_pass or False),
        "speed": speed_score,                          # normalized 1/duration
    }
}
```

**Current gap in `src/evaluator.py`:** The current `side_info` truncates `bash_commands` to 8, `files_written` to 6, `error_messages` to 3, and `final_assistant_msg` to 400 chars. The paper reports that the reflection LM benefits most from full `tool_calls` and complete test output. Increase truncation limits, especially for `error_messages` and `bash_commands`.

### 3.2 Skills Transfer Across Models Without Re-optimization

The paper’s most actionable finding: skills learned on a cheap model (gpt-5-mini via mini-SWE-Agent) transferred directly to Claude Code with Haiku 4.5 and Sonnet 4.5 without reoptimization, boosting pass rates from 79.3% → 98.3% on Bleve and 55% → 82% on Jinja. This validates the current project’s REPLAY mode — you can optimize cheaply with MiniMax M2.7-highspeed and deploy to Claude Sonnet without re-evaluation.

### 3.3 The Loop Proposer vs. Batch Proposer

gskill ships two proposer strategies:

- **Batch proposer** (default): Sends all evaluation results to the reflection LM at once. Fewer API calls, but the LM may not focus on any single failure mode.
- **Loop proposer**: Processes each evaluation result one at a time, then merges. More API calls, but produces more detailed skills because the reflection LM focuses on one failure at a time.

The `--proposer loop` flag is available in gskill but has no equivalent in cc-skill-optimizer. This maps to `custom_candidate_proposer` in GEPA’s `optimize_anything` API.

**Action in `optimize.py`:** Add `--proposer {batch,loop}` flag. In batch mode (current behavior), pass all episode side_info at once. In loop mode, run one episode per reflection call and use GEPA’s merge to combine the evolved candidates.

-----

## 4. DSPy 3.3.0b1 — `dspy.GEPA` and `ReActV2` Changes

### 4.1 Per-Predictor Tool Description Optimization

DSPy PR #8928 (merged December 2025, shipping in 3.3.0b1) adds `enable_tool_optimization=True` to `dspy.GEPA`. When enabled, GEPA jointly optimizes:

- ReAct module instructions
- Extract predictor instructions
- Individual tool descriptions and argument descriptions

This is directly applicable to optimizing AGENTS.md-style orchestration configs where tool routing is described in natural language. A `ToolProposer` with a specialized reflection prompt discovers patterns in successful vs. unsuccessful tool usage.

**Action:** For the `--target agent` mode, enable tool description optimization by wrapping the AGENTS.md evaluation as a `dspy.ReAct` program rather than bare text. This lets GEPA optimize the role descriptions, routing rules, and tool descriptions jointly.

### 4.2 `ReActV2` with Structured History

DSPy 3.3.0b1 ships `dspy.ReActV2` with native tool calls and structured `dspy.History` instead of a flat `trajectory` string. The history is now structured per-turn, enabling prompt caching (50% cost reduction in internal tests). For LIVE mode evaluations that invoke Claude Code repeatedly, switching the mock/judge call to use `ReActV2` would reduce reflection costs.

### 4.3 GEPA `"improvement_or_equal"` Acceptance Criterion

The `acceptance_criterion="improvement_or_equal"` option (vs. default `"strict_improvement"`) allows GEPA to keep candidates that tie the current best. For small corpora where the score landscape is flat, this helps escape plateau stagnation.

**Action in `optimize.py`:** Expose `--acceptance-criterion` flag; set to `"improvement_or_equal"` automatically when corpus size is below 20 episodes.

-----

## 5. Evaluator Signal Improvements

### 5.1 Richer Feedback String Construction

The current `_build_feedback()` generates 1–3 sentences. The GEPA DSPy docs emphasize that the quality of the `feedback` string directly determines mutation quality. Based on the gskill SI design:

**Add to `_build_feedback()` in `src/evaluator.py`:**

```python
# Current: mentions errors, tool count, compaction
# Add: bash command sequence summary, files-written pattern, outcome confidence

if episode.get("bash_commands"):
    cmds = episode["bash_commands"][:12]
    parts.append(f"Commands run: {'; '.join(cmds)}")

if episode.get("files_written"):
    parts.append(f"Files touched: {', '.join(episode['files_written'][:8])}")

n_errors = len(episode.get("error_messages", []))
if n_errors > 1:
    parts.append(f"{n_errors} distinct errors encountered.")

if episode.get("neutral_closing"):
    parts.append("Session ended with likely completion (neutral close signal).")
```

### 5.2 Evaluation Caching (`cache_evaluation=True`)

GEPA 0.1.1’s `optimize_anything` API supports `cache_evaluation=True`, which caches `(score, output, objective_scores)` per `(candidate, example)` pair. For the watch-and-learn daemon, which re-evaluates across overlapping episode sets, this could eliminate 30–50% of redundant judge calls.

**Action in `src/synthetic_evaluator.py` and `watch_and_learn.py`:** Pass `cache_evaluation=True` in `EngineConfig` for runs that might re-use episodes across iterations.

### 5.3 Two-Pass Judge Scoring for Long Skills

The current `llm_judge_score()` truncates the candidate skill to 8000 chars before the judge sees it. The GEPA evaluate-anything paper recommends a two-pass approach for artifacts over this limit: split the candidate at its midpoint, judge each half, average the scores. This is especially relevant for multi-component targets where individual sections may exceed 8000 chars.

**Action in `src/evaluator.py`:** Implement `_two_pass_judge_score()` that activates when `len(candidate_skill) > 8000`.

-----

## 6. CLAUDE.md / AGENTS.md Content Improvements

### 6.1 Repository-Split Training Strategy

The Arize Prompt Learning paper (November 2025) reports a +10.87% improvement when training CLAUDE.md on past issues from the same repository vs. a +5.19% improvement on cross-repository generalization. The current `--project-filter` flag achieves the in-repo split. However, the paper recommends a time-based train/test split within the same repo (not random) to simulate the actual developer workflow — you train on older sessions, test on newer ones.

**Action in `optimize.py`:** Add `--time-split` flag. When set, `split_corpus()` sorts episodes by `timestamp` (already parsed by `parse_session.py`) and splits chronologically rather than randomly. This better matches real-world deployment and avoids data leakage from future sessions.

### 6.2 AGENTS.md Optimization Gaps

The Claudelab.net guide (March 2026) reports AGENTS.md has become a standard with 60,000+ GitHub repos. Current AGENTS.md optimization in cc-skill-optimizer targets the content of role descriptions but doesn’t model the YAML-plus-Markdown hybrid format that AGENTS.md now uses in multi-tool environments. When evolving AGENTS.md content, the reflection LM should be told that changes must remain valid YAML.

**Action in `optimize.py`** for `--target agent`:

```python
# In OBJECTIVE_BY_TARGET["agent"], append:
# "Preserve YAML validity. Role constraints use YAML lists.
#  Verification commands must be shell-executable strings.
#  Prohibitions must be imperative sentences."
```

Also add a YAML validation step in the evaluator for agent candidates before scoring — invalid YAML should score 0.0 immediately rather than failing at judge time.

### 6.3 Step-Depth Hierarchy for CLAUDE.md Sections

The Claudelab guide describes CLAUDE.md “Step 0–8 workflow” patterns as best practice. The current `--section-depth` parameter already handles `##` and `###` headings. A new `--preserve-ordering` flag would prevent GEPA from reordering sections (which breaks Step 0–8 workflows where sequence is critical).

**Action in `src/section_parser.py`:** Add a `preserve_order=True` option to `merge_sections()` that reconstructs sections in the original document order even after independent evolution.

-----

## 7. Hermes Agent Self-Evolution — Applicable Patterns

The NousResearch Hermes Agent Self-Evolution project (Context7 verified, benchmark score 89.67) implements a 6-step autonomous optimization loop that extends the cc-skill-optimizer’s `watch_and_learn.py` concept:

1. Select target (SKILL.md, tool description, prompt section, code)
1. Build evaluation dataset from session DB
1. Wrap as DSPy module (Signature → ReAct → Predict)
1. Run GEPA (primary) or MIPROv2 (fallback for few-shot)
1. Statistical significance check before accepting
1. Git commit + optional A/B test + rollback via `git revert`

Steps 5 and 6 are missing from the current cc-skill-optimizer. Specifically:

**Missing: Statistical significance check.** The current project accepts any improvement over the previous best. Adding a two-sample t-test or bootstrap confidence interval before writing `best_candidate.md` would reduce false-positive “improvements” that reflect noise rather than genuine gains.

**Missing: Git-backed rollback.** Currently, `watch_and_learn.py` uses a backup rotation system (max 5 backups). A cleaner approach is to commit each accepted candidate to a `gepa-optimized` branch, enabling `git revert` to undo any degraded optimization.

**Action in `watch_and_learn.py`:** Add `--git-commit` flag. When set, each accepted skill update triggers `git add` + `git commit -m "gepa: skill iteration N (score: X.XX)"` on a dedicated branch. The backup rotation system remains as a fallback.

-----

## 8. Priority Implementation Roadmap

|Priority|Item                                                             |Effort |Expected Gain                                                                          |
|--------|-----------------------------------------------------------------|-------|---------------------------------------------------------------------------------------|
|P0      |Multi-objective `"scores"` dict in evaluator                     |1–2 hrs|Better Pareto diversity; stops good-efficiency/bad-outcome candidates from being pruned|
|P0      |GEPA 0.1.1 `DspyGEPAResult` shape compatibility                  |1 hr   |Prevents crash in `run_dspy_gepa()` on current gepa version                            |
|P1      |`frontier_type="hybrid"` in EngineConfig                         |30 min |Exploits both instance and objective diversity                                         |
|P1      |`--candidate-strategy` CLI flag with `top_k_pareto`              |1 hr   |Faster convergence on small corpora                                                    |
|P1      |Richer feedback string (bash commands, file pattern, error count)|1–2 hrs|Directly improves reflection LM mutation quality                                       |
|P1      |`cache_evaluation=True` in EngineConfig                          |30 min |30–50% API cost reduction in watch-and-learn daemon                                    |
|P2      |`--time-split` chronological corpus split                        |1 hr   |Better generalisation signal; avoids data leakage                                      |
|P2      |YAML validation gate for `--target agent` candidates             |1 hr   |Eliminates wasted judge calls on syntactically invalid AGENTS.md                       |
|P2      |Loop proposer mode (`--proposer loop`)                           |2–3 hrs|More targeted per-failure reflection; matches gskill validated approach                |
|P2      |Two-pass judge scoring for skills >8000 chars                    |2 hrs  |Eliminates truncation bias for large multi-component targets                           |
|P3      |Reflection template placeholder update (GEPA 0.1.1)              |30 min |Removes deprecation warning / ValueError                                               |
|P3      |`--acceptance-criterion improvement_or_equal` auto-detect        |1 hr   |Helps small corpora escape score plateaus                                              |
|P3      |`--git-commit` rollback support in `watch_and_learn.py`          |2 hrs  |Safer continuous daemon operation                                                      |
|P3      |Per-predictor tool description optimization for `--target agent` |3–4 hrs|Joint optimization of role + tool descriptions                                         |
|P3      |WandB/MLflow candidate tree visualization                        |1 hr   |Free with GEPA 0.1.1; provides full optimization lineage                               |

-----

## 9. Reflection Template Guidance (from gskill)

The gskill paper’s loop proposer uses a reflection prompt that explicitly asks the LM to:

1. Identify what the agent did wrong (not just that it failed).
1. Propose a skill rule that would have prevented the failure.
1. Verify the proposed rule doesn’t contradict existing rules.
1. Format the rule as a single, imperative, actionable sentence.

This is more specific than the current GEPA default. Consider passing a `reflection_prompt_template` (or `instruction_proposer`) that enforces this structure for the skill optimization target.

**Draft addition to the `--target skill` objective string:**

```
Each proposed SKILL.md update must:
1. Address a specific observed failure mode from the session trace.
2. Be expressed as a single actionable rule (imperative sentence, ≤25 words).
3. Not contradict or duplicate any existing rule in the current skill.
4. Be placed in the most relevant existing section, not appended as a new section.
```

-----

*Confidence after research: 9/10 — All findings are grounded in primary sources (GEPA docs, DSPy release notes, peer-reviewed papers, or well-sourced blog posts from November 2025–June 2026). One uncertainty: GEPA 0.1.1 reflection template placeholder rename — exact new names were not shown in search results; verify in the `gepa` changelog before updating templates.*