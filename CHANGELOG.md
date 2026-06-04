# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phases 15-18: 10 improvements from cc-skill-optimizer-improvements.md** (2026-06-04)
  - **15.1**: Wire `neutral_closing` into `_outcome_score()` — unknown-outcome episodes with files written and no errors now score 0.7 instead of 0.5 (was a silent accuracy regression).
  - **15.2**: Change `split_corpus()` default to 80/20/0 — matches the GEPA FAQ recommendation; legacy 70/20/10 wasted 10% of every corpus on a test slice that was never used. New `--test-frac 0.0` flag preserves backward compatibility.
  - **16.1**: Raise ASI cap from 2000 to 4000 chars + lower candidate cap from 8000 to 6000 — reflection LM benefits from fuller tool_calls and complete test output (gskill/optimize_anything paper).
  - **16.2**: Reorder `episode_to_asi()` sections so highest-signal content (Outcome → Errors → Task → Final assistant message) appears first. Combines Files read + Files written into a single "Files touched" section.
  - **16.3**: Score all components in `make_multi_evaluator` and `make_nested_evaluator` — was scoring only the primary component. Now scores the full concatenated candidate AND populates `side_info["scores"]` with per-component scores for multi-objective Pareto via `frontier_type="hybrid"`. Wraps combined-call `base_evaluator` in try/except for graceful 0.0 fallback.
  - **17.1**: Create `src/dspy_shared.py` with `SkillGuidedTask`, `SkillProgram`, `ep_to_example`, `_ideal_completion_from_episode` extracted from inline duplicates in both `run_dspy_gepa` and `run_dspy_native_gepa`. Module docstring notes the DSPy path extracts `signature.instructions`, not a SKILL.md.
  - **17.2**: Refactor `optimize.py` to import the shared DSPy helpers; mipro metric now returns `dspy.Prediction(score, feedback)` (was `float`) for consistency with dspy 3.x's `GEPAFeedbackMetric` contract.
  - **17.3**: Test mock surface update — `tests/test_4_5_dspy_guard.py` and `tests/test_11_dspy_native_gepa.py` updated to verify the refactor.
  - **18.1**: Add `make_length_constrained_proposer(max_chars=2000)` factory and `--max-skill-chars` CLI flag. The proposer mutates `reflective_dataset` in-place to inject a length constraint into the reflection prompt. Decagon ablation study found 1,500-char constraint achieved 4× compression with 0.8% performance loss.
  - **18.2**: Add warm-restart seeding to `watch_and_learn.py` via `_get_warm_seed(output_dir, target, original_seed)` helper — uses the prior run's `best_candidate.md` as the next seed. New `--output-dir` and `--target` CLI flags.
  - **18.3**: Add `TASK_GEN_MAX_TOKENS = 8192` constant in `src/llm_config.py` and use it in `src/synthetic_evaluator.py:generate_tasks_for_domain`. The legacy `EVAL_MAX_TOKENS * 8` (4096) was too tight for 20+ tasks; truncation caused silent fallback to the built-in library. New truncation warning fires when fewer tasks are returned than requested.
  - **18.4**: Enrich `_JUDGE_SYSTEM` in `src/evaluator.py` with a SKILL.md format-rubric sentence — well-formed skills are under 2000 tokens, use markdown headers and numbered lists, contain repo-specific commands rather than generic advice. Aligns the LLM judge with the structural scorer.
  - **18.5**: Tune `structural_score()` length bounds from [800, 2500] → [600, 1800] (Decagon 1,500-char optimum), strengthen bloat penalty 0.04 → 0.02, switch specificity scoring from raw count `hits * 0.015` to density-based `(hits / word_count * 100) * 0.05` (capped at 0.20).
  - 141 new tests across 11 new test files (and 1 modified test file); 0 regressions across all 4 phases' regression suites; Phase 14 invariant (no `dspy.configure`) intact.

- **Phase 20: Fix DSPy skill extraction — `best_candidate_dspy.md` returns optimized SKILL content** (2026-06-04)
  - **20.1**: Refactor `src/dspy_shared.py` so the seed skill content lives in the predictor's `signature.instructions` field via `SkillGuidedTask.with_instructions(skill_content)`. Previously `skill_content` was held as a constant instance attribute and passed as a `skill_instructions` input field — which MIPROv2 / dspy.GEPA do NOT optimize. They optimize `signature.instructions` instead. As a result, the optimized output was the static `SkillGuidedTask` class docstring (288 bytes), not the user's SKILL.md (17,944 bytes). After the fix, both `run_dspy_gepa` and `run_dspy_native_gepa` extract the optimized SKILL content via the existing `optimized.predictor.signature.instructions` read at `optimize.py:1218-1226` and `:1318-1325`.
  - **20.2**: Update the docstrings of `run_dspy_gepa` and `run_dspy_native_gepa` in `optimize.py` to remove the misleading "NOTE — DSPy path output format" paragraph that incorrectly claimed the output "differs from GEPA's `optimize_anything` path" and was "NOT a SKILL.md". The DSPy path now correctly produces a SKILL.md-shaped output.
  - **20.3**: New `tests/test_20_dspy_extraction_e2e.py` with 10 E2E regression tests using mocked `dspy.MIPROv2` and `dspy.GEPA` optimizers. Includes a size-bounds assertion (output must be ≥0.3x of seed size) that catches the original 288-vs-17944 regression direction.
  - **20.4**: Final QA — fixed 2 F841 lint errors in `tests/test_20_dspy_extraction_e2e.py` and updated the pre-existing test at `tests/test_11_dspy_native_gepa.py:438` that was asserting the old buggy docstring invariant (`"SKILL.md" in doc`) and was therefore locking in the wrong behavior.
  - 27 new tests across 3 new test files; 84/84 dspy-related tests pass (57 pre-existing + 27 new); `ruff check .` reports 0 errors.

### Changed
- `split_corpus()` function gained a new `test_frac` parameter (default 0.0); old 4-arg positional callers (without test_frac) will fail with TypeError — the only such caller was already updated in `main()`.
- MIPROv2 metric return type changed from `float` to `dspy.Prediction(score, feedback)` for dspy 3.x `GEPAFeedbackMetric` contract compatibility.

### Migration
- No breaking changes; all new CLI flags have default values matching prior behavior.

- **Phase 12: Test surface + Final QA** (2026-06-03)
  - New `tests/test_11_dspy_native_gepa.py` with 14 tests covering:
    - `--dspy-backend native-gepa` CLI flag recognition
    - `dspy.GEPA` top-level import verification
    - `dspy.Prediction` metric return from native GEPA
    - `side_info["scores"]` multi-objective threading
  - Updated test mocks for dspy 3.x dual import paths:
    - `mock_dspy_modules()` now mocks both `dspy.GEPA` (top-level) and `dspy.teleprompt.GEPA` (legacy)
    - Extraction tests patch both `RealDSPy.MIPROv2` and `RealDSPy.teleprompt.MIPROv2`
  - 72/74 total tests pass (2 pre-existing failures in test_optimize_fixes.py from defunct gepa.optimize_anything mock)

- **Phase 14: Per-module LM injection** (2026-06-04)
  - Migrated `dspy.configure(lm=...)` → `Program.set_lm(task_lm_obj)` per-module LM injection
  - Updated test mocks for dspy 3.0 Module.set_lm and Module.map_named_predictors API
  - 81/81 tests pass

- **Phase 11: DSPy 3.0 upgrade + dspy.GEPA native backend** (2026-06-03)
  - Upgraded from dspy 2.x to dspy>=3.0.0,<4.0.0 (dspy 3.0+ required)
  - litellm>=1.64.0 is now required (dspy 3.x hard-requires this version)
  - New `--dspy-backend {mipro,native-gepa}` CLI flag:
    - `mipro` (default): Legacy dspy.MIPROv2 path — backward compatible
    - `native-gepa`: Uses dspy.GEPA (dspy 3.0+ native multi-objective optimizer) with reflective feedback
  - New `run_dspy_native_gepa()` function in optimize.py using dspy.GEPA
  - geqvist override-dependencies in pyproject.toml resolves dspy 3.x gepa pin conflict
  - Installation: `uv sync` handles automatically — no manual steps needed

- **Phase 10: Evaluator signal enhancements** (2026-06-03)
  - Increased truncation limits in evaluator `side_info`: error_messages[:10], bash_commands[:20], files_written[:20], final_assistant_msg[:2000]
  - Enriched `_build_feedback` with bash command sequence (first 12), files touched (first 8), distinct error count
  - New YAML validation gate for `--target agent` with `_check_yaml_validity` helper
  - New `--time-split` CLI flag (default False): sorts episodes chronologically by timestamp
  - New `--proposer {batch,loop}` CLI flag (default batch): selects gepa 0.1.1 reflection proposer
  - `sort_by_time` parameter in `build_corpus()` for chronological episode ordering
  - Added `pyyaml>=6.0` dependency
  - New test coverage: `test_bash_command_sequence_included`, `test_files_written_pattern_included`, `test_distinct_error_count_included` in `tests/test_evaluator_fixes.py`

- **Phase 9: Multi-objective Pareto frontier** (2026-06-03)
  - New `scores` dict in evaluator `side_info` with 4 normalized keys: outcome, efficiency, cache_efficiency, low_error_rate
  - New `--hybrid-frontier` CLI flag: enables gepa 0.1.1 multi-objective Pareto tracking
  - Fixed invalid `frontier_type="population"` → `"instance"` for Phase 2 (gepa 0.1.1 compatibility)
- New `src/utils.py` module with shared `_parse_llm_json` helper function
- `--phase` CLI flag to `optimize.py`:
  - `--phase 1` (default): Synthetic exploration mode, 100 evals, 4-thread parallel evaluation
  - `--phase 2`: Session-backed refinement, 60 evals (requires existing session logs)
- `skip_paths` parameter to `build_corpus()` for cumulative dedup across watch sessions
- `neutral_closing` field in episode dict — scores unknown outcomes with files written at ~0.7

### Changed
- **GEPA reflection feedback**: Evaluators now populate `side_info["feedback"]` with natural-language diagnostic strings instead of just numeric scores. Reflection LM receives actionable feedback (e.g., "EACCES: permission denied" → "skill should warn about permission issues")
- **EngineConfig parallel**: Set to `True` by default (4 threads), reducing wall time 4-8x
- **Conservative outcome inference**: Ambiguous sessions now score 0.5 (was 1.0). Uses `POSITIVE_COMPLETION_SIGNALS` for safer inference
- **Section parser idempotency**: `_heading_<key>` metadata enables round-trip across GEPA iterations without drift
- **LLM configuration**: Updated from Anthropic models to MiniMax models. `DEFAULT_MODEL` is now `minimax/minimax-m2.7-highspeed`, `REFLECTION_MODEL` is now `minimax/minimax-m3`. Base URL now includes `/v1` suffix (`https://api.minimax.io/anthropic/v1`)
- **LLM judge context**: Truncation raised from 3000 to 8000 chars for full skill visibility
- **`oa.log()` ASI channel**: 6 diagnostic calls (outcome, duration, errors, tool calls, compaction, judge score) flow to reflection LM via `oa.get_log_context().drain()`
- **DSPy MIPROv2 extraction**: Optimized instructions and few-shot demos injected into output skill
- **Judge weight**: Standardized to 0.65 in both replay and synthetic evaluators
- **`llm_config.configure()`**: Must now be called explicitly (no module-level side effect)
- **Subagent scanning**: Now scans both top-level JSONLs and `project_dir/subagents/agent-*.jsonl`
- **Size stability check**: Two-poll size tracking replaces mtime guard for file stability

### Fixed
- token_stats double-counting in `parse_session.py` (deduplicates by message.id)
- multi-evaluator temp dir scoping in `optimize.py`
- NESTED_ROOT hardcoding issue in `optimize.py` (now from CLI arg)
- DSPy extraction warning when no DSPy signatures found
- Regex pre-compilation in `synthetic_evaluator.py`
- Backup file rotation (max 5 backups) in `watch_and_learn.py`
- `--max-evals` CLI override was silently ignored (now works)
- Heading de-collision with `_2`, `_3` suffixes
- Section parser whitespace stripping to prevent round-trip accumulation
- Nested evaluator prefers root keys (no `/`) over nested

### Removed
- MIPROv2 docstring references (replaced with direct GEPA configuration)
- `_cache_bonus` from score formula (exposed via `side_info["cache_ratio"]` instead)

---

## [0.2.0] - 2026-06-03

### Added
- Phase 1 code review remediation (27 issues fixed across 7 source files)
- 102/102 tests pass; zero regressions

### Changed
- All Phase 1 features merged from unreleased (see [Unreleased] above)

---

## [0.1.0] - 2025-12-XX

### Added
- Initial release of cc-skill-optimizer
- GEPA prompt optimization with Reflective Text Evolution
- Section parser for within-file optimization
- Nested file discovery and optimization
- DSPy integration for program-level optimization

[Unreleased]: https://github.com/Standard-Syntax/cc-skill-optimizer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Standard-Syntax/cc-skill-optimizer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Standard-Syntax/cc-skill-optimizer/releases/tag/v0.1.0