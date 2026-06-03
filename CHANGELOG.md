# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 10: Evaluator signal enhancements** (2026-06-03)
  - Increased truncation limits in evaluator `side_info`: error_messages[:10], bash_commands[:20], files_written[:20], final_assistant_msg[:2000]
  - Enriched `_build_feedback` with bash command sequence (first 12), files touched (first 8), distinct error count
  - New YAML validation gate for `--target agent` with `_check_yaml_validity` helper
  - New `--time-split` CLI flag (default False): sorts episodes chronologically by timestamp
  - New `--proposer {batch,loop}` CLI flag (default batch): selects gepa 0.1.1 reflection proposer
  - `sort_by_time` parameter in `build_corpus()` for chronological episode ordering
  - Added `pyyaml>=6.0` dependency

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