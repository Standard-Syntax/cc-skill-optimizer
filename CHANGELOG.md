# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `src/utils.py` module with shared `_parse_llm_json` helper function
- `--phase` CLI flag to `optimize.py`:
  - `--phase 1` (default): Synthetic exploration mode, 100 evals, 4-thread parallel evaluation
  - `--phase 2`: Session-backed refinement, 60 evals (requires existing session logs)

### Changed
- **GEPA reflection feedback**: Evaluators now populate `side_info["feedback"]` with natural-language diagnostic strings instead of just numeric scores. Reflection LM receives actionable feedback (e.g., "EACCES: permission denied" → "skill should warn about permission issues")
- **EngineConfig parallel**: Set to `True` by default (4 threads), reducing wall time 4-8x
- **Conservative outcome inference**: Ambiguous sessions now score 0.5 (was 1.0). Uses `POSITIVE_COMPLETION_SIGNALS` for safer inference
- **Section parser idempotency**: `_heading_<key>` metadata enables round-trip across GEPA iterations without drift
- **LLM configuration**: Updated from Anthropic models to MiniMax models. `DEFAULT_MODEL` is now `minimax/minimax-m2.7-highspeed`, `REFLECTION_MODEL` is now `minimax/minimax-m3`. Base URL now includes `/v1` suffix (`https://api.minimax.io/anthropic/v1`)

### Fixed
- token_stats double-counting in `parse_session.py` (deduplicates by message.id)
- multi-evaluator temp dir scoping in `optimize.py`
- NESTED_ROOT hardcoding issue in `optimize.py` (now from CLI arg)
- DSPy extraction warning when no DSPy signatures found
- Regex pre-compilation in `synthetic_evaluator.py`
- Backup file rotation (max 5 backups) in `watch_and_learn.py`

### Removed
- MIPROv2 docstring references (replaced with direct GEPA configuration)

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