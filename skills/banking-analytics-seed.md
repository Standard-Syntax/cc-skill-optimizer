# Banking Analytics & AI Pipeline Skills
# Generated seed — GEPA will evolve this from real session logs

## Repository Overview
This workspace contains:
- Power BI semantic models (TMDL format), DAX measures, and credit card analytics dashboards
- Python 3.13 multi-agent orchestration (RPIV pipeline: Research → Plan → Implement → Validate)
- TypeScript/Bun opencode plugins (safety hooks, quality enforcement)
- GitHub Actions workflows targeting self-hosted runner `blazar`
- SQL Server 2016 T-SQL queries for interchange revenue, payment compression, cure rates

## Critical Environment Facts
- Wayland compositor: **niri** (not i3, not sway). Window management differs.
- Python runtime: **uv** is the package manager. Never use pip directly.
  Run scripts with: `uv run python script.py`
  Install packages: `uv add package_name`
- Shell: bash with bash-it. `rg` > `grep`, `fd` > `find`, `bat` > `cat`, `eza` > `ls`
- Node/TypeScript: **Bun** for opencode plugins. `bun run` not `npx`.
- SQL Server 2016: T-SQL only, no window functions beyond what 2016 supports.

## Python 3.13 Development Patterns
- Always use `from __future__ import annotations` (postponed evaluation)
- Use `uv run` for scripts with inline dependencies (PEP 723)
- Pydantic v2 syntax: `model_config = ConfigDict(...)`, not `class Config`
- Async: prefer `anyio` for backend-agnostic async code
- Type checking: `ty` (not mypy). Run `ty check .` before marking done.
- Linting: `ruff check --fix .` then `ruff format .`

## Power BI / TMDL Patterns
- TMDL files live in `Model/` directory with `.tmdl` extension
- DAX measure best practices: use CALCULATE sparingly, prefer FILTER(ALL(...))
- Time intelligence: always use date table relationship, not DATEADD on fact table
- Semantic model deployment: `pbi-tools deploy` (not manual publish)

## Test-Running Strategy
1. For Python: `uv run pytest tests/ -q --tb=short` — check return code 0
2. For TypeScript/Bun plugins: `bun test` — watch for "X tests passed"
3. For DAX: run measure in Power BI Desktop report view and verify against known values
4. For T-SQL: execute against `blazar` SQL Server, compare row counts

## Common Error Patterns & Fixes
- **ModuleNotFoundError with uv**: Package not in venv. Run `uv add <package>` first.
- **Pydantic ValidationError**: Field types changed in v2. Check `model_fields` not `__fields__`.
- **niri Wayland crashes**: Don't use X11 display env vars. Use `WAYLAND_DISPLAY=wayland-1`.
- **opencode plugin not loading**: Check `opencode.json` — plugin must be in `plugins` array.
- **TMDL deploy fails**: Ensure `pbi-tools` version matches semantic model API version.

## RPIV Pipeline Conventions
- Research agent outputs: `research.md` in project root
- Plan agent outputs: `plan.md` + `tasks.md`
- Implement agent: works task-by-task from `tasks.md`, commits after each
- Validate agent: runs full test suite, writes `validation.md`
- Never skip the Validate stage — always run tests before marking complete.

## Legal Docs (pro se, LVNV case)
- Case: First City Court, Docket 2026-01926-1
- Use `la-credit-debt-defense` skill for procedure
- Use `la-civil-law-credit-doctrine` skill for substantive law
- Use `fcc-legal-typography` skill for formatting
- Always cite Louisiana Revised Statutes with full citation: `La. R.S. §X:XXX`

## File Organization Conventions
- Skills: `.claude/skills/<domain>/SKILL.md`
- Agent configs: `AGENTS.md` in project root
- Python projects: `src/<package>/` layout, `tests/` at root
- Outputs: `outputs/` directory, never commit to git directly

## When You're Stuck
1. Read the error carefully — most errors are self-explaining
2. Check if the file actually exists before reading: `fd <filename>`
3. For bash errors: check exit code, not just stderr
4. For import errors: verify `uv run` was used, not bare `python`
