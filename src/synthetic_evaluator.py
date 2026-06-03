"""
synthetic_evaluator.py
======================
GEPA-compatible evaluator that works without any Claude Code sessions.

Two layers:

  1. LLM judge  — scores how much the candidate skill would help an agent
     complete a hand-written or auto-generated task description.

  2. Structural scorer — fast heuristic that doesn't cost API calls:
     checks length, specificity, presence of key sections, etc.
     Used as a secondary signal alongside the judge.

The module also ships built-in task libraries for common domains so you
can bootstrap without writing any tasks yourself.

Usage (standalone)
------------------
    from synthetic_evaluator import (
        make_synthetic_evaluator,
        load_task_library,
        generate_tasks_for_domain,
    )

    tasks = load_task_library("banking")         # 20 built-in tasks
    # or:
    tasks = generate_tasks_for_domain(           # LLM-generated tasks
        domain="my-custom-domain",
        domain_description="Python service that processes SWIFT messages ...",
        judge_lm="anthropic/claude-haiku-4-5-20251001",
        n=15,
    )

    evaluate = make_synthetic_evaluator(
        task_library=tasks,
        judge_lm="anthropic/claude-haiku-4-5-20251001",
    )

Usage in optimize_anything
--------------------------
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig
    from synthetic_evaluator import make_synthetic_evaluator, load_task_library

    tasks = load_task_library("banking")
    train, val = tasks[:14], tasks[14:]

    evaluate = make_synthetic_evaluator(tasks, judge_lm="anthropic/claude-haiku-4-5-20251001")

    result = optimize_anything(
        seed_candidate=open("my-seed.md").read(),   # or None for seedless
        evaluator=evaluate,
        dataset=train,
        valset=val,
        objective="Optimize a SKILL.md for a banking analytics repo.",
    )
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

# Import llm_config constants (also sets env vars as side-effect when available)
try:
    from llm_config import (
        DEFAULT_MODEL,
        DIRECT_OUTPUT_PREFIX,
        EVAL_MAX_TOKENS,
        EXTRA_BODY,
        INFERENCE_PARAMS,
        THINKING_CONFIG_EVAL,
    )
except ImportError:
    DEFAULT_MODEL = "minimax/minimax-m2.7-highspeed"
    INFERENCE_PARAMS = {}
    EXTRA_BODY = {}
    EVAL_MAX_TOKENS = 512
    THINKING_CONFIG_EVAL = {"type": "disabled"}
    DIRECT_OUTPUT_PREFIX = "Respond directly and concisely. Output only what is requested."

from utils import _parse_llm_json

# ---------------------------------------------------------------------------
# GEPA ASI channel
# oa.log() is a no-op outside the gepa wrapper context (e.g. unit tests).
# The wrapper drains LogContext and injects it as side_info["log"]; the
# reflection LM reads side_info["feedback"] — so we also append the captured
# log string to feedback below.
# ---------------------------------------------------------------------------
try:
    import gepa.optimize_anything as oa
except ImportError:

    class _NoOpOa:
        @staticmethod
        def log(*args, **kwargs):  # noqa: ARG001
            pass

        @staticmethod
        def get_log_context():
            class _NoOpCtx:
                def drain(self) -> str:
                    return ""

            return _NoOpCtx()

    oa = _NoOpOa()

# ---------------------------------------------------------------------------
# Built-in task libraries
# Each task is a dict with:
#   task_description  : str   — what the agent is asked to do
#   domain_context    : str   — extra context the judge should know
#   pitfalls          : list  — known failure modes the skill should prevent
#   success_criteria  : list  — what a good agent response looks like
# ---------------------------------------------------------------------------

_BANKING_TASKS: list[dict] = [
    {
        "task_description": "Add a DAX measure for trailing-twelve-month net interchange income",
        "domain_context": "Power BI semantic model, TMDL format, SQL Server 2016 date table",
        "pitfalls": [
            "using DATEADD instead of date table relationship",
            "incorrect CALCULATE filter context",
            "missing ALL() on date table",
        ],
        "success_criteria": [
            "uses CALCULATE with DATESINPERIOD or DATESYTD",
            "references date table relationship",
            "adds measure to correct table",
        ],
    },
    {
        "task_description": "Debug a ModuleNotFoundError for litellm in a uv-managed project",
        "domain_context": "Python 3.13, uv package manager, Debian Trixie",
        "pitfalls": [
            "running bare python instead of uv run",
            "installing with pip instead of uv add",
            "wrong venv activated",
        ],
        "success_criteria": [
            "runs uv add litellm",
            "verifies with uv run python -c 'import litellm'",
            "does not use pip",
        ],
    },
    {
        "task_description": "Write a Polars pipeline that reads an 8M-row payment CSV and normalizes merchant names using rapidfuzz",
        "domain_context": "Python 3.13, Polars, rapidfuzz, PEP 723 inline script",
        "pitfalls": [
            "using pandas instead of Polars",
            "forgetting lazy evaluation",
            "not handling null merchant names",
            "regex instead of fuzzy match",
        ],
        "success_criteria": [
            "uses pl.scan_csv for lazy read",
            "applies rapidfuzz via map_elements or struct",
            "handles nulls explicitly",
        ],
    },
    {
        "task_description": "Create a new TMDL measure table for credit card delinquency KPIs",
        "domain_context": "Power BI TMDL semantic model, tabular model definition language",
        "pitfalls": [
            "wrong TMDL file extension",
            "missing table reference in measure",
            "incorrect formatString",
        ],
        "success_criteria": [
            "creates .tmdl file with correct syntax",
            "adds formatString for percentages",
            "references correct fact table",
        ],
    },
    {
        "task_description": "Fix a Pydantic v2 ValidationError caused by using the v1 __fields__ API",
        "domain_context": "Python 3.13, Pydantic v2, existing codebase migrated from v1",
        "pitfalls": [
            "still using __fields__ instead of model_fields",
            "using @validator instead of @field_validator",
            "not using model_config = ConfigDict(...)",
        ],
        "success_criteria": [
            "replaces __fields__ with model_fields",
            "updates validators to v2 syntax",
            "runs mypy or ty to verify",
        ],
    },
    {
        "task_description": "Set up a pytest fixture for a DuckDB in-memory database with the card inventory schema",
        "domain_context": "Python 3.13, pytest, DuckDB, FIFO inventory model, Decimal arithmetic",
        "pitfalls": [
            "using float instead of Decimal for monetary values",
            "not isolating DB per test",
            "forgetting to close connection",
        ],
        "success_criteria": [
            "fixture uses duckdb.connect(':memory:')",
            "creates schema in fixture setup",
            "uses DECIMAL(19,4) not FLOAT",
        ],
    },
    {
        "task_description": "Deploy a TMDL semantic model update to Power BI Premium workspace",
        "domain_context": "pbi-tools CLI, Power BI Premium, XMLA endpoint",
        "pitfalls": [
            "using Power BI Desktop publish instead of pbi-tools",
            "wrong workspace ID",
            "not checking API version compatibility",
        ],
        "success_criteria": [
            "uses pbi-tools deploy command",
            "verifies workspace connection",
            "checks deployment status",
        ],
    },
    {
        "task_description": "Run the full pytest suite and fix a test that fails due to an import cycle",
        "domain_context": "Python 3.13, uv, pytest, src layout project",
        "pitfalls": [
            "running python -m pytest instead of uv run pytest",
            "fixing the wrong import direction",
            "not checking __init__.py",
        ],
        "success_criteria": [
            "uses uv run pytest",
            "identifies circular import",
            "resolves by moving shared code to a common module",
        ],
    },
    {
        "task_description": "Write a GitHub Actions workflow step that runs on the blazar self-hosted runner",
        "domain_context": "GitHub Actions, self-hosted runner named blazar, opencode pipeline",
        "pitfalls": [
            "using ubuntu-latest instead of self-hosted",
            "not specifying runner label blazar",
            "missing ANTHROPIC_API_KEY secret",
        ],
        "success_criteria": [
            "uses runs-on: [self-hosted, blazar]",
            "correctly references secrets",
            "step has correct working-directory",
        ],
    },
    {
        "task_description": "Refactor a synchronous httpx call to async using anyio",
        "domain_context": "Python 3.13, httpx, anyio, existing sync codebase",
        "pitfalls": [
            "using asyncio.run() instead of anyio.run()",
            "forgetting to make caller async",
            "not using async with for client",
        ],
        "success_criteria": [
            "uses async with httpx.AsyncClient()",
            "wraps entry point with anyio.run()",
            "propagates async correctly up call stack",
        ],
    },
    {
        "task_description": "Implement a cyclopts CLI command for the MerchantNorm pipeline with --input and --output flags",
        "domain_context": "Python 3.13, cyclopts CLI library, Polars, YAML rules file",
        "pitfalls": [
            "using argparse or click instead of cyclopts",
            "not using Path type annotations for file args",
            "forgetting to handle stdin/stdout",
        ],
        "success_criteria": [
            "imports from cyclopts",
            "uses typed Path parameters",
            "adds --dry-run flag for safety",
        ],
    },
    {
        "task_description": "Add ruff and ty configuration to pyproject.toml for a new Python package",
        "domain_context": "Python 3.13, ruff, ty (type checker), uv, pyproject.toml",
        "pitfalls": [
            "using mypy config instead of ty",
            "not setting target-version to py313",
            "forgetting ruff format settings",
        ],
        "success_criteria": [
            "[tool.ruff] section with target-version = 'py313'",
            "[tool.ty] section present",
            "ruff check and ruff format configured",
        ],
    },
    {
        "task_description": "Write an opencode TypeScript plugin that blocks shell commands matching a blocklist pattern",
        "domain_context": "opencode AI coding assistant, Bun runtime, TypeScript plugins, AGENTS.md pipeline",
        "pitfalls": [
            "using Node.js APIs instead of Bun",
            "not registering in opencode.json",
            "wrong hook type (should be tool.execute.before)",
        ],
        "success_criteria": [
            "uses Bun.file or Bun.write for I/O",
            "registers hook on tool.execute.before",
            "updates opencode.json plugins array",
        ],
    },
    {
        "task_description": "Query SQL Server 2016 to find cure rates for 30-day delinquent accounts over the past 12 months",
        "domain_context": "SQL Server 2016, T-SQL, card operations analytics, no window functions beyond 2016 support",
        "pitfalls": [
            "using window functions not supported in 2016",
            "incorrect date arithmetic",
            "joining on wrong key (account vs. customer)",
        ],
        "success_criteria": [
            "T-SQL compatible with SQL Server 2016",
            "correct date range filter",
            "groups by delinquency bucket correctly",
        ],
    },
    {
        "task_description": "Set up structlog for JSON structured logging in a PydanticAI agent",
        "domain_context": "Python 3.13, structlog, PydanticAI, async context",
        "pitfalls": [
            "using Python logging.basicConfig instead of structlog.configure",
            "not binding request_id to context",
            "blocking calls in async code",
        ],
        "success_criteria": [
            "configures structlog with JSONRenderer",
            "uses structlog.contextvars.bind_contextvars for request ID",
            "async-safe log calls",
        ],
    },
    {
        "task_description": "Create a LangGraph state machine for the RPIV research pipeline with typed state",
        "domain_context": "Python 3.13, LangGraph, RPIV (Research-Plan-Implement-Validate), Pydantic v2",
        "pitfalls": [
            "using untyped dict state instead of TypedDict",
            "not handling conditional edges",
            "forgetting checkpointer for persistence",
        ],
        "success_criteria": [
            "defines state as TypedDict or Pydantic BaseModel",
            "uses conditional_edge for branching",
            "adds MemorySaver or SqliteSaver",
        ],
    },
    {
        "task_description": "Fix a failing WeasyPrint PDF generation that errors on missing C059 fonts",
        "domain_context": "WeasyPrint, C059 fonts, Debian Trixie, legal documents for FCC (First City Court)",
        "pitfalls": [
            "installing wrong font package",
            "not refreshing font cache",
            "confusing C059 with Century font family",
        ],
        "success_criteria": [
            "installs fonts-urw-base35 package",
            "runs fc-cache -f",
            "verifies with fc-list | grep C059",
        ],
    },
    {
        "task_description": "Write a Pydantic v2 model for a credit card payment event with Decimal amounts and ISO 8601 timestamps",
        "domain_context": "Python 3.13, Pydantic v2, financial data, strict decimal arithmetic",
        "pitfalls": [
            "using float for amounts",
            "not using datetime with tzinfo",
            "using old validator syntax",
        ],
        "success_criteria": [
            "amount field uses Decimal type",
            "timestamp uses datetime with timezone",
            "model_config = ConfigDict(strict=True)",
        ],
    },
    {
        "task_description": "Configure the niri Wayland compositor to launch a floating foot terminal on a keybind",
        "domain_context": "niri compositor, Wayland, foot terminal, Debian Trixie, JetBrainsMono Nerd Font",
        "pitfalls": [
            "using i3/sway config syntax instead of niri KDML",
            "wrong keybind syntax",
            "foot config in wrong location",
        ],
        "success_criteria": [
            "uses niri config KDML syntax",
            "spawn-at-startup or keybind with spawn action",
            "foot config at ~/.config/foot/foot.ini",
        ],
    },
    {
        "task_description": "Implement a retry decorator with exponential backoff for MiniMax M2.7 API calls using tenacity",
        "domain_context": "Python 3.13, tenacity, MiniMax M2.7 via Anthropic-compatible endpoint, httpx",
        "pitfalls": [
            "not handling 429 rate limit separately from 5xx",
            "using time.sleep in async code",
            "not logging retry attempts",
        ],
        "success_criteria": [
            "uses @retry with wait_exponential",
            "filters on specific HTTP status codes",
            "logs attempt number with structlog",
        ],
    },
]

_RPIV_TASKS: list[dict] = [
    {
        "task_description": "Configure opencode.json to add a new TypeScript plugin as the first entry in the plugins array",
        "domain_context": "opencode AI coding assistant, RPIV pipeline, AGENTS.md",
        "pitfalls": [
            "wrong JSON schema",
            "forgetting to restart opencode after config change",
            "relative vs absolute plugin path",
        ],
        "success_criteria": [
            "valid JSON",
            "plugin path resolved correctly",
            "plugin appears before others that depend on it",
        ],
    },
    {
        "task_description": "Write an AGENTS.md that defines Research, Plan, Implement, and Validate sub-agents with clear handoff schemas",
        "domain_context": "Multi-agent RPIV pipeline, Claude Code subagents, opencode",
        "pitfalls": [
            "vague role descriptions",
            "no explicit output format for handoffs",
            "missing error escalation path",
        ],
        "success_criteria": [
            "each agent has input/output schema",
            "handoff format is machine-parseable",
            "validate stage has pass/fail criteria",
        ],
    },
    {
        "task_description": "Debug why the py-critic subagent is not being invoked after the py-implementer finishes",
        "domain_context": "opencode multi-agent, AGENTS.md, subagent invocation, MiniMax M2.7",
        "pitfalls": [
            "wrong subagent file name referenced in AGENTS.md",
            "task.complete not being called to trigger next agent",
            "opencode version mismatch",
        ],
        "success_criteria": [
            "checks AGENTS.md handoff trigger",
            "verifies subagent file exists at correct path",
            "tests with minimal example task",
        ],
    },
]

_GENERAL_CODING_TASKS: list[dict] = [
    {
        "task_description": "Add a type-safe generic repository pattern for a FastAPI + SQLAlchemy project",
        "domain_context": "Python 3.13, FastAPI, SQLAlchemy 2.0, Pydantic v2",
        "pitfalls": [
            "not using Generic[T]",
            "mixing sync and async session",
            "not handling not-found as exception vs None",
        ],
        "success_criteria": [
            "BaseRepository[T] with TypeVar bound to SQLAlchemy model",
            "async session injection",
            "raises NotFoundError or returns None consistently",
        ],
    },
    {
        "task_description": "Set up a pre-commit hook that runs ruff check --fix and ruff format on staged files",
        "domain_context": "Python project, git hooks, ruff, uv",
        "pitfalls": [
            "hook runs on all files not just staged",
            "not making hook executable",
            "ruff config not found",
        ],
        "success_criteria": [
            "pre-commit config targets staged files only",
            "hook is executable",
            "ruff config in pyproject.toml is found",
        ],
    },
]

TASK_LIBRARIES: dict[str, list[dict]] = {
    "banking": _BANKING_TASKS,
    "banking-analytics": _BANKING_TASKS,
    "rpiv": _RPIV_TASKS,
    "agent": _RPIV_TASKS,
    "general": _GENERAL_CODING_TASKS,
    "coding": _GENERAL_CODING_TASKS,
}


def load_task_library(domain: str) -> list[dict]:
    """
    Return the built-in task library for a domain.
    Falls back to 'general' if the domain isn't found.
    """
    key = domain.lower().replace(" ", "-")
    tasks = TASK_LIBRARIES.get(key) or TASK_LIBRARIES.get("general", [])
    print(f"[synthetic] Loaded {len(tasks)} built-in tasks for domain='{domain}'")
    return tasks


# ---------------------------------------------------------------------------
# LLM-generated task library
# ---------------------------------------------------------------------------

_TASK_GEN_SYSTEM = (
    f"{DIRECT_OUTPUT_PREFIX}\n\n"
    "Given a domain description, output a JSON array of software engineering tasks "
    "an AI coding agent might be asked to perform.\n\n"
    "Each task must have exactly these keys:\n"
    "  task_description : 1-2 sentences\n"
    "  domain_context   : key tools, frameworks, versions\n"
    "  pitfalls         : list of 2-4 common mistakes\n"
    "  success_criteria : list of 2-4 concrete success signs\n\n"
    "Output ONLY a valid JSON array. No preamble, no markdown fences."
)


def generate_tasks_for_domain(
    domain: str,
    domain_description: str,
    judge_lm: str = DEFAULT_MODEL,
    n: int = 12,
) -> list[dict]:
    """
    Use an LLM to generate n task descriptions for a domain you describe.
    Returns a list of task dicts in the same schema as the built-in libraries.
    """
    import litellm  # type: ignore

    prompt = (
        f"Domain: {domain}\n"
        f"Description: {domain_description}\n"
        f"Generate exactly {n} diverse tasks. "
        f"Range from simple bug fixes to complex features. "
        f"Focus on tasks that are likely to fail in specific ways."
    )

    print(f"[synthetic] Generating {n} tasks for domain='{domain}' via {judge_lm} ...")
    try:
        resp = litellm.completion(
            model=judge_lm,
            messages=[
                {"role": "system", "content": _TASK_GEN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=EVAL_MAX_TOKENS * 8,  # task gen needs more room than scoring
            **INFERENCE_PARAMS,
        )
        raw = resp.choices[0].message.content or "[]"
        tasks = _parse_llm_json(raw, [])
        print(f"[synthetic] Generated {len(tasks)} tasks")
        return tasks
    except Exception as exc:
        print(f"[synthetic] Task generation failed: {exc}. Falling back to built-in tasks.")
        return load_task_library(domain)


# ---------------------------------------------------------------------------
# Structural scorer (free — no API calls)
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = [
    ("overview", r"##?\s*(overview|about|repository|repo)", 0.10),
    ("commands", r"##?\s*(command|run|build|test|install|usage)", 0.12),
    ("pitfalls", r"##?\s*(pitfall|error|common mistake|gotcha|warning|caution)", 0.12),
    ("conventions", r"##?\s*(convention|pattern|style|standard|rule)", 0.08),
    ("workflow", r"##?\s*(workflow|process|step|procedure|how to)", 0.06),
]

_SPECIFICITY_PATTERNS = [
    r"`[^`]+`",  # inline code
    r"\b(uv|ruff|ty|pytest|rg|fd|eza|bat)\b",  # specific tool names
    r"\b(Pydantic|Polars|LangGraph|DuckDB|WeasyPrint|structlog)\b",
    r"\b(TMDL|DAX|CALCULATE|DATESINPERIOD)\b",
    r"--\w+",  # CLI flags
    r"\$\s*\w+",  # shell commands
    r"\.py\b|\.md\b|\.toml\b",  # file extensions
]

_COMPILED_SPECIFICITY = [re.compile(p) for p in _SPECIFICITY_PATTERNS]


def structural_score(candidate: str) -> tuple[float, dict]:
    """
    Score a skill candidate using fast structural heuristics.
    Returns (score ∈ [0,1], breakdown dict).
    """
    breakdown: dict[str, float] = {}
    total = 0.0

    # 1. Length check (target: 500-3000 chars; 1000-2000 is sweet spot)
    n = len(candidate)
    if 800 <= n <= 2500:
        length_score = 0.15
    elif 500 <= n < 800 or 2500 < n <= 4000:
        length_score = 0.08
    elif n < 200:
        length_score = 0.0
    else:
        length_score = 0.04  # too long
    breakdown["length"] = length_score
    total += length_score

    # 2. Section presence
    for name, pattern, weight in _REQUIRED_SECTIONS:
        if re.search(pattern, candidate, re.IGNORECASE):
            breakdown[f"section_{name}"] = weight
            total += weight
        else:
            breakdown[f"section_{name}"] = 0.0

    # 3. Specificity (inline code, tool names, file paths)
    specificity_hits = sum(
        len(_COMPILED_SPECIFICITY[i].findall(candidate)) for i in range(len(_SPECIFICITY_PATTERNS))
    )
    specificity_score = min(0.20, specificity_hits * 0.015)
    breakdown["specificity"] = specificity_score
    total += specificity_score

    # 4. Not generic (penalize boilerplate phrases)
    generic_phrases = [
        "follow best practices",
        "write clean code",
        "be careful",
        "make sure to",
        "don't forget to",
        "always remember",
    ]
    generic_hits = sum(1 for p in generic_phrases if p.lower() in candidate.lower())
    generic_penalty = min(0.12, generic_hits * 0.03)
    breakdown["generic_penalty"] = -generic_penalty
    total -= generic_penalty

    # 5. Has numbered or bulleted lists (structured = better)
    has_lists = bool(re.search(r"^\s*[-*•]\s|\b\d+\.\s", candidate, re.MULTILINE))
    list_score = 0.08 if has_lists else 0.0
    breakdown["has_lists"] = list_score
    total += list_score

    return max(0.0, min(1.0, total)), breakdown


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    f"{DIRECT_OUTPUT_PREFIX}\n\n"
    "Rate 0.0–1.0 how much this SKILL.md would help an agent complete the task.\n\n"
    "Rubric:\n"
    "  0.0–0.2  irrelevant or misleading\n"
    "  0.2–0.4  tangential, misses key guidance\n"
    "  0.4–0.6  relevant but doesn't address pitfalls\n"
    "  0.6–0.8  addresses most pitfalls\n"
    "  0.8–1.0  directly prevents all pitfalls, success criteria obvious\n\n"
    'Output ONLY: {"score": float, "reasoning": "one sentence", "gaps": ["missing1", ...]}'
)

# M2.7-specific judge system prompt — mitigates tunnel-vision and verbosity
# M2.7 tends to: (1) over-think simple tasks, (2) repeat failed approaches
# This directive encourages exploration of multiple solution strategies
_JUDGE_SYSTEM_M2_7 = (
    f"{DIRECT_OUTPUT_PREFIX}\n\n"
    "Rate 0.0–1.0 how much this SKILL.md would help an agent complete the task.\n\n"
    "IMPORTANT — M2.7 tuning:\n"
    "- If the skill mentions multiple approaches when one fails, award higher scores\n"
    "- Skills that explicitly name error recovery strategies score higher\n"
    "- Concise skills (under 150 words) that cover essentials score higher\n\n"
    "Rubric:\n"
    "  0.0–0.2  irrelevant or misleading\n"
    "  0.2–0.4  tangential, misses key guidance\n"
    "  0.4–0.6  relevant but doesn't address pitfalls\n"
    "  0.6–0.8  addresses most pitfalls + includes recovery strategies\n"
    "  0.8–1.0  directly prevents pitfalls, has error recovery, concise\n\n"
    'Output ONLY: {"score": float, "reasoning": "one sentence", "gaps": ["missing1", ...]}'
)


def judge_score_task_m2_7(
    candidate: str,
    task: dict,
    judge_lm: str = DEFAULT_MODEL,
) -> tuple[float, dict]:
    """
    M2.7-optimized judge — uses _JUDGE_SYSTEM_M2_7 with tunnel-vision mitigation.

    M2.7's known weaknesses:
    - Tunnel-vision: repeats failed approaches instead of trying different strategies
    - Verbosity: 4x more tokens than average
    - "Read everything first": causes timeouts

    This variant:
    - Rewards error recovery strategies in skills
    - Rewards multiple solution approaches
    - Penalizes verbose skills (>150 words unless essential)
    """
    import litellm  # type: ignore

    # Import M2.7-specific token budget
    try:
        from llm_config import M2_7_EVAL_MAX_TOKENS
    except ImportError:
        M2_7_EVAL_MAX_TOKENS = 256

    pitfalls = "\n".join(f"- {p}" for p in task.get("pitfalls", []))
    criteria = "\n".join(f"- {c}" for c in task.get("success_criteria", []))

    user_msg = (
        f"SKILL.md (MUST be under 150 words for full score):\n{candidate[:2000]}\n\n"
        f"Task: {task['task_description']}\n"
        f"Context: {task.get('domain_context', '')}\n"
        f"Pitfalls to prevent:\n{pitfalls}\n"
        f"Success criteria:\n{criteria}\n\n"
        'Output: {"score": <0-1>, "reasoning": "<one sentence>", "gaps": [...]}'
    )

    try:
        resp = litellm.completion(
            model=judge_lm,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_M2_7},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=M2_7_EVAL_MAX_TOKENS,
            **INFERENCE_PARAMS,
        )
        raw = resp.choices[0].message.content or "{}"
        data = _parse_llm_json(raw, {})
        return float(data.get("score", 0.5)), {
            "judge_score": float(data.get("score", 0.5)),
            "reasoning": data.get("reasoning", ""),
            "gaps": data.get("gaps", []),
            "task": task["task_description"][:100],
            "model": "m2_7_optimized",
        }
    except Exception as exc:
        return 0.5, {"error": str(exc), "task": task.get("task_description", "")[:100]}


def judge_score_task(
    candidate: str,
    task: dict,
    judge_lm: str = DEFAULT_MODEL,
) -> tuple[float, dict]:
    """
    Call an LLM to score the candidate against one task.
    Returns (score, side_info dict).

    M2.7 tuning: explicit JSON contract, max_tokens cap, official inference params.
    """
    import litellm  # type: ignore

    pitfalls = "\n".join(f"- {p}" for p in task.get("pitfalls", []))
    criteria = "\n".join(f"- {c}" for c in task.get("success_criteria", []))

    user_msg = (
        f"SKILL.md:\n{candidate[:4000]}\n\n"
        f"Task: {task['task_description']}\n"
        f"Context: {task.get('domain_context', '')}\n"
        f"Pitfalls:\n{pitfalls}\n"
        f"Success criteria:\n{criteria}\n\n"
        'Output: {"score": <0-1>, "reasoning": "<one sentence>", "gaps": [...]}'
    )

    try:
        resp = litellm.completion(
            model=judge_lm,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=EVAL_MAX_TOKENS,
            **INFERENCE_PARAMS,
        )
        raw = resp.choices[0].message.content or "{}"
        data = _parse_llm_json(raw, {})
        return float(data.get("score", 0.5)), {
            "judge_score": float(data.get("score", 0.5)),
            "reasoning": data.get("reasoning", ""),
            "gaps": data.get("gaps", []),
            "task": task["task_description"][:100],
        }
    except Exception as exc:
        return 0.5, {"error": str(exc), "task": task.get("task_description", "")[:100]}


# ---------------------------------------------------------------------------
# Combined synthetic evaluator factory
# ---------------------------------------------------------------------------


def make_synthetic_evaluator(
    task_library: list[dict],
    judge_lm: str = "anthropic/claude-haiku-4-5-20251001",
    judge_weight: float = 0.65,  # LLM judge is the primary semantic signal; heuristic score is compressed in [0.35, 0.85]
    structural_weight: float = 0.35,
    use_judge: bool = True,
) -> Any:
    """
    Returns a GEPA-compatible evaluate(candidate, example) function
    that works with zero session history.

    candidate : str  — the SKILL.md content being evaluated
    example   : dict — one task from task_library (passed via dataset= arg)

    Args:
        task_library:      List of task dicts (from load_task_library or generate_tasks_for_domain).
        judge_lm:          litellm model string for the judge.
        judge_weight:      Weight of judge score in final blend [0,1].
        structural_weight: Weight of structural score in final blend [0,1].
        use_judge:         Set False to use only the structural scorer (free, no API calls).
                           Useful for very fast first-pass optimization.
    """
    if use_judge and abs(judge_weight + structural_weight - 1.0) >= 0.01:
        raise ValueError(
            f"judge_weight ({judge_weight}) + structural_weight ({structural_weight}) must sum to 1.0 when use_judge=True"
        )

    # Pick M2.7-specific judge if the configured model contains "m2.7" (case-insensitive).
    # The M2.7 judge uses a rubric that rewards error-recovery strategies, multiple
    # solution approaches, and penalizes verbose skills over 150 words.
    _judge_fn = judge_score_task_m2_7 if "m2.7" in judge_lm.lower() else judge_score_task

    def evaluate(candidate: str, example: dict) -> tuple[float, dict]:
        import time

        eval_start = time.monotonic()

        struct_score, struct_breakdown = structural_score(candidate)

        if use_judge:
            j_score, j_info = _judge_fn(candidate, example, judge_lm)
            final = judge_weight * j_score + structural_weight * struct_score
        else:
            j_score, j_info = struct_score, {}
            final = struct_score

        eval_duration = time.monotonic() - eval_start

        # --- GEPA ASI channel: log diagnostic fields before building side_info ---
        try:
            # Episode outcome
            oa.log(f"Outcome: {example.get('outcome', 'unknown')}")

            # Duration
            oa.log(f"Duration: {eval_duration:.1f}s")

            # First 2 error_messages (truncated to 200 chars)
            errors = example.get("error_messages", [])
            for err in errors[:2]:
                oa.log(f"Error: {err[:200]}")

            # First 10 tool names joined with " → "
            tool_calls = example.get("tool_calls", [])
            if tool_calls:
                tool_names = [
                    tc.get("tool", str(tc)) if isinstance(tc, dict) else str(tc)
                    for tc in tool_calls[:10]
                ]
                oa.log(f"Tool calls: {' → '.join(tool_names)}")

            # Context compaction signal
            if example.get("compaction_summary"):
                oa.log("Context compaction hit — skill may be too verbose")

            # Judge score / reasoning (high-level verdict only)
            if use_judge and j_info:
                jsc = j_info.get("judge_score")
                if jsc is not None:
                    oa.log(f"Judge score: {jsc:.2f}")
        except Exception:
            # oa.log() is a no-op outside wrapper context — tolerate here
            pass

        side_info = {
            "score": final,
            "structural_score": struct_score,
            "structural_breakdown": struct_breakdown,
            "task": example.get("task_description", "")[:150],
            "domain_context": example.get("domain_context", ""),
            "pitfalls": example.get("pitfalls", []),
            "success_criteria": example.get("success_criteria", []),
            "candidate_length": len(candidate),
            "candidate_preview": candidate[:300],
            **j_info,
        }

        # Capture GEPA log context for injection into feedback (reflection LM reads side_info["feedback"])
        _captured_log = ""
        with contextlib.suppress(Exception):
            _captured_log = oa.get_log_context().drain()

        # Build feedback for GEPA reflection LM
        if use_judge and j_info:
            j_reasoning = j_info.get("reasoning", "")
            j_gaps = j_info.get("gaps", [])
            parts = [f"Score {final:.2f}. Structural: {struct_score:.2f}."]
            if j_reasoning:
                parts.append(f"Judge: {j_reasoning[:200]}")
            if j_gaps:
                gaps_text = "; ".join(j_gaps[:3])[:200]
                parts.append(f"Gaps identified: {gaps_text}")
            if _captured_log:
                parts.append(f"Diagnostic log: {_captured_log}")
            side_info["feedback"] = " ".join(parts)
        else:
            # No judge — feedback from structural score alone
            if struct_breakdown:
                missing = [
                    name
                    for name, score in struct_breakdown.items()
                    if score == 0.0 and name.startswith("section_")
                ]
                if missing:
                    side_info["feedback"] = (
                        f"Structural score {struct_score:.2f}. Missing sections: {', '.join(missing[:3])}."
                    )
                else:
                    side_info["feedback"] = (
                        f"Structural score {struct_score:.2f}. No specific gaps detected."
                    )
            else:
                side_info["feedback"] = f"Score {final:.2f}. No specific feedback available."
            if _captured_log:
                side_info["feedback"] += f" Diagnostic log: {_captured_log}"

        return final, side_info

    return evaluate


# ---------------------------------------------------------------------------
# DSPy version: BootstrapFewShot → GEPA pipeline
# ---------------------------------------------------------------------------


def make_dspy_synthetic_pipeline(
    task_library: list[dict],
    seed_candidate: str,
    task_lm: str = "anthropic/claude-haiku-4-5-20251001",
    reflection_lm: str = "anthropic/claude-haiku-4-5-20251001",
    max_bootstrap_evals: int = 30,
    max_gepa_evals: int = 80,
    output_dir: str = "outputs/synthetic",
) -> str:
    """
    Two-stage DSPy optimization pipeline for zero-session skills:

      Stage 1: BootstrapFewShot on task_library
               → finds few-shot demos showing good skill content per task type
               → fast, cheap, sets a solid baseline

      Stage 2: MIPROv2 on top of the bootstrapped program
               → prompt proposal optimization using the metric as a scalar score
               → MIPROv2 is used (not dspy.GEPA) because the metric returns a flat float;
                 dspy.GEPA would require a dspy.Prediction(score, feedback) to drive
                 its reflection loop, which would need a richer signal than we have here.

    Returns the best skill string found.
    """

    import dspy
    from dspy.teleprompt import MIPROv2

    task_lm_obj = dspy.LM(model=task_lm, temperature=0.7, max_tokens=4096)
    reflect_lm_obj = dspy.LM(model=reflection_lm, temperature=1.0, max_tokens=16000)
    dspy.configure(lm=task_lm_obj)

    # ------------------------------------------------------------------ #
    # DSPy Signature
    # ------------------------------------------------------------------ #
    class SkillQuality(dspy.Signature):
        """
        Evaluate and improve SKILL.md content for a Claude Code coding agent.
        Given a task description and known pitfalls, rate whether the skill
        provides sufficient guidance to avoid those pitfalls.
        """

        skill_content: str = dspy.InputField(desc="Current SKILL.md content")
        task_description: str = dspy.InputField(desc="Software engineering task")
        domain_context: str = dspy.InputField(desc="Relevant tools and frameworks")
        pitfalls: str = dspy.InputField(desc="Known failure modes for this task")
        success_criteria: str = dspy.InputField(desc="Signs the task was done correctly")
        quality_assessment: str = dspy.OutputField(
            desc="Assessment of whether the skill covers this task well, and what's missing"
        )
        improved_guidance: str = dspy.OutputField(
            desc="Specific guidance to add to the skill for this task type"
        )

    class SkillEvaluator(dspy.Module):
        def __init__(self):
            self.assess = dspy.Predict(SkillQuality)

        def forward(
            self,
            skill_content: str,
            task_description: str,
            task_domain_context: str,
            task_pitfalls_str: str,
            task_success_criteria_str: str,
        ) -> dspy.Prediction:
            return self.assess(
                skill_content=skill_content,
                task_description=task_description,
                domain_context=task_domain_context,
                pitfalls=task_pitfalls_str,
                success_criteria=task_success_criteria_str,
            )

    # ------------------------------------------------------------------ #
    # Convert task library to DSPy examples
    # ------------------------------------------------------------------ #
    def task_to_example(task: dict, seed: str) -> dspy.Example:
        task_description = task.get("task_description", "")
        task_domain_context = task.get("domain_context", "")
        task_pitfalls = task.get("pitfalls", [])
        task_success_criteria = task.get("success_criteria", [])
        return dspy.Example(
            skill_content=seed,
            task_description=task_description,
            task_domain_context=task_domain_context,
            # Store ONLY strings in the Example to avoid unhashable list/dict
            task_pitfalls_str="\n".join(f"- {p}" for p in task_pitfalls),
            task_success_criteria_str="\n".join(f"- {c}" for c in task_success_criteria),
            # Gold: ideal assessment — the skill should address all pitfalls
            quality_assessment=(
                f"The skill must address: {'; '.join(task_pitfalls)}. "
                f"Success looks like: {'; '.join(task_success_criteria)}."
            ),
            improved_guidance=(
                f"Add guidance for: {task_description}. "
                f"Key pitfalls to warn about: {', '.join(task_pitfalls[:2])}."
            ),
        ).with_inputs(
            "skill_content",
            "task_description",
            "task_domain_context",
            "task_pitfalls_str",
            "task_success_criteria_str",
        )

    import random as _random

    rng = _random.Random(42)
    shuffled = task_library[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    t = max(5, int(n * 0.75))
    train_tasks = shuffled[:t]
    val_tasks = shuffled[t:] or shuffled[:3]

    dspy_train = [task_to_example(t, seed_candidate) for t in train_tasks]
    dspy_val = [task_to_example(t, seed_candidate) for t in val_tasks]

    # ------------------------------------------------------------------ #
    # Metric: judge the quality_assessment for completeness
    # ------------------------------------------------------------------ #
    def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        assessment = getattr(pred, "quality_assessment", "")
        guidance = getattr(pred, "improved_guidance", "")
        # Parse pitfalls list from string format "- pitfall1\n- pitfall2"
        pitfalls_str = getattr(gold, "task_pitfalls_str", "") or ""
        pitfalls = [line[2:] for line in pitfalls_str.split("\n") if line.startswith("- ")]

        # Check if pitfalls are acknowledged in the assessment
        pitfall_coverage = sum(
            1 for p in pitfalls if any(word.lower() in assessment.lower() for word in p.split()[:3])
        ) / max(1, len(pitfalls))

        # Check if improved_guidance is specific (has code/commands)
        has_code = bool(re.search(r"`[^`]+`|--\w+|\$ \w+", guidance))
        specificity_bonus = 0.15 if has_code else 0.0

        return min(1.0, pitfall_coverage * 0.85 + specificity_bonus)

    # ------------------------------------------------------------------ #
    # Stage 1: BootstrapFewShot
    # ------------------------------------------------------------------ #
    print("\n[dspy-synthetic] Stage 1: BootstrapFewShot")
    print(f"  train={len(dspy_train)} val={len(dspy_val)} task_lm={task_lm}\n")

    program = SkillEvaluator()
    bootstrap = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        max_errors=10,
    )
    try:
        bootstrapped = bootstrap.compile(program, trainset=dspy_train)
        print("[dspy-synthetic] BootstrapFewShot complete")
    except Exception as exc:
        print(f"[dspy-synthetic] BootstrapFewShot failed ({exc}), continuing with base program")
        bootstrapped = program

    # ------------------------------------------------------------------ #
    # Stage 2: dspy.MIPROv2 on top
    # ------------------------------------------------------------------ #
    print("\n[dspy-synthetic] Stage 2: MIPROv2 refinement")
    print(f"  max_metric_calls={max_gepa_evals} reflection_lm={reflection_lm}\n")

    mipro_optimizer = MIPROv2(
        metric=metric,
        prompt_model=reflect_lm_obj,
        num_threads=2,
        auto="medium",
    )
    try:
        optimized = mipro_optimizer.compile(
            bootstrapped,
            trainset=dspy_train,
            valset=dspy_val,
        )
        print("[dspy-synthetic] MIPROv2 complete")
    except Exception as exc:
        print(f"[dspy-synthetic] MIPROv2 failed ({exc}), using bootstrapped program")
        optimized = bootstrapped

    # ------------------------------------------------------------------ #
    # Extract MIPROv2-optimized instructions and few-shot demos.
    # Access path verified for DSPy 2.6.27 with dspy.Predict modules.
    # Graceful fallback if API path changes between DSPy versions.
    # ------------------------------------------------------------------ #
    optimized_instructions: str | None = None
    optimized_demos: list = []
    try:
        optimized_instructions = optimized.assess.signature.instructions
    except AttributeError:
        print(
            "[dspy-synthetic] WARN: could not extract optimized instructions — DSPy API may have changed"
        )
    with contextlib.suppress(Exception):
        optimized_demos = list(getattr(optimized.assess, "demos", []) or [])

    # ------------------------------------------------------------------ #
    # Extract improved skill content by running the optimized program
    # on each task and collecting the improved_guidance fields
    # ------------------------------------------------------------------ #
    print("\n[dspy-synthetic] Synthesizing optimized SKILL.md from guidance outputs ...")
    guidance_blocks: list[str] = []
    for task in val_tasks[:6]:  # sample a few representative tasks
        try:
            pred = optimized(skill_content=seed_candidate, task=task)
            g = getattr(pred, "improved_guidance", "").strip()
            if g and len(g) > 30:
                guidance_blocks.append(
                    f"<!-- Guidance for: {task['task_description'][:60]} -->\n{g}"
                )
        except Exception:
            pass

    # Combine seed with extracted guidance
    combined = seed_candidate.rstrip()
    sections: list[str] = []

    # Section 1: MIPROv2-optimized instructions (the most valuable output)
    if optimized_instructions and optimized_instructions.strip():
        sections.append(
            "## Optimized Instructions (MIPROv2-refined)\n\n"
            f"{optimized_instructions.strip()}\n\n"
            "<!-- This block was discovered by MIPROv2 instruction optimization; it represents\n"
            "the prompt template the reflection LM found to produce the highest-scoring outputs. -->"
        )

    # Section 2: Few-shot demos (if any were selected)
    if optimized_demos:
        demo_blocks: list[str] = []
        for i, demo in enumerate(optimized_demos[:3], start=1):  # limit to first 3
            demo_input = getattr(demo, "task_description", "<demo>")
            demo_output = getattr(demo, "improved_guidance", "<output>")
            demo_blocks.append(
                f"### Example {i}\n\n"
                f"**Task:** {str(demo_input)[:120]}\n\n"
                f"**Guidance:**\n{str(demo_output)[:300]}"
            )
        if demo_blocks:
            sections.append(
                "## Few-Shot Examples (from MIPROv2 bootstrap)\n\n"
                + "\n\n".join(demo_blocks)
                + "\n\n<!-- These examples were selected by MIPROv2 as high-scoring demonstrations. -->"
            )

    # Section 3: Auto-generated guidance (existing)
    if guidance_blocks:
        sections.append(
            "## Auto-Generated Guidance (GEPA-refined)\n\n" + "\n\n".join(guidance_blocks)
        )

    if sections:
        combined += "\n\n" + "\n\n".join(sections)

    # Save
    from pathlib import Path as _Path

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_candidate_dspy_synthetic.md").write_text(combined, encoding="utf-8")
    try:
        optimized.save(str(out / "dspy_synthetic_program.json"))
    except Exception:
        pass
    print(f"[dspy-synthetic] Saved to {out}/best_candidate_dspy_synthetic.md")
    return combined


# ---------------------------------------------------------------------------
# CLI (standalone test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test synthetic evaluator against a candidate skill")
    ap.add_argument("--domain", default="banking")
    ap.add_argument("--candidate", default=None, help="Path to SKILL.md to evaluate")
    ap.add_argument("--no-judge", action="store_true", help="Structural scoring only (free)")
    ap.add_argument("--judge-lm", default="anthropic/claude-haiku-4-5-20251001")
    args = ap.parse_args()

    tasks = load_task_library(args.domain)
    candidate = ""
    if args.candidate:
        from pathlib import Path

        candidate = Path(args.candidate).read_text(encoding="utf-8")

    evaluate = make_synthetic_evaluator(
        tasks,
        judge_lm=args.judge_lm,
        use_judge=not args.no_judge,
    )

    print(f"\nEvaluating against {len(tasks)} tasks (use_judge={not args.no_judge}):\n")
    total = 0.0
    for task in tasks[:5]:
        score, info = evaluate(candidate, task)
        total += score
        print(f"  [{score:.2f}] {task['task_description'][:70]}")
        if info.get("reasoning"):
            print(f"         → {info['reasoning'][:100]}")
        if info.get("gaps"):
            print(f"         gaps: {info['gaps'][:2]}")
    print(f"\n  Average (5 tasks): {total / 5:.3f}")
