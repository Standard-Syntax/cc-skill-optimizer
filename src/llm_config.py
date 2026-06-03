"""
llm_config.py
=============
Central LLM configuration — MiniMax Anthropic-compatible endpoint with extended thinking.

Research sources
----------------
- https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
- https://platform.claude.com/docs/en/build-with-claude/context-editing

Thinking blocks: do we need to pass them back?
-----------------------------------------------
SHORT ANSWER: Yes, always pass them back unchanged. Here's the nuance:

  REQUIRED (will raise an error if omitted):
    During tool use — you MUST pass the complete, unmodified thinking blocks
    (both `thinking` and `redacted_thinking` types) back with the tool result.
    The `signature` field carries encrypted reasoning continuity data that
    Claude needs to continue from where it left off.

  OPTIONAL but strongly recommended:
    In regular multi-turn conversations without tool use, you can omit thinking
    blocks from prior assistant turns. The API auto-filters what it needs.
    However, Anthropic explicitly says: "always pass back all thinking blocks
    to the API for any multi-turn conversation" for cache efficiency and
    reasoning continuity. Sonnet 4.6+ and Opus 4.5+ keep all prior thinking
    blocks by default; earlier models keep only the last turn's.

  IMPORTANT: also pass back `redacted_thinking` blocks (not just `thinking`).
    Filtering on `block.type == "thinking"` alone silently drops
    `redacted_thinking` blocks and breaks the multi-turn protocol.

Thinking × inference parameters: incompatibilities
--------------------------------------------------
  BREAKS THINKING (do not pass with thinking enabled):
    - temperature  (any value other than default)
    - top_k        (incompatible entirely)
    - pre-filled responses

  WORKS with thinking enabled:
    - top_p  (must be between 0.95 and 1.0; values below 0.95 are rejected)
    - max_tokens
    - streaming

  GEPA internal calls:
    GEPA's own litellm.completion() calls (reflection/mutation proposals) are
    single-turn — no tool use loops, no multi-turn history. Thinking blocks do
    not accumulate there. However, GEPA does pass the reflection_lm_kwargs we
    provide, so we must NOT include temperature or top_k in those kwargs when
    using an Anthropic thinking-enabled model.

Model recommendations (as of May 2026)
----------------------------------------
  minimax/minimax-m2.7-highspeed  → fast student/eval/judge calls
                        adaptive thinking with low effort recommended.
                        Strengths: 97% skill adherence, low hallucination.
                        Use for: skill generation, eval, judge calls.
  minimax/minimax-m3  → deep reflection and GEPA mutation proposals.
                        adaptive thinking recommended (medium effort).
                        Use for: reflection, high-quality synthesis.

Endpoint routing
----------------
  GEPA's internal litellm.completion() uses the model string to route.
  Using `minimax/` prefix routes directly through MiniMax's provider.
  ANTHROPIC_BASE_URL must include the /v1 suffix
  (https://api.minimax.io/anthropic/v1) so litellm routes correctly.
  MiniMax's endpoint accepts standard Anthropic request formats with these
  model strings: minimax/minimax-m2.7-highspeed, minimax/minimax-m3.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Anthropic endpoint + models
# ---------------------------------------------------------------------------
# MiniMax Anthropic-compatible endpoint
ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic/v1"

# Sonnet 4.6: adaptive thinking recommended; best balance of speed and quality
# Haiku 4.5: fastest and cheapest; use for evaluation/judge calls
DEFAULT_MODEL = "minimax/minimax-m2.7-highspeed"  # student / eval / judge calls
REFLECTION_MODEL = "minimax/minimax-m3"  # deep reflection / GEPA mutation proposals

# MiniMax M2.7 — for skill generation and coding optimization
# Strengths: 97% skill adherence, 34% hallucination (low), SWE-Pro 56.22%, cost $0.30/M
# Weaknesses: verbose (4x avg), slow (~50 tps), tunnel-vision debugging
# Use for: skill generation, skill adherence tasks, low-hallucination content
# Use Haiku/Sonnet for: eval calls (fast), deep reflection (quality)
MINIMAX_M2_7 = "minimax/minimax-m2.7"
MINIMAX_M2_7_HIGHSPEED = "minimax/minimax-m2.7-highspeed"

# ---------------------------------------------------------------------------
# Inference parameters
#
# CRITICAL: temperature and top_k are INCOMPATIBLE with extended thinking.
# Do not include them in any call that has thinking enabled.
# top_p is allowed at 0.95–1.0.
# ---------------------------------------------------------------------------
INFERENCE_PARAMS: dict = {
    # No temperature — incompatible with thinking
    # No top_k — incompatible with thinking
    # top_p allowed; leave at default (1.0) unless explicitly overriding
}

# Extra body — empty for Anthropic; kept for interface compatibility
EXTRA_BODY: dict = {}

# ---------------------------------------------------------------------------
# Thinking configuration
# ---------------------------------------------------------------------------

# Use adaptive thinking for Sonnet 4.6 (recommended over manual budget_tokens)
THINKING_CONFIG_REFLECTION = {
    "type": "adaptive",
    # effort controls depth: "low" | "medium" | "high"
    # "medium" is appropriate for reflection proposals — complex enough to
    # benefit from thinking, but doesn't need maximum depth.
    "effort": "medium",
}

# For Haiku (eval/judge calls) — disable thinking for speed and cost.
# Haiku is already fast; thinking would be overkill for a JSON score response.
THINKING_CONFIG_EVAL = {
    "type": "disabled",
}

# For MiniMax M2.7 — adaptive thinking with low effort to mitigate verbosity
# M2.7 tends to over-think simple tasks and generate 4x more tokens than average
# Use "low" effort to keep outputs concise
THINKING_CONFIG_M2_7 = {
    "type": "adaptive",
    "effort": "low",  # Mitigates M2.7's 4x verbosity tendency
}

# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------

# Reflection: Sonnet 4.6 with adaptive thinking.
# max_tokens must exceed budget_tokens when using manual thinking.
# With adaptive thinking, set max_tokens generously — the model decides depth.
# A revised SKILL.md + reasoning rarely exceeds 8192 output tokens.
REFLECTION_MAX_TOKENS = 8192

# Eval/judge: Haiku with thinking disabled.
# JSON score response: {"score": 0.7, "reasoning": "one sentence"} ≈ 50 tokens.
# 512 is generous.
EVAL_MAX_TOKENS = 512

# MiniMax M2.7 token budgets — lower than Sonnet to prevent timeouts
# M2.7 "reads everything first" behavior can cause timeouts; strict budgets help
# Reflection is limited to 4096 since M2.7 is verbose even with thinking
M2_7_REFLECTION_MAX_TOKENS = 4096
M2_7_EVAL_MAX_TOKENS = 256  # Very low for speed in judge calls

# ---------------------------------------------------------------------------
# System prompt prefix
# (Anthropic models respond well to explicit output contracts)
# ---------------------------------------------------------------------------
DIRECT_OUTPUT_PREFIX = (
    "Respond directly and concisely. "
    "Output only what is explicitly requested. "
    "Do not restate the task or add unsolicited commentary."
)

# M2.7-specific output directive — mitigates 4x verbosity
# Use when calling M2.7 for skill generation or content creation
M2_7_CONCISE_PREFIX = (
    "Be extremely concise. Use short sentences and bullet points. "
    "Do not restate the task or add preamble. Output only what is requested. "
    "Target 50% fewer tokens than typical LLM output. "
    "If the task can be done in 3 sentences, do not use 10."
)


def configure() -> None:
    """
    Set environment variables for the MiniMax Anthropic-compatible endpoint.
    Validates that ANTHROPIC_API_KEY is present (your MiniMax API key).
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Export your MiniMax API key: export ANTHROPIC_API_KEY=..."
        )

    # Route litellm through MiniMax's Anthropic-compatible endpoint.
    # litellm reads ANTHROPIC_BASE_URL when using the anthropic/ model prefix.
    os.environ["ANTHROPIC_BASE_URL"] = ANTHROPIC_BASE_URL

    # Unset any conflicting overrides from previous sessions.
    for var in ("OPENAI_API_BASE", "ANTHROPIC_API_BASE", "LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX"):
        os.environ.pop(var, None)

    os.environ["ANTHROPIC_API_KEY"] = key


configure()
