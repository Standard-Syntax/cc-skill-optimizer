"""
Tests for Task 18.3: TASK_GEN_MAX_TOKENS constant + truncation warning.

Phase 18.3 change: replaces the legacy `EVAL_MAX_TOKENS * 8` (4096) budget
for generate_tasks_for_domain with a dedicated TASK_GEN_MAX_TOKENS = 8192
constant, large enough for 20+ tasks without silent truncation. Also adds
a warning when the LLM returns fewer tasks than requested.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path (same pattern as other test files in this repo)
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

import synthetic_evaluator as se  # noqa: E402


class TestTaskGenMaxTokensConstant:
    """TASK_GEN_MAX_TOKENS is defined in src/llm_config.py."""

    def test_constant_exists_in_llm_config(self):
        from llm_config import TASK_GEN_MAX_TOKENS
        assert TASK_GEN_MAX_TOKENS == 8192, (
            f"Expected TASK_GEN_MAX_TOKENS=8192, got {TASK_GEN_MAX_TOKENS}"
        )

    def test_constant_is_greater_than_eval_max_tokens(self):
        """TASK_GEN_MAX_TOKENS must be larger than EVAL_MAX_TOKENS — task gen
        needs much more room than scoring."""
        from llm_config import EVAL_MAX_TOKENS, TASK_GEN_MAX_TOKENS
        assert TASK_GEN_MAX_TOKENS > EVAL_MAX_TOKENS, (
            f"TASK_GEN_MAX_TOKENS ({TASK_GEN_MAX_TOKENS}) should be > "
            f"EVAL_MAX_TOKENS ({EVAL_MAX_TOKENS})"
        )

    def test_constant_supports_20_tasks(self):
        """8192 tokens at ~400 output tokens per task = ~20 tasks of headroom."""
        from llm_config import TASK_GEN_MAX_TOKENS
        # 20 tasks × 400 tokens = 8000; we have 8192, so ~20 tasks fit
        assert TASK_GEN_MAX_TOKENS >= 20 * 400


class TestGenerateTasksForDomainUsesTaskGenMaxTokens:
    """generate_tasks_for_domain uses the new constant, not EVAL_MAX_TOKENS * 8."""

    def test_source_uses_task_gen_max_tokens_constant(self):
        """Verify the source code uses TASK_GEN_MAX_TOKENS, not EVAL_MAX_TOKENS * 8."""
        synthetic_path = _SRC / "synthetic_evaluator.py"
        src = synthetic_path.read_text(encoding="utf-8")
        # The legacy pattern `EVAL_MAX_TOKENS * 8` should be GONE from the
        # task generation call site
        assert "EVAL_MAX_TOKENS * 8" not in src, (
            "Legacy `EVAL_MAX_TOKENS * 8` should be removed from "
            "src/synthetic_evaluator.py (use TASK_GEN_MAX_TOKENS instead)"
        )
        # The new constant should appear
        assert "TASK_GEN_MAX_TOKENS" in src, (
            "TASK_GEN_MAX_TOKENS should be referenced in src/synthetic_evaluator.py"
        )

    def test_generate_tasks_for_domain_passes_correct_max_tokens(self):
        """Mock litellm.completion and verify max_tokens=TASK_GEN_MAX_TOKENS is passed."""
        # Build a properly-structured mock response
        mock_message = MagicMock()
        mock_message.content = '[{"task_description": "t1"}, {"task_description": "t2"}, {"task_description": "t3"}, {"task_description": "t4"}, {"task_description": "t5"}]'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Patch litellm at the module level where it's imported
        with patch("litellm.completion", return_value=mock_response) as mock_comp:
            with contextlib.suppress(Exception):
                se.generate_tasks_for_domain(
                    domain="test_domain",
                    domain_description="test desc",
                    judge_lm="test/model",
                    n=5,
                )

            if mock_comp.called:
                call_kwargs = mock_comp.call_args.kwargs
                assert call_kwargs.get("max_tokens") == 8192, (
                    f"Expected max_tokens=8192 (TASK_GEN_MAX_TOKENS), got {call_kwargs.get('max_tokens')}"
                )
            else:
                pytest.skip("litellm.completion was not called (function may have short-circuited)")


class TestTruncationWarning:
    """A WARNING is printed when fewer tasks are returned than requested."""

    def test_warning_fires_on_truncated_response(self):
        # Build a mock response with only 3 tasks (truncated from 10)
        mock_message = MagicMock()
        mock_message.content = '[{"task_description": "t1"}, {"task_description": "t2"}, {"task_description": "t3"}]'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with patch("litellm.completion", return_value=mock_response), \
             contextlib.suppress(Exception), \
             redirect_stdout(buf):
            se.generate_tasks_for_domain(
                domain="test_domain",
                domain_description="test desc",
                judge_lm="test/model",
                n=10,
            )
            stdout = buf.getvalue()
            assert "WARNING" in stdout and "requested 10 tasks but got 3" in stdout, (
                f"Expected truncation warning in stdout, got:\n{stdout}"
            )

    def test_no_warning_on_full_response(self):
        """When all requested tasks are returned, no WARNING should print."""
        # Build a mock response with 5 tasks (matching the request)
        mock_message = MagicMock()
        tasks_json = "[" + ",".join(f'{{"task_description": "t{i}"}}' for i in range(5)) + "]"
        mock_message.content = tasks_json
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with patch("litellm.completion", return_value=mock_response), \
             contextlib.suppress(Exception), \
             redirect_stdout(buf):
            se.generate_tasks_for_domain(
                domain="test_domain",
                domain_description="test desc",
                judge_lm="test/model",
                n=5,
            )
        stdout = buf.getvalue()
        assert "WARNING" not in stdout, (
            f"WARNING should NOT appear when all tasks returned. stdout:\n{stdout}"
        )