"""
Tests for LLM judge skill truncation fix in Task 4.4.
Verifies the truncation changed from [:3000] to [:8000].

Tests llm_judge_score() function in src/evaluator.py.
"""

import json
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

# Sample episode dict for testing
SAMPLE_EPISODE = {
    "task_prompt": "Test task prompt for evaluation",
    "outcome": "success",
    "duration_s": 45.0,
    "tool_calls": [
        {"tool": "read", "file": "foo.py"},
        {"tool": "write", "file": "bar.py"},
    ],
    "error_messages": [],
    "bash_commands": ["echo hello"],
    "files_read": ["foo.py"],
    "files_written": ["bar.py"],
    "compaction_summary": None,
    "token_stats": {
        "input": 1000,
        "output": 500,
        "cache_create": 200,
        "cache_read": 300,
    },
    "assistant_text": ["Final assistant response"],
}


def make_skill(length: int) -> str:
    """Generate a skill string of exactly the given length."""
    # Use repeating pattern of meaningful characters
    pattern = "x skill content block "
    # Calculate how many repetitions we need
    repeats = (length // len(pattern)) + 1
    result = pattern * repeats
    return result[:length]


mock_litellm_instance: MagicMock | None = None


@pytest.fixture(autouse=True)
def setup_litellm_mock() -> Iterator[MagicMock]:
    """Setup litellm mock once for all tests in session."""
    global mock_litellm_instance
    mock_litellm_instance = MagicMock()
    sys.modules["litellm"] = mock_litellm_instance
    yield mock_litellm_instance
    # Cleanup
    if "litellm" in sys.modules:
        del sys.modules["litellm"]


class TestJudgeSkillTruncation:
    """Test suite for llm_judge_score truncation limits."""

    def test_candidate_skill_full_text_passed_5000(self):
        """Test: candidate_skill < 8000 chars — full text passed."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        skill_5000 = make_skill(5000)

        # Mock litellm response
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({"score": 0.8, "reasoning": "Good"})
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        # Call the function
        score, reasoning = llm_judge_score(skill_5000, SAMPLE_EPISODE, "test/model")

        # Verify litellm was called
        assert mock_litellm_instance.completion.called

        # Get the messages that were passed to litellm
        call_args = mock_litellm_instance.completion.call_args
        messages = call_args.kwargs["messages"]

        # Find the user message
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        skill_section = user_msg.split("SKILL.md:\n")[1].split("\n\nEpisode:")[0]

        # Verify the skill section contains the full 5000 chars
        assert len(skill_section) == 5000, f"Expected 5000 chars, got {len(skill_section)}"
        print(f"PASS: Full 5000 chars passed through ({len(skill_section)} chars)")

    def test_candidate_skill_exactly_8000_chars(self):
        """Test: candidate_skill exactly 8000 chars — full text passed."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        skill_8000 = make_skill(8000)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({"score": 0.9, "reasoning": "Great"})
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        score, reasoning = llm_judge_score(skill_8000, SAMPLE_EPISODE, "test/model")

        call_args = mock_litellm_instance.completion.call_args
        messages = call_args.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        skill_section = user_msg.split("SKILL.md:\n")[1].split("\n\nEpisode:")[0]

        assert len(skill_section) == 8000, f"Expected 8000 chars, got {len(skill_section)}"
        print(f"PASS: Exactly 8000 chars passed ({len(skill_section)})")

    def test_candidate_skill_truncated_from_12000_to_8000(self):
        """Test: candidate_skill > 8000 chars — truncated to 8000."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        skill_12000 = make_skill(12000)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({"score": 0.7, "reasoning": "Okay"})
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        score, reasoning = llm_judge_score(skill_12000, SAMPLE_EPISODE, "test/model")

        call_args = mock_litellm_instance.completion.call_args
        messages = call_args.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        skill_section = user_msg.split("SKILL.md:\n")[1].split("\n\nEpisode:")[0]

        assert len(skill_section) == 8000, f"Expected 8000 chars, got {len(skill_section)}"
        assert skill_12000.startswith(skill_section), "Truncated text should be prefix of original"
        print(f"PASS: Truncated from 12000 to {len(skill_section)} chars")

    def test_candidate_skill_very_long_50000_chars(self):
        """Test: candidate_skill very long (50000 chars) — truncated to 8000."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        skill_50000 = make_skill(50000)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({"score": 0.6, "reasoning": "Fair"})
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        score, reasoning = llm_judge_score(skill_50000, SAMPLE_EPISODE, "test/model")

        call_args = mock_litellm_instance.completion.call_args
        messages = call_args.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        skill_section = user_msg.split("SKILL.md:\n")[1].split("\n\nEpisode:")[0]

        assert len(skill_section) == 8000, f"Expected 8000 chars, got {len(skill_section)}"
        print(f"PASS: Truncated from 50000 to {len(skill_section)} chars")

    def test_episode_context_truncated_at_2000(self):
        """Test: episode context (asi) truncation preserved at 2000."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        # episode_to_asi is defined in parse_session, import from there
        from src.parse_session import episode_to_asi

        # Verify the expected ASI from episode
        episode_to_asi(SAMPLE_EPISODE)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({"score": 0.8, "reasoning": "OK"})
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        score, reasoning = llm_judge_score("short skill", SAMPLE_EPISODE, "test/model")

        call_args = mock_litellm_instance.completion.call_args
        messages = call_args.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")

        # Extract Episode section
        episode_part = user_msg.split("Episode:\n")[1].split("\n\nOutput:")[0]

        # Verify Episode is truncated to 2000
        assert len(episode_part) <= 2000, f"Episode should be ≤2000 chars, got {len(episode_part)}"
        print(f"PASS: Episode truncated to {len(episode_part)} chars (≤2000)")

    def test_score_output_format(self):
        """Test: score output format — parsed correctly."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        expected_score = 0.75
        expected_reasoning = "Reasonably helpful skill"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(
            {
                "score": expected_score,
                "reasoning": expected_reasoning,
            }
        )
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        score, reasoning = llm_judge_score("test skill", SAMPLE_EPISODE, "test/model")

        assert score == expected_score, f"Expected {expected_score}, got {score}"
        assert reasoning == expected_reasoning, (
            f"Expected '{expected_reasoning}', got '{reasoning}'"
        )
        print(f"PASS: Score {score}, reasoning '{reasoning}'")

    def test_invalid_json_from_judge_handled(self):
        """Test: invalid JSON from judge handled gracefully."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "INVALID JSON {{{"
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        score, reasoning = llm_judge_score("test skill", SAMPLE_EPISODE, "test/model")

        assert score == 0.5, f"Expected default score 0.5, got {score}"
        assert "judge_error" in reasoning.lower() or reasoning == "", (
            f"Expected error in reasoning, got '{reasoning}'"
        )
        print(f"PASS: Handled malformed JSON gracefully: score={score}, reasoning='{reasoning}'")

    def test_judge_lm_parameter_passed_through(self):
        """Test: judge_lm parameter passed to litellm as model."""
        import importlib

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        expected_model = "anthropic/claude-sonnet-4-6"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({"score": 0.9, "reasoning": "Fine"})
        assert mock_litellm_instance is not None
        mock_litellm_instance.completion = MagicMock(return_value=mock_resp)

        score, reasoning = llm_judge_score("skill", SAMPLE_EPISODE, expected_model)

        call_args = mock_litellm_instance.completion.call_args
        actual_model = call_args.kwargs.get("model")
        assert actual_model == expected_model, (
            f"Expected model '{expected_model}', got '{actual_model}'"
        )
        print(f"PASS: Model parameter '{actual_model}' passed correctly")

    def test_function_signature_unchanged(self):
        """Test: no change to function signature."""
        import importlib
        import inspect

        from src import evaluator as ev

        importlib.reload(ev)
        from src.evaluator import llm_judge_score

        sig = inspect.signature(llm_judge_score)
        params = list(sig.parameters.keys())

        assert "candidate_skill" in params, "Missing candidate_skill param"
        assert "episode" in params, "Missing episode param"
        assert "judge_lm" in params, "Missing judge_lm param"

        assert sig.parameters["judge_lm"].default is not inspect.Parameter.empty, (
            "judge_lm should have a default value"
        )

        print(f"PASS: Signature unchanged: {params}")
        print(f"  candidate_skill: {sig.parameters['candidate_skill']}")
        print(f"  episode: {sig.parameters['episode']}")
        print(f"  judge_lm default: {sig.parameters['judge_lm'].default}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
