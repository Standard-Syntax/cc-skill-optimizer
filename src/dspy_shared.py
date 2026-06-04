"""
Shared DSPy infrastructure for run_dspy_gepa and run_dspy_native_gepa.

This module extracts the four DSPy helpers that were previously defined
inline (and had begun to diverge) in both runner functions. After Phase 17,
both run_dspy_gepa and run_dspy_native_gepa in optimize.py import from
this module instead of redefining the helpers locally.

NOTE — DSPy path output format:
  This DSPy path extracts DSPy's internal `signature.instructions` field
  after optimization, NOT a SKILL.md file. So the output format differs
  from GEPA's `optimize_anything` path which writes `best_candidate.md`.
  See improve docs / cc-skill-optimizer-improvements.md for the full note.
"""

from __future__ import annotations

import dspy


class SkillGuidedTask(dspy.Signature):
    """Apply repository skills to complete a software engineering task.

    The skill_instructions field carries the SKILL.md content as
    runtime guidance; task_prompt is the user's actual request; and
    error_context surfaces any prior errors from the session so the
    completion can recover from them.
    """

    skill_instructions: str = dspy.InputField(
        desc="SKILL.md content guiding the agent"
    )
    task_prompt: str = dspy.InputField(
        desc="The software engineering task to complete"
    )
    error_context: str = dspy.InputField(
        desc="Prior errors and context from the session",
        default="",
    )
    completion: str = dspy.OutputField(
        desc="How the agent should approach and complete this task"
    )


class SkillProgram(dspy.Module):
    """Single-predictor DSPy module that applies a fixed skill_content to tasks.

    The skill_content is set at construction (and used as the
    skill_instructions for every forward call). The forward() method
    packages task_prompt + error_context into a SkillGuidedTask call
    and returns the resulting dspy.Prediction.
    """

    def __init__(self, skill_content: str) -> None:
        super().__init__()
        self.skill_content = skill_content
        self.predictor = dspy.Predict(SkillGuidedTask)

    def forward(
        self,
        task_prompt: str,
        error_context: str = "",
    ) -> dspy.Prediction:
        return self.predictor(
            skill_instructions=self.skill_content,
            task_prompt=task_prompt,
            error_context=error_context,
        )


def ep_to_example(ep: dict) -> dspy.Example:
    """Convert a parsed session episode to a DSPy Example for MIPROv2/GEPA demos.

    The completion is the "ideal" response derived from the episode's
    outcome, errors, and command history (see _ideal_completion_from_episode).
    The Example is marked so only task_prompt and error_context are inputs
    (the completion is the supervision signal, not an input).
    """
    errors = "; ".join(ep.get("error_messages", [])[:2])
    return dspy.Example(
        task_prompt=ep.get("task_prompt", ""),
        error_context=errors,
        completion=_ideal_completion_from_episode(ep),
    ).with_inputs("task_prompt", "error_context")


def _ideal_completion_from_episode(ep: dict) -> str:
    """Build a 1-2 sentence 'ideal' completion string from a parsed episode.

    Used as the supervision signal for DSPy MIPROv2 / dspy.GEPA demos —
    the reflection LM uses this as a positive example of what a good
    completion should look like for this kind of session.
    """
    parts: list[str] = []
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
