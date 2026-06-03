"""
utils.py
========
Shared utilities for LLM response parsing used by GEPA evaluators.
"""

from __future__ import annotations

import json
import re


def _parse_llm_json(raw: str, default: dict | list) -> dict | list:
    """Strip markdown code fences and parse LLM JSON response."""
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default
