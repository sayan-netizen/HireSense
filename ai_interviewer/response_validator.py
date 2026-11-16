"""
Response validation and JSON sanitization for AI Interview evaluations.
Ensures strict schema compliance and safe numeric bounds.
"""

import json
import re
from typing import Dict, Any, List

REQUIRED_KEYS = [
    "correctness",
    "depth",
    "completeness",
    "explanation",
    "grammar_spelling",
    "overall_feedback",
    "strengths",
    "areas_to_improve",
    "reference_answer",
]


def extract_json_from_text(text: str) -> str:
    """Extract substring containing JSON object if wrapped in markdown codeblocks."""
    text = text.strip()
    # Check for markdown codeblocks ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Check for raw JSON object { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    return text


def validate_evaluation_json(raw_content: str) -> Dict[str, Any]:
    """
    Parses and validates evaluation JSON from LLM output.
    Enforces score ranges [0.0, 10.0] and list formatting for strengths/improvements.
    """
    cleaned = extract_json_from_text(raw_content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed LLM JSON response: {e}. Raw content: {raw_content[:200]}") from e

    if not isinstance(data, dict):
        raise ValueError(f"LLM evaluation must be a JSON object, got {type(data).__name__}")

    # Check for required fields with fallback defaults if missing
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        # If minor fields are missing, populate sensible defaults
        for k in missing:
            if k in ("correctness", "depth", "completeness"):
                data[k] = 5.0
            elif k in ("strengths", "areas_to_improve"):
                data[k] = []
            else:
                data[k] = "N/A"

    # Enforce score clamping 0.0 - 10.0
    for score_key in ("correctness", "depth", "completeness"):
        try:
            val = float(data.get(score_key, 0.0))
            data[score_key] = max(0.0, min(10.0, val))
        except (ValueError, TypeError):
            data[score_key] = 5.0

    # Ensure list formatting for strengths and areas_to_improve
    for list_key in ("strengths", "areas_to_improve"):
        val = data.get(list_key, [])
        if isinstance(val, str):
            data[list_key] = [s.strip("- ") for s in val.split("\n") if s.strip()]
        elif not isinstance(val, list):
            data[list_key] = [str(val)]
        else:
            data[list_key] = [str(item) for item in val if str(item).strip()]

    # Ensure string types for text fields
    for text_key in ("explanation", "grammar_spelling", "overall_feedback", "reference_answer"):
        if not isinstance(data.get(text_key), str):
            data[text_key] = str(data.get(text_key, ""))

    return data
