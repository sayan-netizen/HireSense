"""
Filler-word analysis module for oral interview responses.
Preserves the exact deterministic regex patterns and word frequency calculations.
"""

import re
from typing import Dict, Any


FILLER_PATTERNS = {
    "um": r"\bum+\b",
    "uh": r"\buh+\b",
    "erm": r"\berm+\b",
    "like": r"\blike\b",
    "you know": r"\byou know\b",
    "actually": r"\bactually\b",
    "basically": r"\bbasically\b",
    "I mean": r"\bi mean\b",
    "sort of": r"\bsort of\b",
    "kind of": r"\bkind of\b",
}


def analyze_fillers(transcript: str) -> Dict[str, Any]:
    """
    Analyze transcript for filler words and compute frequency metrics.
    Returns:
        {
            "counts": dict of {word: count},
            "total": total filler count,
            "word_count": total words in transcript,
            "rate_per_100_words": filler frequency per 100 words,
            "fillers_detected": boolean,
        }
    """
    if not transcript:
        return {
            "counts": {},
            "total": 0,
            "word_count": 0,
            "rate_per_100_words": 0.0,
            "fillers_detected": False,
        }

    counts = {
        word: len(re.findall(pattern, transcript, re.I))
        for word, pattern in FILLER_PATTERNS.items()
    }
    filtered_counts = {w: n for w, n in counts.items() if n > 0}
    total_fillers = sum(filtered_counts.values())

    words = re.findall(r"\b[\w']+\b", transcript)
    word_count = len(words)

    rate = (total_fillers / word_count * 100.0) if word_count > 0 else 0.0

    return {
        "counts": filtered_counts,
        "total": total_fillers,
        "word_count": word_count,
        "rate_per_100_words": round(rate, 1),
        "fillers_detected": total_fillers > 0,
    }
