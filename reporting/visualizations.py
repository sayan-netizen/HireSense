"""
Visualization helpers for reporting metrics in Streamlit and reports.
"""

from typing import Dict, Any


def get_score_color_hex(score: float, max_score: float = 10.0) -> str:
    """Return hex color reflecting score tier."""
    pct = (score / max_score) * 100.0
    if pct >= 80:
        return "#34d399" # Emerald green
    if pct >= 65:
        return "#60a5fa" # Blue
    if pct >= 50:
        return "#fbbf24" # Amber
    return "#f87171"     # Red


def format_subscores_markdown(evaluation: Dict[str, Any]) -> str:
    """Format markdown metric badges for answer evaluation."""
    c = evaluation.get("correctness", 0.0)
    d = evaluation.get("depth", 0.0)
    comp = evaluation.get("completeness", 0.0)
    avg = (c + d + comp) / 3.0

    return f"""
| Dimension | Score | Status |
|---|---|---|
| **Correctness** | `{c:g}/10` | {'[STRONG]' if c >= 7 else '[MODERATE]' if c >= 5 else '[NEEDS FOCUS]'} |
| **Depth** | `{d:g}/10` | {'[IN-DEPTH]' if d >= 7 else '[MODERATE]' if d >= 5 else '[BRIEF]'} |
| **Completeness** | `{comp:g}/10` | {'[COMPREHENSIVE]' if comp >= 7 else '[PARTIAL]' if comp >= 5 else '[INCOMPLETE]'} |
| **Content Average** | **`{avg:.1f}/10`** | **{'[STRONG]' if avg >= 7 else '[COMPETENT]' if avg >= 5 else '[DEVELOPING]'}** |
"""
