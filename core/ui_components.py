"""
UI Component Library for HireSense
Nothing OS / Teenage Engineering Technical Minimalist Design System.
Provides reusable, accessible UI elements for headers, selection cards, metric readouts,
feedback displays, and distraction-free interview containers.
"""

from typing import Dict, Any, List, Optional
import streamlit as st


def clean_html(raw_html: str) -> str:
    """
    Strips all leading whitespace from each line of an HTML string.
    This prevents Streamlit / CommonMark markdown parsers from misinterpreting
    indented lines (>= 4 spaces) as code blocks with copy buttons.
    """
    return "\n".join(line.lstrip() for line in raw_html.strip().splitlines())


# =============================================================================
# TYPOGRAPHY & HEADERS
# =============================================================================

def render_page_header(
    title: str,
    tagline: str,
    badge_text: Optional[str] = None,
    badge_variant: str = "default",
) -> None:
    """Render a unified technical minimalist page header."""
    badge_html = ""
    if badge_text:
        variant_class = {
            "green": "nothing-badge-green",
            "amber": "nothing-badge-amber",
            "red": "nothing-badge-red",
        }.get(badge_variant, "")
        badge_html = f'<span class="nothing-badge {variant_class}">[{badge_text}]</span>'

    html = f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.35rem;">
            <span class="nothing-label" style="margin-bottom: 0;">// HIRESENSE INTERVIEW OS</span>
            {badge_html}
        </div>
        <div class="nothing-title">{title}</div>
        <div class="nothing-subtitle">{tagline}</div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_section_title(label: str, hint: Optional[str] = None) -> None:
    """Render a technical section divider with optional hint text."""
    hint_html = f'<span style="color: var(--beige-dim); font-size: 0.70rem; letter-spacing: 0.05em;">{hint}</span>' if hint else ""
    html = f"""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 1.25rem; margin-bottom: 0.6rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.35rem;">
        <span class="nothing-label" style="margin-bottom: 0;">// {label}</span>
        {hint_html}
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


# =============================================================================
# STATUS BADGES & INDICATORS
# =============================================================================

def render_status_pill(
    text: str,
    status: str = "default",
    show_live_dot: bool = False,
) -> str:
    """Return raw HTML for a styled technical status pill."""
    variant_class = {
        "green": "nothing-badge-green",
        "amber": "nothing-badge-amber",
        "red": "nothing-badge-red",
    }.get(status, "")

    dot_html = '<span class="nothing-live-dot"></span>' if show_live_dot else ""
    return f'<span class="nothing-badge {variant_class}">{dot_html}[ {text} ]</span>'


# =============================================================================
# METRIC CARDS & SCORE READOUTS
# =============================================================================

def render_score_card(
    label: str,
    score: float,
    max_score: float = 10.0,
    status_label: Optional[str] = None,
    color_tier: Optional[str] = None,
) -> str:
    """Generate HTML for a sleek score metric tile."""
    tier_color = "var(--beige-light)"
    if color_tier == "green" or score >= 7.5:
        tier_color = "var(--tech-green)"
    elif color_tier == "amber" or score >= 5.0:
        tier_color = "var(--tech-amber)"
    elif color_tier == "red":
        tier_color = "#ff6b5e"

    status_html = f'<div style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--beige-muted); text-transform: uppercase; margin-top: 0.25rem;">[ {status_label} ]</div>' if status_label else ""

    html = f"""
    <div class="nothing-card" style="padding: 1rem 1.15rem; margin-bottom: 0.75rem;">
        <div class="nothing-label">{label}</div>
        <div style="display: flex; align-items: baseline; gap: 0.25rem;">
            <span class="nothing-metric-val" style="color: {tier_color};">{score:g}</span>
            <span class="nothing-metric-unit">/{max_score:g}</span>
        </div>
        {status_html}
    </div>
    """
    return clean_html(html)


# =============================================================================
# SELECTION CARDS
# =============================================================================

def render_selection_card_html(
    title: str,
    description: str,
    selected: bool = False,
    badge: Optional[str] = None,
) -> str:
    """Generate HTML for an interactive selection option card with zero indent to avoid CommonMark code block parsing."""
    border_style = "border-color: var(--border-beige); background: #18181c;" if selected else ""
    active_marker = '<span style="color: var(--tech-green); font-family: var(--font-mono); font-size: 0.70rem;">[*] ACTIVE</span>' if selected else ""
    badge_html = f'<span class="nothing-badge">{badge}</span>' if badge else ""

    html = f"""
    <div class="nothing-card nothing-selection-card" style="{border_style} padding: 1.1rem 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
            <span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.85rem; color: var(--beige-light); letter-spacing: 0.05em;">{title}</span>
            <div style="display: flex; align-items: center; gap: 0.4rem;">
                {badge_html}
                {active_marker}
            </div>
        </div>
        <div style="font-size: 0.78rem; color: var(--beige-muted); line-height: 1.45;">
            {description}
        </div>
    </div>
    """
    return clean_html(html)


# =============================================================================
# QUESTION CONTAINERS & TOP NAVIGATION
# =============================================================================

def render_interview_topbar(
    current_index: int,
    total_questions: int,
    category: str,
    subject: str,
    difficulty: str,
    time_elapsed_formatted: str = "00:00",
) -> None:
    """Render distraction-free top navigation bar for live interview."""
    progress_pct = int(((current_index + 1) / max(1, total_questions)) * 100)
    
    html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; margin-bottom: 1.25rem; border-bottom: 1px solid var(--border-subtle);">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span class="nothing-badge" style="font-weight: 700;">QUESTION {current_index + 1:02d} / {total_questions:02d}</span>
            <span class="nothing-badge nothing-badge-green"><span class="nothing-live-dot"></span>LIVE SESSION</span>
            <span class="nothing-badge">{category.upper()} // {subject.upper()}</span>
            <span class="nothing-badge nothing-badge-amber">{difficulty.upper()}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem; font-family: var(--font-mono); font-size: 0.78rem;">
            <span style="color: var(--beige-muted);">TIME: <b style="color: var(--beige-light);">{time_elapsed_formatted}</b></span>
            <span style="color: var(--beige-muted);">PROGRESS: <b style="color: var(--beige-light);">{progress_pct}%</b></span>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_question_box(question_text: str, question_index: int) -> None:
    """Render high-clarity focal question box."""
    html = f"""
    <div class="nothing-question-box">
        <div class="nothing-label">// INTERVIEWER PROMPT :: QUESTION {question_index + 1:02d}</div>
        <div class="nothing-question-text">{question_text}</div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


# =============================================================================
# SYSTEM READINESS INDICATOR STRIP
# =============================================================================

def render_system_readiness_strip(
    ollama_ok: bool,
    ollama_models_count: int,
    vision_ok: bool,
    audio_ok: bool,
) -> None:
    """Render a clean, non-intrusive system readiness overview bar."""
    ollama_pill = (
        render_status_pill(f"OLLAMA ONLINE // {ollama_models_count} MODELS", "green", show_live_dot=True)
        if ollama_ok else
        render_status_pill("OLLAMA OFFLINE // HEURISTIC FALLBACK ACTIVE", "amber")
    )
    
    vision_pill = (
        render_status_pill("VISION ENGINE // READY", "green", show_live_dot=True)
        if vision_ok else
        render_status_pill("VISION OFFLINE // HEADLESS", "amber")
    )

    audio_pill = (
        render_status_pill("AUDIO & SPEECH // READY", "green")
        if audio_ok else
        render_status_pill("FFMPEG MISSING // WAV ONLY", "amber")
    )

    html = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 1.25rem; padding: 0.65rem 0.85rem; background: #0c0c0e; border: 1px solid var(--border-subtle); border-radius: 4px;">
        <span class="nothing-label" style="align-self: center; margin-bottom: 0;">SYSTEM STATUS:</span>
        {ollama_pill}
        {vision_pill}
        {audio_pill}
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)
