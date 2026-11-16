"""
PDF Exporter: Generates publication-quality interview reports using fpdf2.
Sanitizes Unicode and formatting to prevent font encoding issues.
"""

import os
from typing import Dict, Any
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def sanitize_text(text: Any) -> str:
    """Sanitize strings for standard PDF font compatibility (latin-1)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u2022": "*",
        "\u00b0": " deg",
        "\u2192": "->",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)

    # Encode to latin-1 replacing unknown chars with '?'
    return text.encode("latin-1", "replace").decode("latin-1")


class HireSensePDFReport(FPDF):
    """Custom styled PDF for HireSense interview reports."""

    def header(self):
        self.set_fill_color(18, 22, 36) # Dark professional header
        self.rect(0, 0, 210, 24, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 14)
        self.set_xy(14, 5)
        self.cell(0, 8, "HIRESENSE | AI Interview & Behavioural Performance Report", align="L")
        self.set_font("helvetica", "I", 9)
        self.set_xy(14, 13)
        self.set_text_color(180, 190, 210)
        self.cell(0, 6, "Multimodal Performance Analysis: Technical - Verbal - Visual", align="L")
        self.ln(16)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(130, 135, 150)
        self.cell(0, 10, f"HireSense Evaluation Report | Page {self.page_no()}/{{nb}}", align="C")


def export_report_to_pdf(report: Dict[str, Any], output_path: str) -> str:
    """
    Generates a comprehensive multi-page PDF from the unified report dictionary.
    Returns the file path of the generated PDF.
    """
    pdf = HireSensePDFReport(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    ov = report.get("overview", {})
    ans_intel = report.get("answer_intelligence", {})
    speech_intel = report.get("speech_intelligence", {})
    vis_intel = report.get("visual_intelligence", {})
    recs = report.get("recommendations", [])

    # ── METADATA SUMMARY TABLE ──
    pdf.set_xy(14, 28)
    pdf.set_fill_color(245, 247, 252)
    pdf.set_draw_color(220, 225, 235)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(40, 50, 70)
    pdf.cell(182, 8, "  INTERVIEW SESSION SUMMARY", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

    pdf.set_font("helvetica", "", 9)
    col_w = 45.5
    row1 = [
        ("Candidate:", sanitize_text(ov.get("candidate_name", "Candidate"))),
        ("Session ID:", sanitize_text(ov.get("session_id", "N/A"))),
        ("Category:", sanitize_text(ov.get("category", "Technical"))),
        ("Subject:", sanitize_text(ov.get("subject", "General"))),
    ]
    for lbl, val in row1:
        pdf.cell(20, 6, lbl, border="L", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_w - 20, 6, val, border="R", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()

    row2 = [
        ("Difficulty:", sanitize_text(ov.get("difficulty", "Easy"))),
        ("Answer Mode:", sanitize_text(ov.get("answer_mode", "Written"))),
        ("Duration:", sanitize_text(ov.get("duration_formatted", "00:00"))),
        ("Questions:", f"{ov.get('completed_questions', 0)}/{ov.get('total_questions', 0)}"),
    ]
    for lbl, val in row2:
        pdf.cell(20, 6, lbl, border="LB", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_w - 20, 6, val, border="RB", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(10)

    # ── THREE SCORE PILLARS CARDS ──
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(20, 25, 40)
    pdf.cell(0, 7, "MULTIMODAL ASSESSMENT PILLARS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    card_w = 58.0
    gap = 4.0
    start_y = pdf.get_y()

    # Pillar 1: Answer Intelligence
    pdf.set_xy(14, start_y)
    pdf.set_fill_color(238, 244, 255)
    pdf.set_draw_color(180, 205, 250)
    pdf.rect(14, start_y, card_w, 32, "FD")
    pdf.set_xy(16, start_y + 2)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(25, 60, 150)
    pdf.cell(card_w - 4, 5, "ANSWER INTELLIGENCE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(16)
    pdf.set_font("helvetica", "B", 15)
    pdf.cell(card_w - 4, 8, f"{ans_intel.get('overall_content_score', 0):.1f} / 10", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(16)
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(60, 70, 90)
    pdf.cell(card_w - 4, 4, f"Correctness: {ans_intel.get('average_correctness', 0):.1f}/10", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(16)
    pdf.cell(card_w - 4, 4, f"Depth: {ans_intel.get('average_depth', 0):.1f} | Comp: {ans_intel.get('average_completeness', 0):.1f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Pillar 2: Speech Intelligence
    x2 = 14 + card_w + gap
    pdf.set_xy(x2, start_y)
    pdf.set_fill_color(245, 250, 240)
    pdf.set_draw_color(190, 225, 175)
    pdf.rect(x2, start_y, card_w, 32, "FD")
    pdf.set_xy(x2 + 2, start_y + 2)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(40, 110, 50)
    pdf.cell(card_w - 4, 5, "SPEECH INTELLIGENCE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(x2 + 2)
    pdf.set_font("helvetica", "B", 15)
    if speech_intel.get("is_evaluated"):
        pdf.cell(card_w - 4, 8, f"{speech_intel.get('filler_rate_per_100', 0):.1f} /100 words", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x2 + 2)
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(60, 70, 90)
        pdf.cell(card_w - 4, 4, f"Total Words: {speech_intel.get('total_words', 0)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x2 + 2)
        pdf.cell(card_w - 4, 4, f"Fillers Detected: {speech_intel.get('total_fillers', 0)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.cell(card_w - 4, 8, "Written Mode", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x2 + 2)
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(100, 105, 120)
        pdf.cell(card_w - 4, 8, "Speech evaluation not active", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Pillar 3: Visual Behaviour Intelligence
    x3 = x2 + card_w + gap
    pdf.set_xy(x3, start_y)
    pdf.set_fill_color(252, 245, 255)
    pdf.set_draw_color(225, 195, 245)
    pdf.rect(x3, start_y, card_w, 32, "FD")
    pdf.set_xy(x3 + 2, start_y + 2)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(110, 40, 150)
    pdf.cell(card_w - 4, 5, "VISUAL BEHAVIOUR", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(x3 + 2)
    pdf.set_font("helvetica", "B", 15)
    if vis_intel.get("is_evaluated"):
        pdf.cell(card_w - 4, 8, f"{vis_intel.get('overall_readiness_score', 0):.0f} / 100", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x3 + 2)
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(60, 70, 90)
        pdf.cell(card_w - 4, 4, f"Posture: {vis_intel.get('posture_score', 0):.0f} | Gaze: {vis_intel.get('eye_contact_score', 0):.0f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x3 + 2)
        pdf.cell(card_w - 4, 4, f"Gestures: {vis_intel.get('gestures_score', 0):.0f} | Rel: {vis_intel.get('reliability_score', 100):.0f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.cell(card_w - 4, 8, "Camera Off", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x3 + 2)
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(100, 105, 120)
        pdf.cell(card_w - 4, 8, "Webcam analysis omitted", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(start_y + 36)

    # ── SECTION E: RECOMMENDATIONS HIGHLIGHT ──
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(20, 25, 40)
    pdf.cell(0, 7, "ACTIONABLE COACHING RECOMMENDATIONS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(40, 45, 60)
    for rec in recs[:5]:
        pdf.multi_cell(182, 5.5, f"* {sanitize_text(rec)}")
    pdf.ln(4)

    # ── SECTION B: QUESTION DETAILS ──
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(20, 25, 40)
    pdf.cell(0, 7, "QUESTION-BY-QUESTION EVALUATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    q_list = ans_intel.get("questions_evaluated", [])
    for q in q_list:
        pdf.set_draw_color(210, 215, 230)
        pdf.set_fill_color(248, 250, 254)
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(25, 30, 45)
        pdf.cell(182, 6, f" Question {q['question_number']}: {sanitize_text(q['question'])}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

        pdf.set_font("helvetica", "I", 8.5)
        pdf.set_text_color(70, 75, 90)
        ans_preview = sanitize_text(q.get("user_answer", ""))
        if len(ans_preview) > 280:
            ans_preview = ans_preview[:280] + "..."
        pdf.multi_cell(182, 5, f"Candidate Answer: {ans_preview}", border="LR")

        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(40, 50, 70)
        score_line = f"Scores -> Correctness: {q['correctness']:.1f}/10 | Depth: {q['depth']:.1f}/10 | Completeness: {q['completeness']:.1f}/10 | Average: {q['average_score']:.1f}/10"
        pdf.cell(182, 5.5, score_line, border="LR", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        expl = sanitize_text(q.get("explanation", ""))
        if expl:
            pdf.multi_cell(182, 5, f"Feedback: {expl}", border="LR")

        if q.get("strengths"):
            st_text = "; ".join(sanitize_text(s) for s in q["strengths"])
            pdf.multi_cell(182, 5, f"Strengths: {st_text}", border="LR")

        if q.get("areas_to_improve"):
            imp_text = "; ".join(sanitize_text(i) for i in q["areas_to_improve"])
            pdf.multi_cell(182, 5, f"Improvements: {imp_text}", border="LR")

        ref = sanitize_text(q.get("reference_answer", ""))
        if ref:
            if len(ref) > 180:
                ref = ref[:180] + "..."
            pdf.set_font("helvetica", "I", 8)
            pdf.multi_cell(182, 4.5, f"Reference Answer: {ref}", border="LRB")
        else:
            pdf.cell(182, 1, "", border="T", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(3)

    # ── SECTION D: VISUAL CUES DISCLAIMER ──
    if vis_intel.get("is_evaluated"):
        pdf.ln(3)
        pdf.set_font("helvetica", "I", 7.5)
        pdf.set_text_color(110, 115, 130)
        disclaimer = (
            "NOTICE: Visual measurements are estimated behavioural coaching indicators designed for practice. "
            "They do NOT represent psychological assessments, emotional states, honesty, or professional competence."
        )
        pdf.multi_cell(182, 4, sanitize_text(disclaimer))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf.output(output_path)
    return output_path
