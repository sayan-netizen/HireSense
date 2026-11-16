"""
Report Generator for HireSense.
Integrates Answer Intelligence, Speech Intelligence, and Visual Behaviour Intelligence
into a structured multimodal performance report.
Maintains independent dimensions without conflating them into an arbitrary composite score.
"""

import time
from typing import Dict, Any, List


class ReportGenerator:
    """Generates structured interview evaluation reports."""

    @staticmethod
    def generate_report(
        session_data: Dict[str, Any],
        speech_summary: Dict[str, Any] = None,
        visual_summary: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes multimodal interview data into a clean report structure.
        """
        answers = session_data.get("answers", {})
        total_questions = len(session_data.get("questions", []))
        completed_questions = len(answers)
        duration_sec = session_data.get("duration_seconds", 0.0)
        mins, secs = divmod(int(duration_sec), 60)

        # ─── SECTION A: INTERVIEW OVERVIEW ───
        overview = {
            "session_id": session_data.get("session_id", "N/A"),
            "candidate_name": session_data.get("candidate_name", "Candidate"),
            "category": session_data.get("settings", {}).get("category", "Technical"),
            "subject": session_data.get("settings", {}).get("subject", "General"),
            "difficulty": session_data.get("settings", {}).get("difficulty", "Easy"),
            "answer_mode": session_data.get("settings", {}).get("answer_mode", "Written"),
            "total_questions": total_questions,
            "completed_questions": completed_questions,
            "duration_formatted": f"{mins:02d}:{secs:02d}",
            "duration_seconds": duration_sec,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_data.get("started_at", time.time()))),
        }

        # ─── SECTION B: ANSWER INTELLIGENCE ───
        correctness_scores = []
        depth_scores = []
        completeness_scores = []
        question_details = []

        for idx, record in answers.items():
            ev = record.get("evaluation", {})
            c = float(ev.get("correctness", 0.0))
            d = float(ev.get("depth", 0.0))
            comp = float(ev.get("completeness", 0.0))
            avg_q = round((c + d + comp) / 3.0, 1)

            correctness_scores.append(c)
            depth_scores.append(d)
            completeness_scores.append(comp)

            question_details.append({
                "question_number": int(idx) + 1,
                "question": record.get("question", ""),
                "category": record.get("category", ""),
                "subject": record.get("subject", ""),
                "difficulty": record.get("difficulty", ""),
                "user_answer": record.get("user_answer", ""),
                "reference_answer": ev.get("reference_answer", record.get("reference_answer", "")),
                "correctness": c,
                "depth": d,
                "completeness": comp,
                "average_score": avg_q,
                "explanation": ev.get("explanation", ""),
                "grammar_spelling": ev.get("grammar_spelling", ""),
                "overall_feedback": ev.get("overall_feedback", ""),
                "strengths": ev.get("strengths", []),
                "areas_to_improve": ev.get("areas_to_improve", []),
                "followup_question": record.get("followup_question"),
            })

        avg_correctness = round(sum(correctness_scores) / len(correctness_scores), 1) if correctness_scores else 0.0
        avg_depth = round(sum(depth_scores) / len(depth_scores), 1) if depth_scores else 0.0
        avg_completeness = round(sum(completeness_scores) / len(completeness_scores), 1) if completeness_scores else 0.0
        overall_content_score = round((avg_correctness + avg_depth + avg_completeness) / 3.0, 1) if correctness_scores else 0.0

        answer_intelligence = {
            "overall_content_score": overall_content_score,
            "average_correctness": avg_correctness,
            "average_depth": avg_depth,
            "average_completeness": avg_completeness,
            "questions_evaluated": question_details,
        }

        # ─── SECTION C: SPEECH INTELLIGENCE ───
        speech_summary = speech_summary or session_data.get("speech_metrics", {})
        speech_intelligence = {
            "is_evaluated": overview["answer_mode"] == "Oral" or bool(speech_summary.get("total_words", 0)),
            "total_words": speech_summary.get("total_words", 0),
            "total_fillers": speech_summary.get("total_fillers", 0),
            "filler_rate_per_100": speech_summary.get("overall_filler_rate", 0.0),
            "top_filler_words": speech_summary.get("top_filler_words", {}),
            "pronunciation_available": speech_summary.get("pronunciation_available", False),
            "average_pronunciation_score": speech_summary.get("average_pronunciation_score"),
            "flagged_words_count": speech_summary.get("flagged_words_count", 0),
        }

        # ─── SECTION D: VISUAL BEHAVIOUR INTELLIGENCE ───
        visual_summary = visual_summary or session_data.get("visual_metrics", {})
        v_averages = visual_summary.get("averages", {})
        v_events = visual_summary.get("events", {})

        visual_intelligence = {
            "is_evaluated": bool(v_averages),
            "overall_readiness_score": v_averages.get("overall", 0.0),
            "posture_score": v_averages.get("posture", 0.0),
            "gestures_score": v_averages.get("gestures", 0.0),
            "eye_contact_score": v_averages.get("eye_contact", 0.0),
            "head_position_score": v_averages.get("head_position", 0.0),
            "expression_score": v_averages.get("expression", 0.0),
            "stability_score": v_averages.get("stability", 0.0),
            "reliability_score": v_averages.get("reliability", 100.0),
            "good_posture_frames": v_events.get("good_posture_frames", 0),
            "poor_posture_frames": v_events.get("poor_posture_frames", 0),
            "posture_warnings": v_events.get("posture_warnings", 0),
            "sustained_gaze_away_events": v_events.get("sustained_gaze_away_events", 0),
            "distraction_events": v_events.get("distraction_events", 0),
            "shoulder_tension_events": v_events.get("shoulder_tension_events", 0),
            "strengths": visual_summary.get("strengths", []),
            "improvements": visual_summary.get("improvements", []),
            "is_calibrated": visual_summary.get("is_calibrated", False),
        }

        # ─── SECTION E: ACTIONABLE RECOMMENDATIONS ───
        recommendations = []

        # Content recommendations
        if avg_correctness < 7.0:
            recommendations.append(
                "Strengthen technical definitions: Review foundational concepts and verify core terminology before framing solutions."
            )
        if avg_depth < 7.0:
            recommendations.append(
                "Enhance explanation depth: Incorporate trade-offs, edge cases, time/space complexity, or practical implementation details."
            )
        if avg_completeness < 7.0:
            recommendations.append(
                "Cover essential key concepts: Structure your answers using a framework like Situation-Definition-Implementation-Result."
            )

        # Speech recommendations
        if speech_intelligence["is_evaluated"]:
            filler_rate = speech_intelligence["filler_rate_per_100"]
            if filler_rate > 5.0:
                recommendations.append(
                    f"Reduce vocal fillers: Filler rate was {filler_rate:.1f} per 100 words. Practice pausing briefly to think rather than filling pauses with 'um' or 'like'."
                )
            if speech_intelligence["pronunciation_available"] and speech_intelligence.get("average_pronunciation_score", 100) < 70:
                recommendations.append(
                    "Articulate technical terminology carefully: Review flagged pronunciation terms for clarity."
                )

        # Visual coaching recommendations
        if visual_intelligence["is_evaluated"]:
            if visual_intelligence["reliability_score"] < 60:
                recommendations.append(
                    "Adjust camera and lighting: Landmark detection reliability was low. Ensure your webcam is at eye level with balanced front lighting."
                )
            else:
                if visual_intelligence["posture_score"] < 75:
                    recommendations.append(
                        "Maintain an upright, comfortable seating posture: Align your shoulders and avoid forward neck hunching."
                    )
                if visual_intelligence["eye_contact_score"] < 75:
                    recommendations.append(
                        "Position gaze toward the webcam lens: Natural, steady eye contact fosters stronger perceived engagement."
                    )
                if visual_intelligence["gestures_score"] < 75:
                    recommendations.append(
                        "Calm hand movements: Keep hands in a relaxed resting position or use open, natural illustrative gestures."
                    )
                if visual_intelligence["shoulder_tension_events"] > 0:
                    recommendations.append(
                        "Relieve upper-body tension: Drop your shoulders and take deep, steady breaths when answering complex questions."
                    )

        if not recommendations:
            recommendations.append("Outstanding performance across all evaluated technical, verbal, and visual dimensions!")

        return {
            "overview": overview,
            "answer_intelligence": answer_intelligence,
            "speech_intelligence": speech_intelligence,
            "visual_intelligence": visual_intelligence,
            "recommendations": recommendations,
        }
