"""
Unit tests for ReportGenerator, JSON export, and PDF generation.
"""

import os
import json
import unittest
from reporting.report_generator import ReportGenerator
from reporting.pdf_exporter import export_report_to_pdf, generate_report_pdf_bytes


class TestReportingEngine(unittest.TestCase):

    def setUp(self):
        self.sample_session = {
            "session_id": "test-1234",
            "candidate_name": "Jane Doe",
            "started_at": 1700000000,
            "ended_at": 1700000600,
            "duration_seconds": 600.0,
            "settings": {
                "category": "Technical",
                "subject": "DSA",
                "difficulty": "Easy",
                "answer_mode": "Oral",
            },
            "questions": [
                {"id": 1, "question": "What is a stack?", "reference_answer": "LIFO", "key_concepts": "LIFO;push;pop"}
            ],
            "answers": {
                0: {
                    "question": "What is a stack?",
                    "category": "Technical",
                    "subject": "DSA",
                    "difficulty": "Easy",
                    "reference_answer": "A stack is a LIFO data structure.",
                    "user_answer": "A stack is a linear data structure following Last In First Out.",
                    "evaluation": {
                        "correctness": 9.0,
                        "depth": 8.0,
                        "completeness": 9.0,
                        "explanation": "Clear explanation of LIFO and operations.",
                        "grammar_spelling": "Well articulated.",
                        "overall_feedback": "Strong answer.",
                        "strengths": ["Clear LIFO definition"],
                        "areas_to_improve": ["Mention practical usage such as call stack"],
                        "reference_answer": "A stack is a linear data structure.",
                    },
                }
            },
            "speech_metrics": {
                "total_words": 120,
                "total_fillers": 3,
                "overall_filler_rate": 2.5,
                "top_filler_words": {"like": 2, "um": 1},
                "pronunciation_available": False,
            },
            "visual_metrics": {
                "averages": {
                    "overall": 86.0,
                    "posture": 88.0,
                    "eye_contact": 84.0,
                    "head_position": 85.0,
                    "expression": 82.0,
                    "stability": 90.0,
                    "gestures": 85.0,
                    "reliability": 95.0,
                },
                "events": {
                    "good_posture_frames": 250,
                    "poor_posture_frames": 15,
                    "posture_warnings": 1,
                    "shoulder_tension_events": 0,
                },
                "strengths": ["Good sitting posture", "Consistent eye contact"],
                "improvements": ["Maintain relaxed hands"],
                "is_calibrated": True,
            },
        }

    def test_report_generation(self):
        report = ReportGenerator.generate_report(self.sample_session)
        self.assertIn("overview", report)
        self.assertIn("answer_intelligence", report)
        self.assertIn("speech_intelligence", report)
        self.assertIn("visual_intelligence", report)
        self.assertIn("recommendations", report)

        self.assertEqual(report["overview"]["session_id"], "test-1234")
        self.assertEqual(report["answer_intelligence"]["overall_content_score"], 8.7)
        self.assertEqual(report["answer_intelligence"]["average_content_score"], 8.7)
        self.assertIn("primary_strengths", report["answer_intelligence"])
        self.assertIn("primary_weaknesses", report["answer_intelligence"])
        self.assertEqual(report["visual_intelligence"]["overall_readiness_score"], 86.0)

    def test_visual_intelligence_disabled_when_webcam_off(self):
        session_copy = dict(self.sample_session)
        session_copy["settings"] = dict(self.sample_session["settings"])
        session_copy["settings"]["webcam_enabled"] = False
        report = ReportGenerator.generate_report(session_copy)
        self.assertFalse(report["visual_intelligence"]["is_evaluated"])
        self.assertEqual(report["visual_intelligence"]["posture_score"], 0.0)
        self.assertEqual(report["visual_intelligence"]["reliability_score"], 0.0)

    def test_pdf_export_bytes(self):
        report = ReportGenerator.generate_report(self.sample_session)
        pdf_bytes = generate_report_pdf_bytes(report)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_pdf_export(self):
        report = ReportGenerator.generate_report(self.sample_session)
        out_pdf = "tests/test_output.pdf"
        try:
            pdf_path = export_report_to_pdf(report, out_pdf)
            self.assertTrue(os.path.isfile(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 500)
        finally:
            if os.path.isfile(out_pdf):
                try:
                    os.remove(out_pdf)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
