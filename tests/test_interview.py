"""
Unit tests for QuestionManager, SessionManager, and AnswerEvaluator components.
Runs without external hardware or live network dependencies.
"""

import unittest
from core.question_manager import QuestionManager
from core.session_manager import InterviewSessionManager
from ai_interviewer.response_validator import validate_evaluation_json
from ai_interviewer.answer_evaluator import AnswerEvaluator


class TestInterviewEngine(unittest.TestCase):

    def setUp(self):
        self.qm = QuestionManager("data/questions.csv")
        self.sm = InterviewSessionManager()

    def test_question_manager_loading(self):
        self.assertGreater(len(self.qm.df), 0, "Question bank should not be empty")
        self.assertIn("category", self.qm.df.columns)
        self.assertIn("key_concepts", self.qm.df.columns)

    def test_subject_filtering_technical(self):
        dsa_qs = self.qm.filter_questions(interview_type="Technical", subject="DSA", difficulty="Easy")
        self.assertGreater(len(dsa_qs), 0)
        for q in dsa_qs:
            self.assertEqual(q["category"].lower(), "technical")
            self.assertEqual(q["subject"].lower(), "dsa")

    def test_os_subject_normalization(self):
        os_qs = self.qm.filter_questions(interview_type="Technical", subject="Operating Systems", difficulty="Easy")
        self.assertGreater(len(os_qs), 0, "Normalizing 'Operating Systems' to 'OS' should return questions")

    def test_fallback_preserves_subject(self):
        # If difficulty does not exist, subject filter must still be strictly preserved
        qs = self.qm.filter_questions(interview_type="Technical", subject="OS", difficulty="NonExistentDiff")
        self.assertGreater(len(qs), 0)
        for q in qs:
            self.assertEqual(q["subject"].lower(), "os")

    def test_hard_difficulty_questions_exist_across_all_subjects(self):
        for subj in ["DSA", "DBMS", "OS", "Computer Networks", "OOP", "HR"]:
            itype = "HR" if subj == "HR" else "Technical"
            qs = self.qm.filter_questions(interview_type=itype, subject=subj, difficulty="Hard")
            self.assertGreater(len(qs), 0, f"Subject {subj} should have Hard difficulty questions")

    def test_session_lifecycle(self):
        qs = self.qm.filter_questions(num_questions=3)
        self.sm.start_session("Alice", {"category": "Technical"}, qs)
        self.assertTrue(self.sm.session_data["is_active"])
        self.assertEqual(self.sm.session_data["current_index"], 0)

        # Advance question
        can_advance = self.sm.advance_question()
        self.assertTrue(can_advance)
        self.assertEqual(self.sm.session_data["current_index"], 1)

    def test_response_validator_sanitization(self):
        sample_json = """
        {
            "correctness": 8.5,
            "depth": 7.0,
            "completeness": 9.0,
            "explanation": "Solid explanation of stack data structures.",
            "grammar_spelling": "Clear phrasing.",
            "overall_feedback": "Good answer.",
            "strengths": ["Clear LIFO explanation", "Accurate operations"],
            "areas_to_improve": ["Add edge case regarding overflow"],
            "reference_answer": "A stack is LIFO."
        }
        """
        validated = validate_evaluation_json(sample_json)
        self.assertEqual(validated["correctness"], 8.5)
        self.assertEqual(len(validated["strengths"]), 2)
        self.assertIsInstance(validated["areas_to_improve"], list)

    def test_response_validator_bounds_clamping(self):
        out_of_bounds = """
        {
            "correctness": 15.0,
            "depth": -3.0,
            "completeness": 10.0,
            "explanation": "Test",
            "grammar_spelling": "Test",
            "overall_feedback": "Test",
            "strengths": "Single string strength",
            "areas_to_improve": [],
            "reference_answer": "Test"
        }
        """
        validated = validate_evaluation_json(out_of_bounds)
        self.assertEqual(validated["correctness"], 10.0, "Score > 10 should clamp to 10")
        self.assertEqual(validated["depth"], 0.0, "Score < 0 should clamp to 0")
        self.assertIsInstance(validated["strengths"], list, "String strength should convert to list")


if __name__ == "__main__":
    unittest.main()
