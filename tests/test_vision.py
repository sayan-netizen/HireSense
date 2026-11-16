"""
Unit tests for Computer Vision Scoring and PostureEngine models.
Verifies model asset presence and mathematical scoring logic without requiring a physical camera.
"""

import os
import unittest
from posture_analysis.scoring import InterviewScorer, StatusDebouncer
from posture_analysis.config import (
    SCORING_PROFILES,
    set_scoring_profile,
    ACTIVE_SCORING_PROFILE,
)
from posture_analysis.engine import PostureEngine


class TestVisionEngine(unittest.TestCase):

    def test_model_files_exist(self):
        models_dir = "posture_analysis/models"
        self.assertTrue(os.path.isfile(os.path.join(models_dir, "pose_landmarker.task")), "pose_landmarker.task missing")
        self.assertTrue(os.path.isfile(os.path.join(models_dir, "face_landmarker.task")), "face_landmarker.task missing")
        self.assertTrue(os.path.isfile(os.path.join(models_dir, "hand_landmarker.task")), "hand_landmarker.task missing")

    def test_scoring_profiles_discrepancy_resolution(self):
        self.assertIn("code_default", SCORING_PROFILES)
        self.assertIn("readme_balanced", SCORING_PROFILES)

        # Verify weights sum to 1.0 (100%)
        sum_code = sum(SCORING_PROFILES["code_default"].values())
        sum_readme = sum(SCORING_PROFILES["readme_balanced"].values())
        self.assertAlmostEqual(sum_code, 1.0, places=4)
        self.assertAlmostEqual(sum_readme, 1.0, places=4)

        # Test profile switching
        set_scoring_profile("readme_balanced")
        from posture_analysis import config
        self.assertEqual(config.ACTIVE_SCORING_PROFILE, "readme_balanced")
        self.assertEqual(config.WEIGHT_GESTURES, 0.10)

        # Reset to code_default
        set_scoring_profile("code_default")
        self.assertEqual(config.ACTIVE_SCORING_PROFILE, "code_default")
        self.assertEqual(config.WEIGHT_GESTURES, 0.25)

    def test_status_debouncer(self):
        debouncer = StatusDebouncer(initial_state="Good", frames_required=3)
        self.assertEqual(debouncer.update("Good"), "Good")
        # 1st frame with new state -> still Good
        self.assertEqual(debouncer.update("Poor"), "Good")
        # 2nd frame with new state -> still Good
        self.assertEqual(debouncer.update("Poor"), "Good")
        # 3rd frame with new state -> shifts to Poor
        self.assertEqual(debouncer.update("Poor"), "Poor")

    def test_interview_scorer_initialization(self):
        scorer = InterviewScorer()
        summary = scorer.get_session_summary()
        self.assertIn("INTERVIEW SESSION SUMMARY", summary)
        averages = scorer.get_averages()
        self.assertIn("posture", averages)
        self.assertIn("overall", averages)
        self.assertIn("reliability", averages)


if __name__ == "__main__":
    unittest.main()
