"""
Model paths and inference hyperparameters for vision and LLM models.
"""

import os
from .settings import MODELS_DIR

# MediaPipe Task Asset Paths
POSE_MODEL_PATH = str(MODELS_DIR / "pose_landmarker.task")
FACE_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")

# Font Paths for OpenCV Canvas Rendering
FONT_BOLD_PATH = str(MODELS_DIR / "Roboto-Bold.ttf")
FONT_REGULAR_PATH = str(MODELS_DIR / "Roboto-Regular.ttf")

# MediaPipe Inference Parameters
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
FACE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MIN_TRACKING_CONFIDENCE = 0.5
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

# Standard Evaluation JSON Schema Keys
REQUIRED_EVALUATION_KEYS = [
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
