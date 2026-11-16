"""
Application Configuration and Settings for HireSense.
"""

import os
from pathlib import Path

# Base Paths
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "posture_analysis" / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Application Metadata
APP_NAME = "HireSense"
APP_TAGLINE = "Multimodal AI-Powered Mock Interview & Behavioural Analysis Platform"
APP_VERSION = "2.0.0"

# LLM / Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT", "180"))
LLM_TEMPERATURE = 0.1
LLM_NUM_PREDICT = 700
LLM_NUM_CTX = 4096

# Data Sources
QUESTIONS_CSV_PATH = DATA_DIR / "questions.csv"

# Subject Mappings & Normalization
SUBJECT_NORMALIZATION = {
    "dsa": "DSA",
    "dbms": "DBMS",
    "os": "Operating Systems",
    "operating systems": "Operating Systems",
    "computer networks": "Computer Networks",
    "cn": "Computer Networks",
    "oop": "OOP",
    "general": "General",
}

AVAILABLE_TECHNICAL_SUBJECTS = ["DSA", "DBMS", "Operating Systems", "Computer Networks", "OOP"]
AVAILABLE_HR_SUBJECTS = ["General"]
AVAILABLE_ALL_SUBJECTS = AVAILABLE_TECHNICAL_SUBJECTS + AVAILABLE_HR_SUBJECTS
AVAILABLE_DIFFICULTIES = ["Easy", "Medium", "Hard"]
AVAILABLE_MODES = ["Written", "Oral"]

# Audio / Speech Recognition
SPEECH_RECOGNITION_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "en-IN")
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Vision / Calibration Settings
CALIBRATION_FRAMES = 120
CALIBRATION_WARMUP_SKIP = 15
CALIBRATION_DURATION_SEC = 4.0
CALIBRATION_OUTLIER_TRIM = 0.10
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
