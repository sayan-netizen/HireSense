# HireSense
## Multimodal AI-Powered Mock Interview & Behavioural Analysis Platform

HireSense is an integrated virtual interviewer that prepares candidates for technical and HR interviews by evaluating three independent dimensions:

1. **Answer Intelligence**: Technical accuracy, reasoning depth, completeness, and grammar structure evaluated using a locally hosted LLM via Ollama.
2. **Speech Intelligence**: Spoken answer transcription, deterministic filler-word analysis, speech patterns, and optional pronunciation feedback.
3. **Visual Behaviour Intelligence**: Real-time computer vision analysis tracking posture alignment, solvePnP head orientation, iris gaze direction, hand gestures, and posture stability using MediaPipe Tasks.

> **Ethical & Coaching Disclaimer:** Visual measurements are coaching indicators designed to assist self-directed interview practice. They are **not** psychological diagnoses, emotional state classifiers, lie detectors, or validated tools for automated employment screening.

---

## 1. System Architecture

```text
                    HIRESENSE
                        |
                 Streamlit Frontend
                        |
              Interview Session Manager
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     AI INTERVIEW   SPEECH ENGINE   VISION ENGINE
          |             |             |
          v             v             v
     Question       Microphone      Webcam
     Controller     Capture         Capture
          |             |             |
          v             v             v
     Ollama API     FFmpeg          OpenCV
          |             |             |
          v             v             v
     Qwen Model     Speech-to-Text  MediaPipe Tasks
          |             |             |
          v             v             v
     Answer         Transcript      Pose / Face /
     Evaluation     Analysis        Hand Analysis
          |             |             |
          v             v             v
     AI Feedback    Speech Scores   Visual Metrics
          |             |             |
          +-------------+-------------+
                        |
                        v
              INTERVIEW REPORT ENGINE
                        |
                        v
                RESULTS DASHBOARD
                        |
                        v
              PDF / JSON EXPORT
```

---

## 2. Project Structure

```text
INNOVATIVE PROJECT 2026/
│
├── app.py                      # Main unified Streamlit application entry point
├── requirements.txt            # System dependencies
├── README.md                   # System documentation & setup guide
├── .gitignore                  # Git ignore rules
├── .env.example                # Sample environment variables
│
├── config/                     # Application & model configuration
│   ├── __init__.py
│   ├── settings.py             # Global constants, subjects, timeouts
│   └── model_config.py         # MediaPipe asset paths and thresholds
│
├── data/
│   └── questions.csv           # 30 curated Technical & HR interview questions
│
├── core/                       # Orchestration & session coordination
│   ├── __init__.py
│   ├── session_manager.py      # Thread-safe session data store
│   ├── interview_controller.py # Multi-engine interview coordinator
│   ├── question_manager.py     # Question filter, normalizer, and bank
│   └── event_manager.py        # System event logger
│
├── ai_interviewer/             # Local LLM evaluation engine
│   ├── __init__.py
│   ├── ollama_client.py        # Robust Ollama HTTP client with timeout & fallbacks
│   ├── answer_evaluator.py     # 3-score schema evaluator with offline fallback
│   ├── evaluation_prompts.py   # System prompts and adaptive follow-up prompts
│   ├── question_generator.py   # Dynamic follow-up generator
│   └── response_validator.py   # Strict JSON sanitization & bounds clamping
│
├── speech_analysis/            # Speech & verbal delivery engine
│   ├── __init__.py
│   ├── audio_capture.py        # Audio bytes & FFmpeg 16 kHz WAV converter
│   ├── speech_to_text.py       # Google Speech Recognition with correction
│   ├── filler_analyzer.py      # Deterministic regex filler analysis (10 patterns)
│   ├── pronunciation_analyzer.py # OpenPronounce integration
│   └── speech_metrics.py       # Session-wide speech metrics aggregator
│
├── posture_analysis/           # MediaPipe computer vision engine
│   ├── __init__.py
│   ├── config.py               # Biomechanical thresholds & weight profiles
│   ├── engine.py               # PostureEngine class for WebRTC & OpenCV
│   ├── posture_analyzer.py     # Spine lean, shoulder tilt, slouching, stability
│   ├── face_analyzer.py        # solvePnP head yaw/pitch/roll, iris gaze, blinks
│   ├── hand_analyzer.py        # Hand visibility, fidgeting variance, face-touching
│   ├── scoring.py              # Debounced sub-scores, EMA smoothing, session stats
│   ├── visualization.py        # OpenCV & Pillow dashboard renderer
│   └── models/                 # Model assets
│       ├── pose_landmarker.task
│       ├── face_landmarker.task
│       ├── hand_landmarker.task
│       ├── Roboto-Bold.ttf
│       └── Roboto-Regular.ttf
│
├── reporting/                  # Report compilation & export
│   ├── __init__.py
│   ├── report_generator.py     # Unified 3-pillar report generator
│   ├── pdf_exporter.py         # Multi-page PDF generation via fpdf2
│   └── visualizations.py       # Metric formatting utilities
│
├── assets/
│   ├── logo.png                # HireSense branded logo
│   └── styles.css              # Dark professional theme & glassmorphism
│
└── tests/                      # Automated test suite
    ├── test_interview.py       # Questions, session lifecycle, LLM validator
    ├── test_speech.py          # Filler detection & audio metrics
    ├── test_vision.py          # Posture scoring, debouncing, profile weights
    └── test_reporting.py       # Multimodal report & PDF generation
```

---

## 3. Visual Scoring Methodology & Discrepancy Resolution

### Documented Weight Profiles

HireSense explicitly documents the discrepancy between the original `config.py` implementation and the visual `README.md`:

| Metric | Profile A: Code Default (Gesture-Focused) | Profile B: Balanced (Visual README) | Description |
|---|:---:|:---:|---|
| **Posture** | **25%** | **25%** | Sitting alignment, torso lean, shoulder tilt relative to calibration |
| **Gestures** | **25%** | **10%** | Hand composure, fidgeting variance, face-touching |
| **Eye Contact** | **20%** | **20%** | Rolling gaze consistency directed toward camera lens |
| **Head Position** | **15%** | **20%** | Head yaw / pitch / roll orientation via solvePnP |
| **Expression** | **10%** | **15%** | Facial engagement using 52 Face Blendshapes |
| **Stability** | **5%** | **10%** | Torso landmark movement stillness over rolling window |
| **Total** | **100%** | **100%** | Both profiles sum to 1.0 (100%) |

**Default Profile:** Profile A is active by default to preserve the refined gesture emphasis of the latest codebase. Users can switch between profiles via the sidebar or `posture_analysis/config.py`.

---

## 4. Setup and Installation (Windows)

### Prerequisites

1. **Python 3.10 to 3.12** installed on Windows.
2. **Ollama** installed from [ollama.com](https://ollama.com).
3. **FFmpeg** installed and added to your Windows `PATH` (for microphone audio conversion).
4. A standard webcam and microphone.

### Step 1: Install Dependencies

Open PowerShell in the project root:

```powershell
# Install required Python packages
python -m pip install -r requirements.txt
```

### Step 2: Start Ollama and Download Model

Ensure the Ollama application or daemon is running, then pull the recommended Qwen model:

```powershell
ollama pull qwen3.5:4b
```

*(Note: HireSense includes an intelligent offline fallback engine. If Ollama is temporarily offline, the application will degrade gracefully and continue evaluating answers using heuristic concept coverage).*

### Step 3: Run Unit Tests

Verify that all engines, validators, and exporters function correctly:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

All 16 unit tests should pass with `OK`.

---

## 5. Running HireSense

Launch the unified Streamlit application:

```powershell
streamlit run app.py
```

Streamlit will open in your default browser at `http://localhost:8501`.

---

## 6. Application Workflow

1. **Screen 1 — Landing Page**: Overview of HireSense capabilities, system diagnostic checks (Ollama, MediaPipe, FFmpeg, WebRTC).
2. **Screen 2 — Configuration**: Select Candidate Name, Interview Category (Technical, HR, Mixed), Subject (DSA, DBMS, OS, Computer Networks, OOP, General), Difficulty, Number of Questions, Written/Oral mode, and Webcam toggle.
3. **Screen 3 — Calibration**: 4-second baseline calibration capturing individual shoulder width, posture, and head orientation.
4. **Screen 4 — Live Interview**: Virtual interviewer questions on the left; live webcam preview with visual cues on the right. Submit written answers or record oral audio.
5. **Screen 5 — Inline Evaluation**: Real-time content scores (Correctness, Depth, Completeness out of 10), explanations, strengths, areas to improve, filler-word counts, and optional adaptive follow-up questions.
6. **Screen 6 — Final Report & Export**: Comprehensive 3-pillar performance report combining Answer Intelligence, Speech Intelligence, and Visual Behaviour Intelligence. Download as **PDF** or **JSON**.

---

## 7. Privacy & Security

- **Local Inference**: LLM evaluation runs locally via Ollama. No candidate responses are sent to third-party proprietary LLM APIs.
- **Local Vision**: MediaPipe vision inference executes entirely on your local CPU/GPU inside the Python process. No video feeds are recorded or stored without explicit consent.
- **Speech Transparency**: Audio transcription uses Google Web Speech API when recording oral responses, requiring an active internet connection. Written mode operates completely offline.
