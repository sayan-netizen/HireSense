"""
HireSense :: Multimodal AI-Powered Mock Interview & Behavioural Analysis Platform.
Nothing UI Technical Minimalist Edition (Pitch Black & Warm Beige with Neon Green Wireframes).
"""

import os
import json
import time
from typing import Optional
import cv2
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# Configuration
from config.settings import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    AVAILABLE_TECHNICAL_SUBJECTS,
    AVAILABLE_HR_SUBJECTS,
    AVAILABLE_ALL_SUBJECTS,
    AVAILABLE_DIFFICULTIES,
    AVAILABLE_MODES,
    DEFAULT_OLLAMA_MODEL,
    CALIBRATION_DURATION_SEC,
)

# Core Engines
from core.session_manager import InterviewSessionManager
from core.question_manager import QuestionManager
from core.interview_controller import InterviewController
from core.telemetry_server import start_telemetry_server
from ai_interviewer.ollama_client import OllamaClient
from speech_analysis.audio_capture import is_ffmpeg_available
from speech_analysis.speech_to_text import transcribe_audio
from speech_analysis.filler_analyzer import analyze_fillers
from speech_analysis.pronunciation_analyzer import analyze_pronunciation, is_openpronounce_available
from posture_analysis.engine import PostureEngine
from posture_analysis.config import set_scoring_profile, SCORING_PROFILES, ACTIVE_SCORING_PROFILE
from reporting.report_generator import ReportGenerator
from reporting.pdf_exporter import export_report_to_pdf
from reporting.visualizations import format_subscores_markdown, get_score_color_hex

# WebRTC Streaming
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# =============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# =============================================================================

st.set_page_config(
    page_title=f"{APP_NAME} :: Multimodal AI Interview Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Nothing UI CSS (Black & Warm Beige with Technical Monospace) — cached
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "styles.css")
if "css_loaded" not in st.session_state:
    if os.path.isfile(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.session_state._css_content = f.read()
    else:
        st.session_state._css_content = ""
    st.session_state.css_loaded = True
if st.session_state._css_content:
    st.markdown(f"<style>{st.session_state._css_content}</style>", unsafe_allow_html=True)


# =============================================================================
# PERSISTENT SESSION STATE INITIALIZATION
# =============================================================================

if "controller" not in st.session_state:
    st.session_state.controller = InterviewController()

if "question_manager" not in st.session_state:
    st.session_state.question_manager = QuestionManager()

if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = OllamaClient()

if "camera_mirror" not in st.session_state:
    # User request: "i dont the camera view to be mirrored" -> False by default
    st.session_state.camera_mirror = False

if "posture_engine" not in st.session_state:
    try:
        engine = PostureEngine()
        engine.start()
        st.session_state.posture_engine = engine
        start_telemetry_server(engine.get_current_metrics)
    except Exception as e:
        st.session_state.posture_engine = None
        st.session_state.posture_engine_error = str(e)
elif st.session_state.posture_engine:
    start_telemetry_server(st.session_state.posture_engine.get_current_metrics)

if "current_screen" not in st.session_state:
    st.session_state.current_screen = "landing"

if "current_transcript" not in st.session_state:
    st.session_state.current_transcript = ""

if "current_audio_bytes" not in st.session_state:
    st.session_state.current_audio_bytes = None

if "current_evaluation" not in st.session_state:
    st.session_state.current_evaluation = None

if "current_speech_data" not in st.session_state:
    st.session_state.current_speech_data = None

if "current_pronunciation" not in st.session_state:
    st.session_state.current_pronunciation = None

if "final_report" not in st.session_state:
    st.session_state.final_report = None


# =============================================================================
# THREAD-SAFE ENGINE BRIDGE FOR WEBRTC WORKERS
# =============================================================================

class AppEngineBridge:
    """Thread-safe bridge sharing engine and UI state with WebRTC worker threads."""
    engine: Optional[PostureEngine] = None
    mirror: bool = False
    mode: str = "calibration"


if WEBRTC_AVAILABLE:
    # Optimized processing width: 540px gives crisp landmark tracking at minimal CPU load
    _MAX_PROCESS_W = 540

    class StreamlitPostureProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")

                # ── Fast downscale to processing resolution if needed ──
                h, w = img.shape[:2]
                if w > _MAX_PROCESS_W:
                    scale = _MAX_PROCESS_W / w
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

                engine = AppEngineBridge.engine
                if engine and engine.is_running:
                    mirror = AppEngineBridge.mirror
                    mode = AppEngineBridge.mode

                    if mode == "calibration" and not engine.is_calibrated:
                        annotated, _, _ = engine.calibrate_frame(img, mirror=mirror)
                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                    else:
                        annotated, _ = engine.process_frame(img, mirror=mirror, mesh_only=True)
                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception as e:
                import traceback
                print(f"[WebRTC Video Error] {e}")
                traceback.print_exc()
            return frame


# Synchronize Bridge with Streamlit Session State
AppEngineBridge.engine = st.session_state.get("posture_engine")
AppEngineBridge.mirror = st.session_state.get("camera_mirror", False)
AppEngineBridge.mode = "calibration" if st.session_state.get("current_screen") == "calibration" else "interview"


# =============================================================================
# NAVIGATION HELPER
# =============================================================================

def go_to_screen(screen_name: str):
    st.session_state.current_screen = screen_name
    st.rerun()


# =============================================================================
# SIDEBAR (TECHNICAL MINIMALIST NOTHING UI)
# =============================================================================

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
    if os.path.isfile(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("<div class='nothing-title'>HIRESENSE</div>", unsafe_allow_html=True)
        st.markdown("<div class='nothing-label'>// MULTIMODAL INTERVIEW OS</div>", unsafe_allow_html=True)

    st.markdown("<div class='nothing-label'>[ SYSTEM TELEMETRY ]</div>", unsafe_allow_html=True)

    # 1. Ollama status
    is_ollama_ok, ollama_msg, available_models = st.session_state.ollama_client.check_connection()
    if is_ollama_ok:
        st.markdown(f"<span class='nothing-badge nothing-badge-green'><span class='nothing-live-dot'></span>OLLAMA ONLINE // {len(available_models)} MODELS</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='nothing-badge nothing-badge-amber'>OLLAMA OFFLINE // HEURISTICS ACTIVE</span>", unsafe_allow_html=True)

    # 2. Vision Models status
    has_vision = bool(st.session_state.posture_engine and st.session_state.posture_engine.is_running)
    if has_vision:
        st.markdown("<span class='nothing-badge nothing-badge-green'><span class='nothing-live-dot'></span>VISION ENGINE // GREEN MESH READY</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='nothing-badge nothing-badge-red'>VISION OFFLINE</span>", unsafe_allow_html=True)

    # 3. Audio & FFmpeg status
    has_ffmpeg = is_ffmpeg_available()
    if has_ffmpeg:
        st.markdown("<span class='nothing-badge nothing-badge-green'>FFMPEG AUDIO CONVERTER // OK</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='nothing-badge nothing-badge-amber'>FFMPEG MISSING // WAV ONLY</span>", unsafe_allow_html=True)

    st.divider()

    # Camera Settings (Unmirrored by default)
    st.markdown("<div class='nothing-label'>[ CAMERA ORIENTATION ]</div>", unsafe_allow_html=True)
    m_val = st.toggle(
        "Mirror Camera View",
        value=bool(st.session_state.get("camera_mirror", False)),
        key="camera_mirror_sidebar_toggle",
        help="Toggle camera mirroring. ON = Selfie mirror view. OFF = Natural camera view.",
    )
    if m_val != st.session_state.get("camera_mirror", False):
        st.session_state.camera_mirror = m_val
        AppEngineBridge.mirror = m_val
        st.rerun()
    AppEngineBridge.mirror = bool(st.session_state.get("camera_mirror", False))

    st.divider()

    # Navigation Controls
    st.markdown("<div class='nothing-label'>[ SYSTEM NAVIGATION ]</div>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("01 :: HOME", use_container_width=True):
            go_to_screen("landing")
    with col_nav2:
        if st.button("02 :: SETUP", use_container_width=True):
            go_to_screen("config")

    if st.session_state.controller.session_manager.session_data.get("is_active"):
        if st.button("TERMINATE SESSION [END]", use_container_width=True, type="secondary"):
            v_summary = st.session_state.posture_engine.get_summary() if st.session_state.posture_engine else {}
            s_summary = st.session_state.controller.speech_aggregator.get_summary()
            st.session_state.controller.session_manager.finish_session(
                speech_summary=s_summary, visual_summary=v_summary
            )
            report = ReportGenerator.generate_report(
                st.session_state.controller.session_manager.get_session_data(),
                speech_summary=s_summary,
                visual_summary=v_summary,
            )
            st.session_state.final_report = report
            go_to_screen("report")

    # Scoring Profile Selector
    st.divider()
    st.markdown("<div class='nothing-label'>[ SCORING WEIGHT PROFILE ]</div>", unsafe_allow_html=True)
    profile_choice = st.selectbox(
        "Active Profile",
        options=["code_default", "readme_balanced"],
        format_func=lambda x: "01 :: Gesture-Focused (Default)" if x == "code_default" else "02 :: Balanced Profile",
        label_visibility="collapsed",
    )
    if profile_choice != ACTIVE_SCORING_PROFILE:
        set_scoring_profile(profile_choice)
        st.caption(f"// Active: {profile_choice}")


# =============================================================================
# SCREEN 1: LANDING PAGE (NOTHING UI)
# =============================================================================

if st.session_state.current_screen == "landing":
    st.markdown(
        f"""
        <div class="nothing-card">
            <div class="nothing-label">// HIRESENSE OS :: INTERVIEW ENGINE</div>
            <div class="nothing-title">{APP_NAME}</div>
            <div class="nothing-subtitle">{APP_TAGLINE}</div>
            <p style="color: var(--beige-primary); font-size: 0.95rem; line-height: 1.6; max-width: 900px;">
                An intelligent virtual interview system designed for technical and behavioural preparation. 
                Evaluates three independent dimensions: <strong>Answer Intelligence</strong> (Local LLM), 
                <strong>Speech Intelligence</strong> (Vocal metrics), and <strong>Visual Behaviour</strong> (MediaPipe Pose, Face Mesh & Hand Gestures).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="nothing-card" style="min-height: 220px;">
                <div class="nothing-card-header">
                    <span>[ 01 ] ANSWER INTELLIGENCE</span>
                    <span class="nothing-badge">OLLAMA</span>
                </div>
                <p style="color: var(--beige-muted); font-size: 0.88rem; line-height: 1.5;">
                    Local LLM reasoning assessing technical correctness, depth of explanation, conceptual completeness, 
                    and grammatical precision using Qwen 3.5.
                </p>
                <div style="margin-top: 1rem;">
                    <span class="nothing-badge nothing-badge-green">LOCAL INFERENCE</span>
                    <span class="nothing-badge">DETERMINISTIC</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="nothing-card" style="min-height: 220px;">
                <div class="nothing-card-header">
                    <span>[ 02 ] SPEECH INTELLIGENCE</span>
                    <span class="nothing-badge">AUDIO</span>
                </div>
                <p style="color: var(--beige-muted); font-size: 0.88rem; line-height: 1.5;">
                    Oral response capture, Google Speech-to-Text transcription with user correction, 
                    deterministic filler-word tracking across 10 patterns, and pronunciation feedback.
                </p>
                <div style="margin-top: 1rem;">
                    <span class="nothing-badge nothing-badge-green">SPEECH RECOGNITION</span>
                    <span class="nothing-badge">FILLER RATE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="nothing-card" style="min-height: 220px;">
                <div class="nothing-card-header">
                    <span>[ 03 ] VISUAL BEHAVIOUR</span>
                    <span class="nothing-badge nothing-badge-green">LIVE MESH</span>
                </div>
                <p style="color: var(--beige-muted); font-size: 0.88rem; line-height: 1.5;">
                    Real-time unmirrored MediaPipe vision tracking spine alignment, solvePnP 3D head pose, 
                    iris gaze consistency, fidgeting variance, and upper-body stability with neon green wireframes.
                </p>
                <div style="margin-top: 1rem;">
                    <span class="nothing-badge nothing-badge-green">GREEN WIREFRAME</span>
                    <span class="nothing-badge">SOLVEPNP</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2, _ = st.columns([1.8, 2.0, 3.5])
    with c_btn1:
        if st.button("INITIALIZE SESSION", type="primary", use_container_width=True):
            go_to_screen("config")
    with c_btn2:
        if st.button("DIAGNOSTIC TELEMETRY", use_container_width=True):
            st.info(
                f"Ollama Endpoint: {st.session_state.ollama_client.chat_endpoint} | "
                f"Models: {available_models} | "
                f"Camera Mirroring: {st.session_state.camera_mirror} | "
                f"WebRTC Engine: {WEBRTC_AVAILABLE}"
            )


# =============================================================================
# SCREEN 2: INTERVIEW CONFIGURATION (NOTHING UI)
# =============================================================================

elif st.session_state.current_screen == "config":
    st.markdown("<div class='nothing-label'>// STEP 01 :: SESSION CONFIGURATION</div>", unsafe_allow_html=True)
    st.markdown("<div class='nothing-title'>INTERVIEW PARAMETERS</div>", unsafe_allow_html=True)

    with st.form("interview_config_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='nothing-label'>[ CANDIDATE IDENTIFIER ]</div>", unsafe_allow_html=True)
            candidate_name = st.text_input("Candidate Name", value="Candidate", max_chars=40, label_visibility="collapsed")

            st.markdown("<div class='nothing-label'>[ INTERVIEW CATEGORY ]</div>", unsafe_allow_html=True)
            interview_type = st.selectbox("Interview Category", ["Technical", "HR", "Mixed"], label_visibility="collapsed")

            st.markdown("<div class='nothing-label'>[ SUBJECT FOCUS ]</div>", unsafe_allow_html=True)
            available_subjects = st.session_state.question_manager.get_available_subjects(interview_type)
            subject = st.selectbox("Subject Focus", available_subjects, label_visibility="collapsed")

            st.markdown("<div class='nothing-label'>[ DIFFICULTY TIER ]</div>", unsafe_allow_html=True)
            difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], label_visibility="collapsed")

            st.markdown("<div class='nothing-label'>[ QUESTION COUNT ]</div>", unsafe_allow_html=True)
            num_questions = st.slider("Number of Questions", min_value=1, max_value=15, value=5, label_visibility="collapsed")

        with col2:
            st.markdown("<div class='nothing-label'>[ MODALITY SELECTION ]</div>", unsafe_allow_html=True)
            answer_mode = st.radio("Response Mode", ["Written", "Oral"], help="Written uses code/text area; Oral captures microphone audio.", label_visibility="collapsed")

            st.markdown("<div class='nothing-label'>[ BIOMECHANICAL VISION ENGINE ]</div>", unsafe_allow_html=True)
            enable_webcam = st.checkbox(
                "Enable Live MediaPipe Visual Analysis",
                value=True,
                help="Tracks sitting posture, head pose, eye gaze, and hand gestures with neon green wireframes.",
            )
            adaptive_followups = st.checkbox(
                "Enable Adaptive Follow-Up Questioning",
                value=False,
                help="AI interviewer generates dynamic clarifying or challenging questions based on previous answers.",
            )

            st.markdown("<div class='nothing-label'>[ OLLAMA LLM MODEL ]</div>", unsafe_allow_html=True)
            selected_model = st.text_input(
                "Ollama Model Name",
                value=DEFAULT_OLLAMA_MODEL,
                help="Local Ollama model name (e.g. qwen3.5:4b, llama3, mistral).",
                label_visibility="collapsed",
            )

        submit_setup = st.form_submit_button("LOCK PARAMETERS & PROCEED", type="primary", use_container_width=True)

    if submit_setup:
        st.session_state.controller.setup_interview(
            candidate_name=candidate_name,
            category=interview_type,
            subject=subject,
            difficulty=difficulty,
            answer_mode=answer_mode,
            adaptive_followups=adaptive_followups,
            webcam_enabled=enable_webcam,
            num_questions=num_questions,
        )
        st.session_state.selected_model = selected_model
        st.session_state.current_evaluation = None
        st.session_state.current_transcript = ""
        st.session_state.current_audio_bytes = None

        if enable_webcam and st.session_state.posture_engine:
            st.session_state.posture_engine.reset_calibration()
            go_to_screen("calibration")
        else:
            go_to_screen("interview")


# =============================================================================
# SCREEN 3: WEBCAM CALIBRATION (NOTHING UI)
# =============================================================================

elif st.session_state.current_screen == "calibration":
    st.markdown("<div class='nothing-label'>// STEP 02 :: BASELINE CALIBRATION</div>", unsafe_allow_html=True)
    st.markdown("<div class='nothing-title'>OPTICAL BASELINE CAPTURE</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style="color: var(--beige-muted); font-size: 0.90rem;">
            Sit comfortably and face your camera directly. 
            Calibration captures your individual baseline shoulder width, eye position, and head alignment 
            in <strong>4 seconds</strong> without mirroring.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        c_hdr1, c_hdr2 = st.columns([3, 2])
        with c_hdr1:
            st.markdown(
                f"<div class='nothing-label'><span class='nothing-live-dot'></span>WEBCAM FEED ({'MIRRORED' if st.session_state.get('camera_mirror') else 'NATURAL'})</div>",
                unsafe_allow_html=True,
            )
        with c_hdr2:
            cal_m = st.toggle("Mirror View", value=bool(st.session_state.get("camera_mirror", False)), key="cal_mirror_toggle")
            if cal_m != st.session_state.get("camera_mirror", False):
                st.session_state.camera_mirror = cal_m
                AppEngineBridge.mirror = cal_m
                st.rerun()

        if WEBRTC_AVAILABLE:
            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            webrtc_streamer(
                key="calibration_stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=StreamlitPostureProcessor,
                media_stream_constraints={"video": {"width": {"ideal": 540, "max": 640}, "height": {"ideal": 405, "max": 480}, "frameRate": {"ideal": 20, "max": 24}}, "audio": False},
                async_processing=True,
            )
        else:
            st.warning("// WebRTC browser video unavailable. Proceeding in headless mode.")

    with col_info:
        st.markdown(
            """
            <div class="nothing-card">
                <div class="nothing-card-header">[ CALIBRATION PROTOCOL ]</div>
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--beige-muted); line-height: 1.8;">
                    01. Position upper torso & shoulders in frame.<br>
                    02. Maintain neutral spine alignment.<br>
                    03. Look directly into camera lens.<br>
                    04. Green wireframe lines verify detection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Live dynamic calibration meter (smooth sub-second updates from telemetry daemon)
        cal_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                background: transparent;
                font-family: 'Space Mono', 'Consolas', 'Courier New', monospace;
                color: #E8DEC8;
                overflow: hidden;
            }
            .cal-card {
                background-color: #121214;
                border: 1px solid #262420;
                border-radius: 4px;
                padding: 0.85rem 1rem;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
            .cal-label-row {
                display: flex;
                justify-content: space-between;
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                margin-bottom: 0.4rem;
            }
            .cal-status {
                color: #E5A93C;
                font-weight: 700;
            }
            .cal-track {
                width: 100%;
                height: 6px;
                background: #1C1B18;
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 0.4rem;
            }
            .cal-fill {
                height: 100%;
                background: #00FF88;
                border-radius: 3px;
                transition: width 0.2s ease-out;
            }
            .cal-sub {
                font-size: 0.63rem;
                color: #A69B88;
                letter-spacing: 0.04em;
            }
        </style>
        </head>
        <body>
        <div class="cal-card">
            <div class="cal-label-row">
                <span>PROGRESS</span>
                <span id="cal-status" class="cal-status">CONNECTING...</span>
            </div>
            <div class="cal-track">
                <div id="cal-bar" class="cal-fill" style="width: 0%;"></div>
            </div>
            <div id="cal-sub" class="cal-sub">POSITION IN FRAME TO BEGIN BASELINE ACQUISITION</div>
        </div>

        <script>
        let isPolling = false;
        async function pollCal() {
            if (isPolling) return;
            isPolling = true;
            try {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), 250);
                const res = await fetch("http://127.0.0.1:8502/telemetry", { cache: "no-store", signal: controller.signal });
                clearTimeout(timer);
                if (!res.ok) return;
                const d = await res.json();
                const prog = Math.min(100, Math.max(0, d.calibration_progress || 0));
                document.getElementById("cal-bar").style.width = prog + "%";
                if (d.is_calibrated || prog >= 100) {
                    document.getElementById("cal-status").innerText = "100% // READY";
                    document.getElementById("cal-status").style.color = "#00FF88";
                    document.getElementById("cal-sub").innerText = "BASELINE ACQUIRED. READY TO PROCEED.";
                } else if (prog > 0) {
                    document.getElementById("cal-status").innerText = Math.round(prog) + "% // CALIBRATING";
                    document.getElementById("cal-status").style.color = "#00FF88";
                    document.getElementById("cal-sub").innerText = "HOLD NEUTRAL POSTURE & LOOK AT LENS";
                } else {
                    document.getElementById("cal-status").innerText = "0% // DETECTING";
                    document.getElementById("cal-status").style.color = "#E5A93C";
                    document.getElementById("cal-sub").innerText = "GREEN MESH ON FACE VERIFIES DETECTION";
                }
            } catch(e) {} finally {
                isPolling = false;
            }
        }
        setInterval(pollCal, 250);
        pollCal();
        </script>
        </body>
        </html>
        """
        components.html(cal_html, height=85)

        metrics = st.session_state.posture_engine.get_current_metrics() if st.session_state.posture_engine else {}
        is_cal = metrics.get("is_calibrated", False)
        cal_prog = metrics.get("calibration_progress", 0.0)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("CONFIRM CALIBRATION", type="primary", use_container_width=True):
                if st.session_state.posture_engine:
                    st.session_state.posture_engine.is_calibrated = True
                go_to_screen("interview")
        with c2:
            if st.button("SKIP [MANUAL]", use_container_width=True):
                if st.session_state.posture_engine:
                    st.session_state.posture_engine.is_calibrated = True
                go_to_screen("interview")


# =============================================================================
# SCREEN 4: LIVE INTERVIEW (LIVE POSTURE TRACKING & GREEN MESH)
# =============================================================================

elif st.session_state.current_screen == "interview":
    curr_q = st.session_state.controller.session_manager.get_current_question()
    if not curr_q:
        # Session complete
        v_summary = st.session_state.posture_engine.get_summary() if st.session_state.posture_engine else {}
        s_summary = st.session_state.controller.speech_aggregator.get_summary()
        st.session_state.controller.session_manager.finish_session(
            speech_summary=s_summary, visual_summary=v_summary
        )
        report = ReportGenerator.generate_report(
            st.session_state.controller.session_manager.get_session_data(),
            speech_summary=s_summary,
            visual_summary=v_summary,
        )
        st.session_state.final_report = report
        go_to_screen("report")

    session_data = st.session_state.controller.session_manager.get_session_data()
    q_idx = session_data.get("current_index", 0)
    total_q = len(session_data.get("questions", []))
    settings = session_data.get("settings", {})
    webcam_enabled = settings.get("webcam_enabled", True)

    # ── TOP STATUS BAR ──
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem;">
                <span class="nothing-badge">[ SEQUENCE : {q_idx + 1:02d} / {total_q:02d} ]</span>
                <span class="nothing-badge nothing-badge-green">{curr_q.get('subject', 'TECHNICAL').upper()}</span>
                <span class="nothing-badge nothing-badge-amber">{curr_q.get('difficulty', 'EASY').upper()}</span>
                <span class="nothing-badge"><span class="nothing-live-dot"></span>LIVE SESSION ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h_col2:
        st.progress((q_idx + 1) / total_q)

    # ── TWO COLUMN LAYOUT: INTERVIEWER (LEFT) vs LIVE WEBCAM & TELEMETRY (RIGHT) ──
    col_main, col_vision = st.columns([3, 2] if webcam_enabled else [1, 0.01])

    # ── LEFT: INTERVIEWER & ANSWER INPUT ──
    with col_main:
        st.markdown(
            f"""
            <div class="nothing-question-box">
                <div class="nothing-label">// INTERVIEWER PROMPT :: QUESTION {q_idx + 1:02d}</div>
                <div class="nothing-question-text">{curr_q.get('question')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user_answer_text = ""
        speech_result_data = None

        if settings.get("answer_mode") == "Written":
            st.markdown("<div class='nothing-label'>[ ANSWER INPUT : WRITTEN TERMINAL ]</div>", unsafe_allow_html=True)
            user_answer_text = st.text_area(
                "Written Answer",
                height=180,
                key=f"written_ans_{q_idx}",
                placeholder="Type your structured explanation here...",
                label_visibility="collapsed",
            )
        else: # Oral Mode
            st.markdown("<div class='nothing-label'>[ AUDIO CAPTURE : ORAL RESPONSE ]</div>", unsafe_allow_html=True)
            audio_recording = st.audio_input("Microphone Input", key=f"oral_input_{q_idx}", label_visibility="collapsed")

            if audio_recording is not None:
                st.session_state.current_audio_bytes = audio_recording.getvalue()
                if st.button("TRANSCRIBE AUDIO // GOOGLE STT", type="secondary"):
                    with st.spinner("Decoding audio..."):
                        transcript, err = transcribe_audio(st.session_state.current_audio_bytes)
                        if err:
                            st.error(err)
                        else:
                            st.session_state.current_transcript = transcript
                            st.session_state.current_evaluation = None

            if st.session_state.current_transcript:
                st.markdown("<div class='nothing-label'>[ RECOGNIZED TRANSCRIPT // EDITABLE ]</div>", unsafe_allow_html=True)
                edited_transcript = st.text_area(
                    "Transcript Edit",
                    value=st.session_state.current_transcript,
                    height=100,
                    key=f"transcript_edit_{q_idx}",
                    label_visibility="collapsed",
                )
                if edited_transcript != st.session_state.current_transcript:
                    st.session_state.current_transcript = edited_transcript
                    st.session_state.current_evaluation = None

                user_answer_text = st.session_state.current_transcript

        # Submit Answer Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("SUBMIT & EVALUATE ANSWER", type="primary", use_container_width=True):
            if not user_answer_text.strip():
                st.warning("// Input required: Please type or record an answer before submission.")
            else:
                with st.spinner("// Ollama LLM analyzing response concepts..."):
                    # Speech filler analysis
                    if settings.get("answer_mode") == "Oral" and user_answer_text.strip():
                        filler_res = analyze_fillers(user_answer_text)
                        speech_result_data = {
                            "filler_analysis": filler_res,
                            "pronunciation": st.session_state.current_pronunciation,
                        }

                    # Capture current visual snapshot
                    v_snapshot = None
                    if st.session_state.posture_engine:
                        v_snapshot = st.session_state.posture_engine.get_current_metrics()

                    # Submit to controller
                    model_name = getattr(st.session_state, "selected_model", DEFAULT_OLLAMA_MODEL)
                    eval_res = st.session_state.controller.submit_answer(
                        user_answer=user_answer_text,
                        speech_data=speech_result_data,
                        visual_snapshot=v_snapshot,
                        model_name=model_name,
                    )
                    st.session_state.current_evaluation = eval_res
                    st.session_state.current_speech_data = speech_result_data

        # ── INLINE EVALUATION (SCREEN 5) ──
        if st.session_state.current_evaluation:
            ev = st.session_state.current_evaluation
            st.divider()
            st.markdown("<div class='nothing-label'>// EVALUATION REPORT :: QUESTION ANALYSIS</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='nothing-card'><div class='nothing-label'>CORRECTNESS</div><div class='nothing-metric-val'>{ev.get('correctness', 0):g}<span class='nothing-metric-unit'>/10</span></div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='nothing-card'><div class='nothing-label'>DEPTH</div><div class='nothing-metric-val'>{ev.get('depth', 0):g}<span class='nothing-metric-unit'>/10</span></div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='nothing-card'><div class='nothing-label'>COMPLETENESS</div><div class='nothing-metric-val'>{ev.get('completeness', 0):g}<span class='nothing-metric-unit'>/10</span></div></div>", unsafe_allow_html=True)
            avg_c = (ev.get("correctness", 0) + ev.get("depth", 0) + ev.get("completeness", 0)) / 3.0
            c4.markdown(f"<div class='nothing-card'><div class='nothing-label'>CONTENT SCORE</div><div class='nothing-metric-val'>{avg_c:.1f}<span class='nothing-metric-unit'>/10</span></div></div>", unsafe_allow_html=True)

            st.markdown(f"**Explanation:** {ev.get('explanation')}")
            if ev.get("grammar_spelling"):
                st.markdown(f"**Grammar & Structure:** {ev.get('grammar_spelling')}")
            st.markdown(f"**Overall Feedback:** {ev.get('overall_feedback')}")

            col_st, col_imp = st.columns(2)
            with col_st:
                st.markdown("<div class='nothing-label'>[ OBSERVED STRENGTHS ]</div>", unsafe_allow_html=True)
                for s in ev.get("strengths", []):
                    st.markdown(f"- [+] {s}")
            with col_imp:
                st.markdown("<div class='nothing-label'>[ AREAS FOR IMPROVEMENT ]</div>", unsafe_allow_html=True)
                for imp in ev.get("areas_to_improve", []):
                    st.markdown(f"- [!] {imp}")

            # Oral Speech Feedback
            if settings.get("answer_mode") == "Oral" and st.session_state.current_transcript:
                st.markdown("<div class='nothing-label'>// SPEECH INTELLIGENCE TELEMETRY</div>", unsafe_allow_html=True)
                fill_report = analyze_fillers(st.session_state.current_transcript)
                sp_c1, sp_c2, sp_c3 = st.columns(3)
                sp_c1.metric("Filler Words", fill_report["total"])
                sp_c2.metric("Total Words", fill_report["word_count"])
                sp_c3.metric("Fillers / 100 Words", f"{fill_report['rate_per_100_words']:.1f}")

                if fill_report["counts"]:
                    st.write("**Detected Fillers:** " + ", ".join(f"`{w}`: {n}" for w, n in fill_report["counts"].items()))

                if is_openpronounce_available():
                    if st.button("RUN PRONUNCIATION ASSESSMENT"):
                        with st.spinner("Analyzing audio phonetics..."):
                            p_res, p_err = analyze_pronunciation(
                                st.session_state.current_audio_bytes,
                                st.session_state.current_transcript
                            )
                            if p_err:
                                st.error(p_err)
                            else:
                                st.session_state.current_pronunciation = p_res
                                st.success(f"Pronunciation Score: {p_res.get('score', 'N/A')}/100")

            with st.expander("// VIEW EXPECTED REFERENCE ANSWER"):
                st.write(ev.get("reference_answer") or curr_q.get("reference_answer"))

            st.markdown("<br>", unsafe_allow_html=True)
            if q_idx + 1 < total_q:
                if st.button("ADVANCE TO NEXT QUESTION", type="primary", use_container_width=True):
                    st.session_state.controller.session_manager.advance_question()
                    st.session_state.current_evaluation = None
                    st.session_state.current_transcript = ""
                    st.session_state.current_audio_bytes = None
                    st.session_state.current_pronunciation = None
                    st.rerun()
            else:
                if st.button("COMPLETE INTERVIEW & GENERATE DOSSIER", type="primary", use_container_width=True):
                    v_summary = st.session_state.posture_engine.get_summary() if st.session_state.posture_engine else {}
                    s_summary = st.session_state.controller.speech_aggregator.get_summary()
                    st.session_state.controller.session_manager.finish_session(
                        speech_summary=s_summary, visual_summary=v_summary
                    )
                    report = ReportGenerator.generate_report(
                        st.session_state.controller.session_manager.get_session_data(),
                        speech_summary=s_summary,
                        visual_summary=v_summary,
                    )
                    st.session_state.final_report = report
                    go_to_screen("report")

    # ── RIGHT: LIVE UNMIRRORED WEBCAM STREAM & GREEN MESH ──
    if webcam_enabled:
        with col_vision:
            iv_hdr1, iv_hdr2 = st.columns([3, 2])
            with iv_hdr1:
                st.markdown(
                    f"""
                    <div class="nothing-label">
                        <span class="nothing-live-dot"></span>BIOMECHANICAL ({'MIRRORED' if st.session_state.get('camera_mirror') else 'NATURAL'})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with iv_hdr2:
                iv_m = st.toggle("Mirror View", value=bool(st.session_state.get("camera_mirror", False)), key="live_mirror_toggle")
                if iv_m != st.session_state.get("camera_mirror", False):
                    st.session_state.camera_mirror = iv_m
                    AppEngineBridge.mirror = iv_m
                    st.rerun()

            # WebRTC Camera Stream (Stable key so it remains live throughout question answering)
            if WEBRTC_AVAILABLE:
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                webrtc_streamer(
                    key="live_interview_camera",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_config,
                    video_processor_factory=StreamlitPostureProcessor,
                    media_stream_constraints={"video": {"width": {"ideal": 540, "max": 640}, "height": {"ideal": 405, "max": 480}, "frameRate": {"ideal": 20, "max": 24}}, "audio": False},
                    async_processing=True,
                )
            else:
                st.info("// WebRTC camera stream is running in headless mode.")

            # Live Biomechanical Telemetry Strip (Real-time live updates via zero-latency daemon)
            if st.session_state.posture_engine:
                telemetry_html = """
                <!DOCTYPE html>
                <html>
                <head>
                <meta charset="utf-8">
                <style>
                    * { box-sizing: border-box; margin: 0; padding: 0; }
                    body {
                        background: transparent;
                        font-family: 'Space Mono', 'Consolas', 'Courier New', monospace;
                        color: #E8DEC8;
                        overflow: hidden;
                    }
                    .telemetry-card {
                        background-color: #121214;
                        border: 1px solid #262420;
                        border-radius: 4px;
                        padding: 1rem 1.15rem;
                    }
                    .card-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        border-bottom: 1px solid #22201C;
                        padding-bottom: 0.5rem;
                        margin-bottom: 0.75rem;
                        font-size: 0.72rem;
                        letter-spacing: 0.12em;
                        font-weight: 700;
                    }
                    .badge-live {
                        display: inline-flex;
                        align-items: center;
                        gap: 0.35rem;
                        padding: 0.2rem 0.5rem;
                        border-radius: 2px;
                        font-size: 0.65rem;
                        letter-spacing: 0.1em;
                        border: 1px solid rgba(0, 255, 136, 0.3);
                        color: #00FF88;
                        background: rgba(0, 255, 136, 0.08);
                    }
                    .live-dot {
                        width: 6px;
                        height: 6px;
                        border-radius: 50%;
                        background-color: #00FF88;
                        box-shadow: 0 0 6px #00FF88;
                        animation: pulse 1.4s infinite;
                    }
                    @keyframes pulse {
                        0% { opacity: 0.4; }
                        50% { opacity: 1; }
                        100% { opacity: 0.4; }
                    }
                    .metric-row {
                        margin-bottom: 0.6rem;
                    }
                    .metric-label-val {
                        display: flex;
                        justify-content: space-between;
                        font-size: 0.72rem;
                        color: #A69B88;
                        letter-spacing: 0.06em;
                        margin-bottom: 0.25rem;
                    }
                    .metric-val {
                        color: #E8DEC8;
                        font-weight: 700;
                    }
                    .bar-track {
                        width: 100%;
                        height: 4px;
                        background: #1C1B18;
                        border-radius: 2px;
                        overflow: hidden;
                    }
                    .bar-fill {
                        height: 100%;
                        background: #00FF88;
                        border-radius: 2px;
                        transition: width 0.22s ease-out;
                    }
                    .disclaimer {
                        margin-top: 0.7rem;
                        padding-top: 0.5rem;
                        border-top: 1px solid #22201C;
                        font-size: 0.62rem;
                        color: #6E685E;
                        line-height: 1.4;
                        letter-spacing: 0.03em;
                    }
                </style>
                </head>
                <body>
                <div class="telemetry-card">
                    <div class="card-header">
                        <span>[ OPTICAL TELEMETRY ]</span>
                        <span class="badge-live"><span class="live-dot"></span>GREEN MESH ACTIVE</span>
                    </div>

                    <div class="metric-row">
                        <div class="metric-label-val">
                            <span>POSTURE ALIGNMENT</span>
                            <span id="val-posture" class="metric-val">-- / 100</span>
                        </div>
                        <div class="bar-track"><div id="bar-posture" class="bar-fill" style="width: 75%;"></div></div>
                    </div>

                    <div class="metric-row">
                        <div class="metric-label-val">
                            <span>EYE CONTACT INDEX</span>
                            <span id="val-gaze" class="metric-val">-- / 100 (CENTERED)</span>
                        </div>
                        <div class="bar-track"><div id="bar-gaze" class="bar-fill" style="width: 85%;"></div></div>
                    </div>

                    <div class="metric-row">
                        <div class="metric-label-val">
                            <span>HEAD POSITION</span>
                            <span id="val-head" class="metric-val">-- / 100</span>
                        </div>
                        <div class="bar-track"><div id="bar-head" class="bar-fill" style="width: 90%;"></div></div>
                    </div>

                    <div class="metric-row">
                        <div class="metric-label-val">
                            <span>GESTURE COMPOSURE</span>
                            <span id="val-gesture" class="metric-val">-- / 100</span>
                        </div>
                        <div class="bar-track"><div id="bar-gesture" class="bar-fill" style="width: 85%;"></div></div>
                    </div>

                    <div class="metric-row">
                        <div class="metric-label-val">
                            <span>RELIABILITY CONFIDENCE</span>
                            <span id="val-rel" class="metric-val">--%</span>
                        </div>
                        <div class="bar-track"><div id="bar-rel" class="bar-fill" style="width: 100%;"></div></div>
                    </div>

                    <div class="disclaimer">
                        NOTICE: Biomechanical indicators are coaching guides for interview posture alignment and do not diagnose competence.
                    </div>
                </div>

                <script>
                let isFetching = false;
                async function fetchTelemetry() {
                    if (isFetching) return;
                    isFetching = true;
                    try {
                        const controller = new AbortController();
                        const timer = setTimeout(() => controller.abort(), 320);
                        const res = await fetch("http://127.0.0.1:8502/telemetry", { cache: "no-store", signal: controller.signal });
                        clearTimeout(timer);
                        if (!res.ok) return;
                        const d = await res.json();
                        
                        document.getElementById("val-posture").innerText = d.posture + " / 100";
                        document.getElementById("val-gaze").innerText = d.eye_contact + " / 100 (" + d.gaze_direction + ")";
                        document.getElementById("val-head").innerText = d.head_position + " / 100";
                        document.getElementById("val-gesture").innerText = d.gesture_score + " / 100";
                        document.getElementById("val-rel").innerText = d.reliability + "%";

                        document.getElementById("bar-posture").style.width = Math.max(0, Math.min(100, d.posture)) + "%";
                        document.getElementById("bar-gaze").style.width = Math.max(0, Math.min(100, d.eye_contact)) + "%";
                        document.getElementById("bar-head").style.width = Math.max(0, Math.min(100, d.head_position)) + "%";
                        document.getElementById("bar-gesture").style.width = Math.max(0, Math.min(100, d.gesture_score)) + "%";
                        document.getElementById("bar-rel").style.width = Math.max(0, Math.min(100, d.reliability)) + "%";
                    } catch(e) {} finally {
                        isFetching = false;
                    }
                }
                setInterval(fetchTelemetry, 350);
                fetchTelemetry();
                </script>
                </body>
                </html>
                """
                components.html(telemetry_html, height=270)


# =============================================================================
# SCREEN 6: COMPREHENSIVE FINAL REPORT & EXPORT (NOTHING UI)
# =============================================================================

elif st.session_state.current_screen == "report":
    if not st.session_state.final_report:
        v_summary = st.session_state.posture_engine.get_summary() if st.session_state.posture_engine else {}
        s_summary = st.session_state.controller.speech_aggregator.get_summary()
        report = ReportGenerator.generate_report(
            st.session_state.controller.session_manager.get_session_data(),
            speech_summary=s_summary,
            visual_summary=v_summary,
        )
        st.session_state.final_report = report

    report = st.session_state.final_report
    ov = report["overview"]
    ans_intel = report["answer_intelligence"]
    sp_intel = report["speech_intelligence"]
    vis_intel = report["visual_intelligence"]
    recs = report["recommendations"]

    st.markdown(
        f"""
        <div class="nothing-card">
            <div class="nothing-label">// HIRESENSE DOSSIER :: PERFORMANCE SYNTHESIS</div>
            <div class="nothing-title">INTERVIEW PERFORMANCE DOSSIER</div>
            <div style="font-family: var(--font-mono); font-size: 0.80rem; color: var(--beige-muted); margin-top: 0.5rem;">
                CANDIDATE: <span style="color: var(--beige-light);">{ov['candidate_name'].upper()}</span> | 
                SESSION ID: <span style="color: var(--beige-light);">{ov['session_id']}</span> | 
                DURATION: <span style="color: var(--beige-light);">{ov['duration_formatted']}</span> | 
                TIMESTAMP: <span style="color: var(--beige-light);">{ov['timestamp']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── THREE SCORE PILLARS DASHBOARD ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="nothing-card">
                <div class="nothing-card-header">
                    <span>[ 01 ] ANSWER INTELLIGENCE</span>
                    <span class="nothing-badge">OLLAMA</span>
                </div>
                <div class="nothing-metric-val">{ans_intel['overall_content_score']:.1f}<span class="nothing-metric-unit"> / 10</span></div>
                <div class="nothing-label" style="margin-top: 0.25rem;">OVERALL CONTENT SCORE</div>
                <hr style="border-color: var(--border-subtle); margin: 0.75rem 0;">
                <div style="font-family: var(--font-mono); font-size: 0.80rem; color: var(--beige-muted); line-height: 1.8;">
                    <div>CORRECTNESS: <strong style="color: var(--beige-light);">{ans_intel['average_correctness']:.1f}/10</strong></div>
                    <div>DEPTH: <strong style="color: var(--beige-light);">{ans_intel['average_depth']:.1f}/10</strong></div>
                    <div>COMPLETENESS: <strong style="color: var(--beige-light);">{ans_intel['average_completeness']:.1f}/10</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if sp_intel["is_evaluated"]:
            st.markdown(
                f"""
                <div class="nothing-card">
                    <div class="nothing-card-header">
                        <span>[ 02 ] SPEECH INTELLIGENCE</span>
                        <span class="nothing-badge nothing-badge-green">AUDIO</span>
                    </div>
                    <div class="nothing-metric-val">{sp_intel['filler_rate_per_100']:.1f}<span class="nothing-metric-unit"> / 100w</span></div>
                    <div class="nothing-label" style="margin-top: 0.25rem;">FILLER WORD FREQUENCY</div>
                    <hr style="border-color: var(--border-subtle); margin: 0.75rem 0;">
                    <div style="font-family: var(--font-mono); font-size: 0.80rem; color: var(--beige-muted); line-height: 1.8;">
                        <div>TOTAL WORDS: <strong style="color: var(--beige-light);">{sp_intel['total_words']}</strong></div>
                        <div>DETECTED FILLERS: <strong style="color: var(--beige-light);">{sp_intel['total_fillers']}</strong></div>
                        <div>PRONUNCIATION: <strong style="color: var(--beige-light);">{sp_intel['average_pronunciation_score'] or 'N/A'}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="nothing-card">
                    <div class="nothing-card-header">
                        <span>[ 02 ] SPEECH INTELLIGENCE</span>
                        <span class="nothing-badge">INACTIVE</span>
                    </div>
                    <div class="nothing-metric-val" style="font-size: 1.4rem;">WRITTEN MODE</div>
                    <div class="nothing-label" style="margin-top: 0.25rem;">NO MICROPHONE AUDIO RECORDED</div>
                    <hr style="border-color: var(--border-subtle); margin: 0.75rem 0;">
                    <div style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--beige-dim);">
                        Oral speech analysis is activated when selecting Oral Response Mode during setup.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col3:
        if vis_intel["is_evaluated"]:
            st.markdown(
                f"""
                <div class="nothing-card">
                    <div class="nothing-card-header">
                        <span>[ 03 ] VISUAL BEHAVIOUR</span>
                        <span class="nothing-badge nothing-badge-green">GREEN MESH</span>
                    </div>
                    <div class="nothing-metric-val">{vis_intel['overall_readiness_score']:.0f}<span class="nothing-metric-unit"> / 100</span></div>
                    <div class="nothing-label" style="margin-top: 0.25rem;">READINESS INDEX</div>
                    <hr style="border-color: var(--border-subtle); margin: 0.75rem 0;">
                    <div style="font-family: var(--font-mono); font-size: 0.80rem; color: var(--beige-muted); line-height: 1.8;">
                        <div>POSTURE: <strong style="color: var(--beige-light);">{vis_intel['posture_score']:.0f}/100</strong> | GAZE: <strong style="color: var(--beige-light);">{vis_intel['eye_contact_score']:.0f}/100</strong></div>
                        <div>GESTURES: <strong style="color: var(--beige-light);">{vis_intel['gestures_score']:.0f}/100</strong> | STABILITY: <strong style="color: var(--beige-light);">{vis_intel['stability_score']:.0f}/100</strong></div>
                        <div>RELIABILITY: <strong style="color: var(--beige-light);">{vis_intel['reliability_score']:.0f}%</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="nothing-card">
                    <div class="nothing-card-header">
                        <span>[ 03 ] VISUAL BEHAVIOUR</span>
                        <span class="nothing-badge">INACTIVE</span>
                    </div>
                    <div class="nothing-metric-val" style="font-size: 1.4rem;">CAMERA OFF</div>
                    <div class="nothing-label" style="margin-top: 0.25rem;">WEBCAM ANALYSIS WAS OMITTED</div>
                    <hr style="border-color: var(--border-subtle); margin: 0.75rem 0;">
                    <div style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--beige-dim);">
                        Visual behaviour coaching is activated when webcam analysis is enabled.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── ACTIONABLE RECOMMENDATIONS ──
    st.markdown("<div class='nothing-label'>// ACTIONABLE COACHING DIRECTIVES</div>", unsafe_allow_html=True)
    for rec in recs:
        st.markdown(f"- [*] {rec}")

    # ── QUESTION BREAKDOWN ──
    st.markdown("<br><div class='nothing-label'>// QUESTION-BY-QUESTION EVALUATION BREAKDOWN</div>", unsafe_allow_html=True)
    for q in ans_intel.get("questions_evaluated", []):
        with st.expander(f"[ Q{q['question_number']:02d} ] {q['question']} — Score: {q['average_score']}/10"):
            st.markdown(f"**Candidate Answer:** {q['user_answer']}")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Correctness", f"{q['correctness']:g}/10")
            col_s2.metric("Depth", f"{q['depth']:g}/10")
            col_s3.metric("Completeness", f"{q['completeness']:g}/10")

            st.markdown(f"**Explanation:** {q['explanation']}")
            if q.get("strengths"):
                st.markdown("**Strengths:** " + ", ".join(f"`{s}`" for s in q["strengths"]))
            if q.get("areas_to_improve"):
                st.markdown("**Areas to Improve:** " + ", ".join(f"`{i}`" for i in q["areas_to_improve"]))
            st.markdown(f"**Reference Answer:** {q['reference_answer']}")

    # ── EXPORT OPTIONS ──
    st.divider()
    st.markdown("<div class='nothing-label'>[ EXPORT PERFORMANCE DOSSIER ]</div>", unsafe_allow_html=True)
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    # 1. PDF Export
    with col_exp1:
        pdf_filename = f"hiresense_report_{ov['session_id']}.pdf"
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reporting", pdf_filename)
        export_report_to_pdf(report, pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="[PDF] EXPORT DOSSIER",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

    # 2. JSON Export
    with col_exp2:
        json_str = json.dumps(report, indent=2)
        st.download_button(
            label="[JSON] EXPORT METRICS",
            data=json_str,
            file_name=f"hiresense_report_{ov['session_id']}.json",
            mime="application/json",
            use_container_width=True,
        )

    # 3. New Interview
    with col_exp3:
        if st.button("INITIALIZE NEW SESSION", use_container_width=True):
            st.session_state.controller = InterviewController()
            st.session_state.final_report = None
            go_to_screen("landing")
