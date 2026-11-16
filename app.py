"""
HireSense :: Multimodal AI-Powered Mock Interview & Behavioural Analysis Platform.
Complete UI/UX Redesign — Nothing OS / Teenage Engineering Technical Minimalist Design System.
"""

import os
import json
import time
import textwrap
from typing import Optional, Dict, Any, List
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

# Core Engines & Architecture
from core.session_manager import InterviewSessionManager
from core.question_manager import QuestionManager
from core.interview_controller import InterviewController
from core.telemetry_server import start_telemetry_server
from core.ui_components import (
    clean_html,
    render_page_header,
    render_section_title,
    render_status_pill,
    render_score_card,
    render_selection_card_html,
    render_interview_topbar,
    render_question_box,
    render_system_readiness_strip,
)

# AI & Speech Analysis
from ai_interviewer.ollama_client import OllamaClient
from speech_analysis.audio_capture import is_ffmpeg_available
from speech_analysis.speech_to_text import transcribe_audio
from speech_analysis.filler_analyzer import analyze_fillers
from speech_analysis.pronunciation_analyzer import analyze_pronunciation, is_openpronounce_available

# Computer Vision & Posture Engine
from posture_analysis.engine import PostureEngine
from posture_analysis.config import set_scoring_profile, SCORING_PROFILES, ACTIVE_SCORING_PROFILE

# Reporting & Export
from reporting.report_generator import ReportGenerator
from reporting.pdf_exporter import export_report_to_pdf
from reporting.visualizations import format_subscores_markdown, get_score_color_hex

# WebRTC Video Streaming
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

# Cache & inject Nothing OS Technical CSS
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "styles.css")
if "css_loaded" not in st.session_state:
    if os.path.isfile(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.session_state._css_content = f.read()
    else:
        st.session_state._css_content = ""
    st.session_state.css_loaded = True

if st.session_state.get("_css_content"):
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
    st.session_state.current_screen = "welcome"

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

if "session_history" not in st.session_state:
    st.session_state.session_history = []

if "mic_test_status" not in st.session_state:
    st.session_state.mic_test_status = None


# =============================================================================
# THREAD-SAFE ENGINE BRIDGE FOR WEBRTC WORKERS
# =============================================================================

class AppEngineBridge:
    """Thread-safe bridge sharing engine and UI state with WebRTC worker threads."""
    engine: Optional[PostureEngine] = None
    mirror: bool = False
    mode: str = "calibration"


if WEBRTC_AVAILABLE:
    # 540px width processing clamp: minimizes latency on high-res webcams
    _MAX_PROCESS_W = 540

    class StreamlitPostureProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")

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
AppEngineBridge.mirror = bool(st.session_state.get("camera_mirror", False))
AppEngineBridge.mode = "calibration" if st.session_state.get("current_screen") == "calibration" else "interview"


# =============================================================================
# NAVIGATION HELPER
# =============================================================================

def go_to_screen(screen_name: str) -> None:
    """Transitions between the 5 primary application screens."""
    # Screen aliases for backwards compatibility
    alias_map = {
        "landing": "welcome",
        "config": "setup",
        "report": "results",
    }
    st.session_state.current_screen = alias_map.get(screen_name, screen_name)
    st.rerun()


# =============================================================================
# SIDEBAR (TECHNICAL MINIMALIST NOTHING UI)
# =============================================================================

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
    if os.path.isfile(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("<div class='nothing-title' style='font-size: 1.4rem;'>HIRESENSE</div>", unsafe_allow_html=True)
        st.markdown("<div class='nothing-label'>// MULTIMODAL INTERVIEW OS</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FLOW STEP INDICATORS ──
    current_sc = st.session_state.current_screen
    steps = [
        ("welcome", "01 // WELCOME"),
        ("setup", "02 // SETUP"),
        ("calibration", "03 // CALIBRATION"),
        ("interview", "04 // INTERVIEW"),
        ("results", "05 // RESULTS"),
    ]

    st.markdown("<div class='nothing-label'>[ INTERVIEW PIPELINE ]</div>", unsafe_allow_html=True)
    for s_key, s_label in steps:
        is_active = (current_sc == s_key) or (s_key == "welcome" and current_sc == "landing") or (s_key == "setup" and current_sc == "config") or (s_key == "results" and current_sc == "report")
        if is_active:
            st.markdown(f"<span class='nothing-badge nothing-badge-green' style='width: 100%; justify-content: flex-start; margin-bottom: 0.35rem;'><span class='nothing-live-dot'></span>{s_label}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='nothing-badge' style='width: 100%; justify-content: flex-start; margin-bottom: 0.35rem; color: var(--beige-dim);'>{s_label}</span>", unsafe_allow_html=True)

    st.divider()

    # ── CAMERA ORIENTATION CONTROL ──
    st.markdown("<div class='nothing-label'>[ CAMERA ORIENTATION ]</div>", unsafe_allow_html=True)
    m_val = st.toggle(
        "Mirror Camera View",
        value=bool(st.session_state.get("camera_mirror", False)),
        key="camera_mirror_sidebar_toggle",
        help="ON = Selfie mirror view. OFF = Natural camera view.",
    )
    if m_val != st.session_state.get("camera_mirror", False):
        st.session_state.camera_mirror = m_val
        AppEngineBridge.mirror = m_val
        st.rerun()
    AppEngineBridge.mirror = bool(st.session_state.get("camera_mirror", False))

    st.divider()

    # ── ACTIVE INTERVIEW CONTROLS ──
    session_active = st.session_state.controller.session_manager.session_data.get("is_active", False)
    if session_active and current_sc == "interview":
        st.markdown("<div class='nothing-label'>[ SESSION CONTROL ]</div>", unsafe_allow_html=True)
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
            go_to_screen("results")

    # ── GLOBAL NAVIGATION BUTTONS ──
    st.markdown("<div class='nothing-label'>[ QUICK NAVIGATION ]</div>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("HOME", use_container_width=True):
            go_to_screen("welcome")
    with col_nav2:
        if st.button("SETUP", use_container_width=True):
            go_to_screen("setup")

    # ── ADVANCED DEVELOPER TELEMETRY (EXPANDABLE) ──
    with st.expander("[+] System Diagnostics"):
        is_ollama_ok, ollama_msg, available_models = st.session_state.ollama_client.check_connection()
        st.write(f"**Ollama:** {'Online' if is_ollama_ok else 'Offline'}")
        st.write(f"**Models:** {len(available_models)} detected")
        st.write(f"**Vision Engine:** {'Active' if st.session_state.posture_engine else 'Offline'}")
        st.write(f"**FFmpeg:** {'Available' if is_ffmpeg_available() else 'Missing'}")
        st.write(f"**WebRTC:** {'Active' if WEBRTC_AVAILABLE else 'Headless'}")


# =============================================================================
# SCREEN 1: WELCOME DASHBOARD
# =============================================================================

if st.session_state.current_screen in ("welcome", "landing"):
    render_page_header(
        title="HIRESENSE INTERVIEW PLATFORM",
        tagline="Multimodal AI-Powered Virtual Interviewer & Real-Time Behavioural Coaching OS",
        badge_text="SYSTEM OPERATIONAL",
        badge_variant="green",
    )

    # 1. System Readiness Overview Strip
    is_ollama_ok, _, available_models = st.session_state.ollama_client.check_connection()
    has_vision = bool(st.session_state.posture_engine and st.session_state.posture_engine.is_running)
    has_audio = is_ffmpeg_available()
    render_system_readiness_strip(is_ollama_ok, len(available_models), has_vision, has_audio)

    # 2. Hero Primary CTA & Introduction
    c_hero1, c_hero2 = st.columns([3, 2])
    with c_hero1:
        st.markdown(
            clean_html("""
            <div class="nothing-card" style="padding: 1.5rem 1.6rem;">
                <div class="nothing-label">// INTRODUCTORY BRIEF</div>
                <div style="font-size: 1.05rem; font-weight: 500; color: var(--beige-light); line-height: 1.6; margin-bottom: 1rem;">
                    Prepare for technical and behavioral interviews with real-time feedback across three independent dimensions:
                    technical accuracy, verbal delivery, and non-verbal biomechanical composure.
                </div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <span class="nothing-badge">OLLAMA LOCAL REASONING</span>
                    <span class="nothing-badge">MEDIAPIPE WIREFRAME</span>
                    <span class="nothing-badge">SPEECH TRANSCRIPTION</span>
                    <span class="nothing-badge">NON-DIAGNOSTIC COACHING</span>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with c_hero2:
        st.markdown(
            clean_html("""
            <div class="nothing-card" style="padding: 1.5rem 1.6rem; height: calc(100% - 1rem); display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div class="nothing-label">// QUICK ACTION</div>
                    <div style="font-size: 0.88rem; color: var(--beige-muted); line-height: 1.5; margin-bottom: 1.2rem;">
                        Start a tailored mock interview session. Configure subject focus, question count, and response modes.
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("START NEW INTERVIEW", type="primary", use_container_width=True):
            go_to_screen("setup")

    # 3. Three Core Pillars Overview
    render_section_title("EVALUATION ARCHITECTURE", "Three Independent Assessment Pillars")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown(
            clean_html("""
            <div class="nothing-card">
                <div class="nothing-card-header">
                    <span>[ PILLAR 01 ]</span>
                    <span class="nothing-badge nothing-badge-green">LOCAL LLM</span>
                </div>
                <div style="font-weight: 700; font-size: 1rem; color: var(--beige-light); margin-bottom: 0.4rem;">
                    Answer Intelligence
                </div>
                <div style="font-size: 0.80rem; color: var(--beige-muted); line-height: 1.5;">
                    Deep semantic evaluation of candidate responses: technical correctness, reasoning depth, completeness, and conceptual clarity evaluated using Qwen through local Ollama.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            clean_html("""
            <div class="nothing-card">
                <div class="nothing-card-header">
                    <span>[ PILLAR 02 ]</span>
                    <span class="nothing-badge nothing-badge-green">STT & PHONETICS</span>
                </div>
                <div style="font-weight: 700; font-size: 1rem; color: var(--beige-light); margin-bottom: 0.4rem;">
                    Speech Intelligence
                </div>
                <div style="font-size: 0.80rem; color: var(--beige-muted); line-height: 1.5;">
                    Real-time speech recognition, deterministic filler-word frequency analysis (e.g. 'um', 'uh', 'you know'), speaking pace, and optional phonetic assessment.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with col_p3:
        st.markdown(
            clean_html("""
            <div class="nothing-card">
                <div class="nothing-card-header">
                    <span>[ PILLAR 03 ]</span>
                    <span class="nothing-badge nothing-badge-green">MEDIAPIPE TASKS</span>
                </div>
                <div style="font-weight: 700; font-size: 1rem; color: var(--beige-light); margin-bottom: 0.4rem;">
                    Visual Behaviour
                </div>
                <div style="font-size: 0.80rem; color: var(--beige-muted); line-height: 1.5;">
                    Unmirrored optical tracking measuring posture alignment, solvePnP 3D head position, iris gaze direction, and gesture composure as non-diagnostic practice indicators.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    # 4. Previous Session History (if available)
    if st.session_state.session_history:
        render_section_title("PREVIOUS SESSIONS", "Recent Interview Performance History")
        for idx, past_sess in enumerate(reversed(st.session_state.session_history[-3:])):
            c_hist1, c_hist2, c_hist3 = st.columns([3, 2, 1.5])
            with c_hist1:
                st.markdown(f"**{past_sess.get('candidate_name')}** — {past_sess.get('category')} ({past_sess.get('subject')})")
            with c_hist2:
                st.markdown(f"Score: `{past_sess.get('average_score', 0):.1f}/10` · {past_sess.get('date')}")
            with c_hist3:
                if st.button(f"VIEW REPORT [{idx+1}]", key=f"hist_btn_{idx}"):
                    st.session_state.final_report = past_sess.get("report")
                    go_to_screen("results")


# =============================================================================
# SCREEN 2: INTERVIEW CONFIGURATION & SETUP
# =============================================================================

elif st.session_state.current_screen in ("setup", "config"):
    render_page_header(
        title="INTERVIEW CONFIGURATION",
        tagline="Select your domain, difficulty, question volume, and evaluation parameters.",
        badge_text="STEP 01 OF 04",
    )

    with st.form("interview_setup_form"):
        # ── SECTION A: INTERVIEW CATEGORY ──
        render_section_title("SECTION A // INTERVIEW CATEGORY", "Select primary interview orientation")
        category_options = ["Technical", "HR", "Mixed"]
        interview_type = st.radio(
            "Category",
            category_options,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            help="Technical drills technical concepts. HR focuses on behavioral scenarios. Mixed combines both.",
        )

        # Visual category description card
        cat_desc = {
            "Technical": "In-depth engineering questions covering core CS fundamentals, system design, and coding concepts.",
            "HR": "Behavioral questions assessing leadership, teamwork, conflict resolution, and communication skills.",
            "Mixed": "Comprehensive interview simulating real-world screening with both technical and behavioral prompts.",
        }[interview_type]
        st.markdown(render_selection_card_html(interview_type.upper(), cat_desc, selected=True), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SECTION B: DOMAIN & DIFFICULTY ──
        render_section_title("SECTION B // DOMAIN & DIFFICULTY", "Specify target technical subject and depth")
        col_s1, col_s2, col_s3 = st.columns([2, 1.5, 1.5])

        with col_s1:
            if interview_type == "Technical":
                available_subj = AVAILABLE_TECHNICAL_SUBJECTS
            elif interview_type == "HR":
                available_subj = AVAILABLE_HR_SUBJECTS
            else:
                available_subj = AVAILABLE_ALL_SUBJECTS

            subject = st.selectbox("Subject Focus", available_subj, index=0)

        with col_s2:
            difficulty = st.selectbox("Difficulty Tier", AVAILABLE_DIFFICULTIES, index=0)

        with col_s3:
            num_questions = st.selectbox("Question Count", [3, 5, 8, 10], index=1)

        candidate_name = st.text_input("Candidate Name", value="Candidate", max_chars=40)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SECTION C: INTERVIEW PREFERENCES ──
        render_section_title("SECTION C // INTERACTION PREFERENCES", "Configure input and sensory channels")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            answer_mode = st.radio(
                "Response Mode",
                ["Written", "Oral"],
                index=0,
                help="Written: type structured answers. Oral: record verbal response with speech-to-text transcription.",
            )

        with col_p2:
            enable_webcam = st.toggle(
                "Enable Computer Vision Analysis",
                value=True,
                help="Tracks posture alignment, head orientation, and eye contact using MediaPipe Tasks.",
            )
            adaptive_followups = st.toggle(
                "Adaptive Follow-up Questions",
                value=False,
                help="AI interviewer generates contextual probing follow-ups based on candidate's answers.",
            )

        # ── SECTION D: ADVANCED SETTINGS (EXPANDABLE) ──
        with st.expander("[+] Advanced Engine Configuration"):
            st.markdown("<div class='nothing-label'>// OLLAMA LLM & EVALUATOR</div>", unsafe_allow_html=True)
            try:
                _, _, ollama_models = st.session_state.ollama_client.check_connection()
            except Exception:
                ollama_models = []
            model_options = ollama_models if ollama_models else [DEFAULT_OLLAMA_MODEL]
            selected_model = st.selectbox(
                "Ollama Model",
                model_options,
                index=0,
                help="Local Ollama model name (e.g. qwen3.5:4b, llama3).",
            )
            st.session_state.selected_model = selected_model

            scoring_profile = st.selectbox(
                "Scoring Weight Profile",
                list(SCORING_PROFILES.keys()),
                index=0,
                format_func=lambda x: f"{x.upper()} (Posture 25%, Eye Contact 20%, Gestures 25%)",
            )
            set_scoring_profile(scoring_profile)

        # ── SECTION E: CONFIGURATION SUMMARY & SUBMIT ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            clean_html(f"""
            <div class="nothing-card" style="padding: 0.9rem 1.1rem; border-color: var(--border-beige);">
                <div class="nothing-label">// CONFIGURATION SUMMARY</div>
                <div style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--beige-primary); line-height: 1.6;">
                    CATEGORY: <b>{interview_type.upper()}</b> · SUBJECT: <b>{subject.upper()}</b> · DIFFICULTY: <b>{difficulty.upper()}</b><br>
                    QUESTIONS: <b>{num_questions}</b> · RESPONSE MODE: <b>{answer_mode.upper()}</b> · VISION: <b>{'ENABLED' if enable_webcam else 'DISABLED'}</b>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
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
        st.session_state.current_transcript = ""
        st.session_state.current_evaluation = None
        st.session_state.current_audio_bytes = None
        st.session_state.current_pronunciation = None

        if enable_webcam and WEBRTC_AVAILABLE:
            if st.session_state.posture_engine:
                st.session_state.posture_engine.reset_calibration()
            go_to_screen("calibration")
        else:
            go_to_screen("interview")


# =============================================================================
# SCREEN 3: CAMERA AND MICROPHONE CALIBRATION
# =============================================================================

elif st.session_state.current_screen == "calibration":
    render_page_header(
        title="CALIBRATION & SENSORY CHECK",
        tagline="Verify optical alignment, baseline posture, and microphone input before commencing.",
        badge_text="STEP 02 OF 04",
    )

    col_cam, col_info = st.columns([1.8, 1.2])

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
            clean_html("""
            <div class="nothing-card">
                <div class="nothing-card-header">[ POSITIONING PROTOCOL ]</div>
                <div style="font-family: var(--font-mono); font-size: 0.74rem; color: var(--beige-muted); line-height: 1.8;">
                    01. Position upper torso & shoulders centered in frame.<br>
                    02. Maintain natural upright spine alignment.<br>
                    03. Look directly into your camera lens.<br>
                    04. Green wireframe lines verify active landmark tracking.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # Dynamic Calibration Telemetry Widget
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
        components.html(cal_html, height=100)

        # Oral Mode Microphone Test (if selected)
        settings = st.session_state.controller.session_manager.session_data.get("settings", {})
        if settings.get("answer_mode") == "Oral":
            st.markdown("<div class='nothing-label'>[ MICROPHONE CHECK ]</div>", unsafe_allow_html=True)
            mic_test = st.audio_input("Record audio check", key="cal_mic_check", label_visibility="collapsed")
            if mic_test is not None:
                st.session_state.mic_test_status = "MICROPHONE OK"
                st.markdown("<span class='nothing-badge nothing-badge-green' style='width: 100%; justify-content: center; padding: 0.4rem;'>MICROPHONE SIGNAL VERIFIED</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Primary Proceed Action & Explicit Skip
        if st.button("CONFIRM CALIBRATION & PROCEED", type="primary", use_container_width=True):
            if st.session_state.posture_engine:
                st.session_state.posture_engine.is_calibrated = True
            go_to_screen("interview")

        if st.button("SKIP CALIBRATION", type="secondary", use_container_width=True):
            if st.session_state.posture_engine:
                st.session_state.posture_engine.is_calibrated = True
            go_to_screen("interview")


# =============================================================================
# SCREEN 4: LIVE INTERVIEW & DISTRACTION-FREE ANSWERING
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
        go_to_screen("results")

    session_data = st.session_state.controller.session_manager.get_session_data()
    q_idx = session_data.get("current_index", 0)
    total_q = len(session_data.get("questions", []))
    settings = session_data.get("settings", {})
    webcam_enabled = settings.get("webcam_enabled", True)

    # Calculate session timer
    started_ts = session_data.get("started_at") or time.time()
    elapsed_sec = int(time.time() - started_ts)
    e_mins, e_secs = divmod(elapsed_sec, 60)
    time_str = f"{e_mins:02d}:{e_secs:02d}"

    # 1. Distraction-Free Top Bar Navigation
    render_interview_topbar(
        current_index=q_idx,
        total_questions=total_q,
        category=curr_q.get("category", "TECHNICAL"),
        subject=curr_q.get("subject", "GENERAL"),
        difficulty=curr_q.get("difficulty", "EASY"),
        time_elapsed_formatted=time_str,
    )

    # 2. Main 68% / 32% Desktop Grid Layout
    col_main, col_vision = st.columns([2.1, 1.0] if webcam_enabled else [1, 0.01])

    # ── LEFT: FOCAL QUESTION & ANSWERING WORKSPACE ──
    with col_main:
        render_question_box(curr_q.get("question", ""), q_idx)

        # Optional Context / Concept hints
        if curr_q.get("key_concepts"):
            with st.expander("// CONCEPTUAL SCOPE"):
                st.markdown(f"Expected coverage topics: `{curr_q.get('key_concepts')}`")

        # ── STATE A & B: ANSWER INPUT AREA ──
        user_answer_text = ""
        speech_result_data = None

        if settings.get("answer_mode") == "Written":
            st.markdown("<div class='nothing-label'>[ ANSWER INPUT : WRITTEN TERMINAL ]</div>", unsafe_allow_html=True)
            user_answer_text = st.text_area(
                "Written Answer",
                height=200,
                key=f"written_ans_{q_idx}",
                placeholder="Type your structured technical explanation here...",
                label_visibility="collapsed",
            )
            # Live word counter
            words_count = len(user_answer_text.split()) if user_answer_text.strip() else 0
            st.markdown(f"<div style='text-align: right; font-family: var(--font-mono); font-size: 0.68rem; color: var(--beige-dim);'>WORD COUNT: {words_count}</div>", unsafe_allow_html=True)

        else: # Oral Mode
            st.markdown("<div class='nothing-label'>[ AUDIO CAPTURE : ORAL RESPONSE ]</div>", unsafe_allow_html=True)
            audio_recording = st.audio_input("Microphone Input", key=f"oral_input_{q_idx}", label_visibility="collapsed")

            if audio_recording is not None:
                st.session_state.current_audio_bytes = audio_recording.getvalue()
                if st.button("TRANSCRIBE AUDIO // GOOGLE STT", type="secondary"):
                    with st.spinner("Decoding audio phonetics..."):
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
                    height=110,
                    key=f"transcript_edit_{q_idx}",
                    label_visibility="collapsed",
                )
                if edited_transcript != st.session_state.current_transcript:
                    st.session_state.current_transcript = edited_transcript
                    st.session_state.current_evaluation = None

                user_answer_text = st.session_state.current_transcript

        # ── STATE C: SUBMISSION ACTION ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("SUBMIT & EVALUATE ANSWER", type="primary", use_container_width=True):
            if not user_answer_text.strip():
                st.warning("// Input required: Please type or record an answer before submission.")
            else:
                with st.spinner("// Ollama LLM analyzing response concepts..."):
                    # Speech filler analysis
                    speech_result_data = None
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

        # ── STATE D & E: DEDICATED ANSWER EVALUATION EXPERIENCE ──
        if st.session_state.current_evaluation:
            ev = st.session_state.current_evaluation
            st.divider()
            render_section_title("EVALUATION DOSSIER", "Structured Assessment Breakdown")

            # Score Cards Row
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(render_score_card("CORRECTNESS", ev.get("correctness", 0)), unsafe_allow_html=True)
            c2.markdown(render_score_card("DEPTH", ev.get("depth", 0)), unsafe_allow_html=True)
            c3.markdown(render_score_card("COMPLETENESS", ev.get("completeness", 0)), unsafe_allow_html=True)
            avg_c = (ev.get("correctness", 0) + ev.get("depth", 0) + ev.get("completeness", 0)) / 3.0
            c4.markdown(render_score_card("CONTENT SCORE", round(avg_c, 1)), unsafe_allow_html=True)

            # Detailed Feedback Narrative
            st.markdown(
                clean_html(f"""
                <div class="nothing-card">
                    <div class="nothing-label">// EVALUATION SYNTHESIS</div>
                    <div style="font-size: 0.90rem; color: var(--beige-light); line-height: 1.55; margin-bottom: 0.75rem;">
                        {ev.get('overall_feedback') or ev.get('explanation')}
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

            # Strengths & Improvement Areas
            col_st, col_imp = st.columns(2)
            with col_st:
                st.markdown("<div class='nothing-label'>[ OBSERVED STRENGTHS ]</div>", unsafe_allow_html=True)
                for s in ev.get("strengths", []):
                    st.markdown(f"- [+] {s}")
            with col_imp:
                st.markdown("<div class='nothing-label'>[ AREAS FOR IMPROVEMENT ]</div>", unsafe_allow_html=True)
                for imp in ev.get("areas_to_improve", []):
                    st.markdown(f"- [!] {imp}")

            # Speech Intelligence Feedback (Oral Mode)
            if settings.get("answer_mode") == "Oral" and st.session_state.current_transcript:
                render_section_title("SPEECH INTELLIGENCE", "Verbal Delivery Telemetry")
                fill_report = analyze_fillers(st.session_state.current_transcript)
                sp_c1, sp_c2, sp_c3 = st.columns(3)
                sp_c1.metric("Filler Words", fill_report["total"])
                sp_c2.metric("Total Words", fill_report["word_count"])
                sp_c3.metric("Fillers / 100 Words", f"{fill_report['rate_per_100_words']:.1f}")

                if fill_report["counts"]:
                    st.write("**Detected Fillers:** " + ", ".join(f"`{w}`: {n}" for w, n in fill_report["counts"].items()))

                if is_openpronounce_available():
                    if st.button("RUN PRONUNCIATION ASSESSMENT"):
                        with st.spinner("Analyzing phonetics..."):
                            p_res, p_err = analyze_pronunciation(
                                st.session_state.current_audio_bytes,
                                st.session_state.current_transcript
                            )
                            if p_err:
                                st.error(p_err)
                            else:
                                st.session_state.current_pronunciation = p_res
                                st.success(f"Pronunciation Score: {p_res.get('score', 'N/A')}/100")

            # Reference Answer Expander
            with st.expander("// VIEW EXPECTED REFERENCE ANSWER"):
                st.write(ev.get("reference_answer") or curr_q.get("reference_answer"))

            # Next Question Navigation Action
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
                    avg_score = report["answer_intelligence"].get("overall_content_score", report["answer_intelligence"].get("average_content_score", 0.0))
                    # Archive into session history
                    st.session_state.session_history.append({
                        "session_id": report["overview"]["session_id"],
                        "candidate_name": report["overview"]["candidate_name"],
                        "category": report["overview"]["category"],
                        "subject": report["overview"]["subject"],
                        "average_score": avg_score,
                        "date": report["overview"]["timestamp"],
                        "report": report,
                    })
                    go_to_screen("results")

    # ── RIGHT: LIVE WEBCAM & BIOMECHANICAL TELEMETRY ──
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

            # WebRTC Camera Stream
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
                st.info("// WebRTC camera stream running in headless mode.")

            # Live Biomechanical Optical Telemetry Strip
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
# SCREEN 5: INTERVIEW RESULTS & COACHING DOSSIER
# =============================================================================

elif st.session_state.current_screen in ("results", "report"):
    if not st.session_state.final_report:
        v_summary = st.session_state.posture_engine.get_summary() if st.session_state.posture_engine else {}
        s_summary = st.session_state.controller.speech_aggregator.get_summary()
        report = ReportGenerator.generate_report(
            st.session_state.controller.session_manager.get_session_data(),
            speech_summary=s_summary,
            visual_summary=v_summary,
        )
        st.session_state.final_report = report
    else:
        report = st.session_state.final_report

    ov = report["overview"]
    ans_intel = report["answer_intelligence"]
    speech_intel = report["speech_intelligence"]
    vis_intel = report["visual_intelligence"]
    recs = report.get("recommendations", report.get("coaching_recommendations", []))

    render_page_header(
        title="INTERVIEW PERFORMANCE DOSSIER",
        tagline=f"Comprehensive Coaching Synthesis for {ov['candidate_name'].upper()} · Session {ov['session_id']}",
        badge_text="COMPLETE",
        badge_variant="green",
    )

    # 1. Performance Overview Top Card
    st.markdown(
        clean_html(f"""
        <div class="nothing-card" style="padding: 1.25rem 1.4rem; border-color: var(--border-beige);">
            <div class="nothing-card-header">
                <span>[ EXECUTIVE SUMMARY ]</span>
                <span>{ov['timestamp']}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 0.8rem;">
                <div><span class="nothing-label">CANDIDATE</span><b style="color: var(--beige-light); font-size: 0.95rem;">{ov['candidate_name']}</b></div>
                <div><span class="nothing-label">CATEGORY</span><b style="color: var(--beige-light); font-size: 0.95rem;">{ov['category']}</b></div>
                <div><span class="nothing-label">SUBJECT</span><b style="color: var(--beige-light); font-size: 0.95rem;">{ov['subject']}</b></div>
                <div><span class="nothing-label">DIFFICULTY</span><b style="color: var(--beige-light); font-size: 0.95rem;">{ov['difficulty']}</b></div>
                <div><span class="nothing-label">QUESTIONS</span><b style="color: var(--beige-light); font-size: 0.95rem;">{ov['completed_questions']} / {ov['total_questions']}</b></div>
                <div><span class="nothing-label">DURATION</span><b style="color: var(--beige-light); font-size: 0.95rem;">{ov['duration_formatted']}</b></div>
            </div>
            <div style="font-size: 0.88rem; color: var(--beige-muted); line-height: 1.5; border-top: 1px solid var(--border-subtle); padding-top: 0.6rem;">
                {ans_intel.get('overall_assessment_narrative', 'Interview completed successfully.')}
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # 2. Three Intelligence Dimension Columns
    render_section_title("THE THREE INTELLIGENCE DIMENSIONS", "Multimodal Performance Breakdown")
    col_d1, col_d2, col_d3 = st.columns(3)

    # Dimension 1: Answer Intelligence
    with col_d1:
        avg_score = ans_intel.get("overall_content_score", ans_intel.get("average_content_score", 0.0))
        tier = "Distinguished" if avg_score >= 8.5 else ("Proficient" if avg_score >= 7.0 else ("Competent" if avg_score >= 5.0 else "Developing"))
        st.markdown(
            clean_html(f"""
            <div class="nothing-card" style="height: 100%;">
                <div class="nothing-card-header">
                    <span>[ DIMENSION 01 ]</span>
                    <span class="nothing-badge nothing-badge-green">ANSWER INTEL</span>
                </div>
                <div style="margin-bottom: 0.75rem;">
                    <span class="nothing-metric-val">{avg_score:.1f}</span>
                    <span class="nothing-metric-unit">/10</span>
                    <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--tech-green); margin-top: 0.2rem;">[ {tier.upper()} ]</div>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--beige-muted); line-height: 1.8;">
                    CORRECTNESS: <b>{ans_intel.get('average_correctness', 0):.1f}/10</b><br>
                    REASONING DEPTH: <b>{ans_intel.get('average_depth', 0):.1f}/10</b><br>
                    COMPLETENESS: <b>{ans_intel.get('average_completeness', 0):.1f}/10</b>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    # Dimension 2: Speech Intelligence
    with col_d2:
        tot_fillers = speech_intel.get("total_fillers", speech_intel.get("total_fillers_detected", 0))
        filler_rate = speech_intel.get("filler_rate_per_100", speech_intel.get("filler_words_per_100_words", 0.0))
        tot_words = speech_intel.get("total_words", speech_intel.get("total_words_spoken", 0))
        st.markdown(
            clean_html(f"""
            <div class="nothing-card" style="height: 100%;">
                <div class="nothing-card-header">
                    <span>[ DIMENSION 02 ]</span>
                    <span class="nothing-badge nothing-badge-green">SPEECH INTEL</span>
                </div>
                <div style="margin-bottom: 0.75rem;">
                    <span class="nothing-metric-val">{filler_rate:.1f}</span>
                    <span class="nothing-metric-unit">fillers/100w</span>
                    <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--beige-muted); margin-top: 0.2rem;">[ TOTAL WORDS: {tot_words} ]</div>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--beige-muted); line-height: 1.8;">
                    TOTAL FILLERS: <b>{tot_fillers}</b><br>
                    SPEAKING PACE: <b>{speech_intel.get('pacing_indicator', 'NATURAL')}</b><br>
                    PHONETIC ACCURACY: <b>{speech_intel.get('average_pronunciation_score') or 'N/A'}</b>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    # Dimension 3: Visual Behaviour Intelligence
    with col_d3:
        posture_avg = vis_intel.get("posture_score", vis_intel.get("posture_alignment_score", 0.0))
        eye_avg = vis_intel.get("eye_contact_score", vis_intel.get("eye_contact_percentage", 0.0))
        head_score = vis_intel.get("head_position_score", 0.0)
        rel_score = vis_intel.get("reliability_score", vis_intel.get("tracking_reliability_percentage", 100.0))
        st.markdown(
            clean_html(f"""
            <div class="nothing-card" style="height: 100%;">
                <div class="nothing-card-header">
                    <span>[ DIMENSION 03 ]</span>
                    <span class="nothing-badge nothing-badge-green">VISUAL INTEL</span>
                </div>
                <div style="margin-bottom: 0.75rem;">
                    <span class="nothing-metric-val">{posture_avg:.0f}</span>
                    <span class="nothing-metric-unit">/100</span>
                    <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--beige-muted); margin-top: 0.2rem;">[ POSTURE ALIGNMENT ]</div>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--beige-muted); line-height: 1.8;">
                    EYE CONTACT: <b>{eye_avg:.0f}%</b><br>
                    HEAD POSITION: <b>{head_score:.0f}/100</b><br>
                    RELIABILITY: <b>{rel_score:.0f}%</b>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    # 3. Candidate Strengths & Improvement Areas
    render_section_title("SYNTHESIZED INSIGHTS", "Key Strengths & Growth Directives")
    c_ins1, c_ins2 = st.columns(2)

    with c_ins1:
        st.markdown("<div class='nothing-label'>[ OBSERVED CANDIDATE STRENGTHS ]</div>", unsafe_allow_html=True)
        strengths = ans_intel.get("primary_strengths", [])
        if not strengths:
            for q in ans_intel.get("questions_evaluated", []):
                strengths.extend(q.get("strengths", []))
        if strengths:
            for s in list(dict.fromkeys(strengths))[:5]:
                st.markdown(f"- [+] {s}")
        else:
            st.markdown("- [+] Consistent structured responses provided.")

    with c_ins2:
        st.markdown("<div class='nothing-label'>[ AREAS FOR IMPROVEMENT ]</div>", unsafe_allow_html=True)
        weaknesses = ans_intel.get("primary_weaknesses", [])
        if not weaknesses:
            for q in ans_intel.get("questions_evaluated", []):
                weaknesses.extend(q.get("areas_to_improve", []))
        if weaknesses:
            for w in list(dict.fromkeys(weaknesses))[:5]:
                st.markdown(f"- [!] {w}")
        else:
            st.markdown("- [!] Deepen technical architecture and edge-case coverage.")

    # 4. Actionable Coaching Directives
    if recs:
        render_section_title("ACTIONABLE COACHING DIRECTIVES", "Strategic Practice Recommendations")
        for rec in recs:
            st.markdown(f"- [*] {rec}")

    # 5. Question-by-Question Detailed Breakdown
    render_section_title("QUESTION-BY-QUESTION EVALUATION BREAKDOWN", "Granular Responses & Scoring")
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

    # 6. Performance Export & New Session
    st.divider()
    render_section_title("EXPORT PERFORMANCE DOSSIER", "Preserve Records or Initialize Next Session")
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
            go_to_screen("welcome")
