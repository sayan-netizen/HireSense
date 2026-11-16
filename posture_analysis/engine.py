"""
PostureEngine: High-Level, Thread-Safe Visual Behaviour Engine for HireSense.
Integrates MediaPipe Pose, Face Mesh, and Hand Landmarkers with real-time
visual scoring, calibration, and dashboard rendering.
"""

import os
import time
import threading
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

try:
    from . import config
    from .posture_analyzer import PostureAnalyzer
    from .face_analyzer import FaceAnalyzer
    from .hand_analyzer import HandAnalyzer
    from .scoring import InterviewScorer
    from .visualization import DashboardVisualizer
except (ImportError, ValueError):
    import config
    from posture_analyzer import PostureAnalyzer
    from face_analyzer import FaceAnalyzer
    from hand_analyzer import HandAnalyzer
    from scoring import InterviewScorer
    from visualization import DashboardVisualizer


class PostureEngine:
    """
    Modular, thread-safe visual analysis engine.
    Orchestrates MediaPipe tasks, geometric analyzers, scoring, and rendering.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.lock = threading.Lock()
        
        # Paths
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.models_dir = models_dir
        self.pose_model_path = os.path.join(self.models_dir, "pose_landmarker.task")
        self.face_model_path = os.path.join(self.models_dir, "face_landmarker.task")
        self.hand_model_path = os.path.join(self.models_dir, "hand_landmarker.task")

        # MediaPipe Task instances
        self.pose_landmarker: Optional[vision.PoseLandmarker] = None
        self.face_landmarker: Optional[vision.FaceLandmarker] = None
        self.hand_landmarker: Optional[vision.HandLandmarker] = None

        # Analysis Modules
        self.posture_analyzer: Optional[PostureAnalyzer] = None
        self.face_analyzer: Optional[FaceAnalyzer] = None
        self.hand_analyzer: Optional[HandAnalyzer] = None
        self.scorer: Optional[InterviewScorer] = None
        self.visualizer: Optional[DashboardVisualizer] = None

        # State flags
        self.is_running: bool = False
        self.is_calibrated: bool = False
        self.calibration_frames_collected: int = 0
        self.target_calibration_frames: int = config.CALIBRATION_FRAMES
        self.calibration_start_time: Optional[float] = None
        self.calibration_duration: float = 4.0

        # Frame timing
        self.frame_ts_ms: int = 0
        self.fps_timer: float = time.time()
        self.fps_count: int = 0
        self.current_fps: float = 0.0
        self.session_start_time: float = time.time()

        # ── Performance: frame skipping & analysis throttle ──
        self._frame_counter: int = 0
        # Run full 3-model inference every N frames; on skipped frames, reuse
        # the last landmarks and just redraw the mesh overlay.  This cuts
        # MediaPipe GPU/CPU cost by ~60-70%.
        self.inference_skip: int = 2          # analyse every 2nd frame
        self.hand_skip: int = 6               # hand model every 6th frame for high responsiveness
        self._last_pose_lm = None
        self._last_face_lm = None
        self._last_blendshapes = None
        self._last_hand_lms: list = []

        # Thread-safe cached metrics snapshot - initialized to realistic 0 / inactive state
        self._latest_metrics: Dict[str, Any] = {
            "scores": {
                "posture": 0.0,
                "eye_contact": 0.0,
                "head_position": 0.0,
                "expression": 0.0,
                "stability": 0.0,
                "gestures": 0.0,
                "overall": 0.0,
                "reliability": 0.0,
            },
            "posture": {},
            "face": {},
            "hand": {},
            "fps": 0.0,
            "session_time": 0.0,
            "is_calibrated": False,
            "calibration_progress": 0.0,
            "reliability_label": "Unknown",
            "warnings": ["Camera initializing or awaiting optical frame"],
        }

    def start(self) -> None:
        """Initialize MediaPipe tasks and analyzers."""
        with self.lock:
            if self.is_running:
                return

            if not os.path.isfile(self.pose_model_path):
                raise FileNotFoundError(f"Missing Pose Landmarker model: {self.pose_model_path}")
            if not os.path.isfile(self.face_model_path):
                raise FileNotFoundError(f"Missing Face Landmarker model: {self.face_model_path}")
            if not os.path.isfile(self.hand_model_path):
                raise FileNotFoundError(f"Missing Hand Landmarker model: {self.hand_model_path}")

            # Pose Landmarker
            pose_options = vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.pose_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

            # Face Landmarker
            face_options = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.face_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

            # Hand Landmarker
            hand_options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

            # Analyzers
            self.posture_analyzer = PostureAnalyzer()
            self.face_analyzer = FaceAnalyzer()
            self.hand_analyzer = HandAnalyzer()
            self.scorer = InterviewScorer()
            self.visualizer = DashboardVisualizer()

            self.frame_ts_ms = 0
            self.fps_timer = time.time()
            self.fps_count = 0
            self.current_fps = 0.0
            self.session_start_time = time.time()
            self.is_calibrated = False
            self.calibration_frames_collected = 0
            self.calibration_start_time = None
            self.is_running = True

    def reset_calibration(self) -> None:
        """Reset calibration progress to re-calibrate."""
        with self.lock:
            if self.posture_analyzer:
                self.posture_analyzer = PostureAnalyzer()
            if self.face_analyzer:
                self.face_analyzer = FaceAnalyzer()
            if self.hand_analyzer:
                self.hand_analyzer = HandAnalyzer()
            self.is_calibrated = False
            self.calibration_frames_collected = 0
            self.calibration_start_time = None
            self._latest_metrics["is_calibrated"] = False
            self._latest_metrics["calibration_progress"] = 0.0

    def calibrate_frame(self, frame: np.ndarray, mirror: bool = False) -> Tuple[np.ndarray, float, bool]:
        """
        Processes a calibration frame.
        Returns:
            (annotated_frame, progress_pct [0.0 - 100.0], is_calibrated)
        """
        with self.lock:
            if not self.is_running:
                self.start()

            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.time() * 1000)
            self.frame_ts_ms = max(self.frame_ts_ms + 1, now_ms)

            pose_result = self.pose_landmarker.detect_for_video(mp_image, self.frame_ts_ms)
            face_result = self.face_landmarker.detect_for_video(mp_image, self.frame_ts_ms)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, self.frame_ts_ms)

            has_pose = len(pose_result.pose_landmarks) > 0
            has_face = len(face_result.face_landmarks) > 0
            has_detection = has_pose or has_face

            pose_lm = pose_result.pose_landmarks[0] if has_pose else None
            face_lm = face_result.face_landmarks[0] if has_face else None

            if has_detection:
                if self.calibration_start_time is None:
                    self.calibration_start_time = time.time()

                self.calibration_frames_collected += 1

                if pose_lm is not None:
                    self.posture_analyzer.calibrate_frame(pose_lm)

                if face_lm is not None:
                    self.face_analyzer.calibrate_frame(face_lm, (frame.shape[1], frame.shape[0]))

                if hand_result.hand_landmarks:
                    self.hand_analyzer.calibrate_frame(hand_result.hand_landmarks)

                elapsed = time.time() - self.calibration_start_time
                frame_ratio = self.calibration_frames_collected / self.target_calibration_frames
                time_ratio = elapsed / self.calibration_duration
                # Progress is driven by frame acquisition with temporal pacing
                progress_pct = min(100.0, max(frame_ratio, min(time_ratio, frame_ratio * 1.5)) * 100.0)

                # Fast OpenCV mesh drawing (zero PIL conversion)
                annotated = self.visualizer.draw_mesh_only(
                    frame,
                    pose_lm=pose_lm,
                    face_lm=face_lm,
                    hand_lms_list=hand_result.hand_landmarks if hand_result.hand_landmarks else [],
                )

                # Require a meaningful sample size (at least 30 frames) before finalizing calibration
                min_frames = min(30, self.target_calibration_frames // 2)
                can_finish = (
                    self.calibration_frames_collected >= self.target_calibration_frames
                    or (elapsed >= self.calibration_duration and self.calibration_frames_collected >= min_frames)
                )

                if can_finish:
                    self.posture_analyzer.finish_calibration()
                    self.face_analyzer.finish_calibration()
                    self.hand_analyzer.finish_calibration()
                    self.is_calibrated = bool(
                        self.face_analyzer.calibrated or self.posture_analyzer.calibrated
                    )
                    progress_pct = 100.0 if self.is_calibrated else 95.0
                    if self.is_calibrated:
                        self.session_start_time = time.time()

                self._latest_metrics["is_calibrated"] = self.is_calibrated
                self._latest_metrics["calibration_progress"] = progress_pct
                return annotated, progress_pct, self.is_calibrated
            else:
                annotated = frame.copy()
                return annotated, self._latest_metrics.get("calibration_progress", 0.0), False

    def process_frame(
        self,
        frame: np.ndarray,
        mirror: bool = False,
        draw_dashboard: bool = False,
        mesh_only: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single video frame through the complete analysis pipeline.
        Uses frame-skipping to reduce MediaPipe inference cost.
        Returns:
            (annotated_frame, current_metrics_dict)
        """
        with self.lock:
            if not self.is_running:
                self.start()

            if mirror:
                frame = cv2.flip(frame, 1)

            if getattr(self, "_last_mirror", None) != mirror:
                self._last_mirror = mirror
                self._last_pose_lm = None
                self._last_face_lm = None
                self._last_blendshapes = None
                self._last_hand_lms = []
                self._frame_counter = 0

            self._frame_counter += 1
            run_inference = (self._frame_counter % self.inference_skip) == 0 or (self._last_pose_lm is None)
            run_hand = (self._frame_counter % self.hand_skip) == 0

            if run_inference:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now_ms = int(time.time() * 1000)
                self.frame_ts_ms = max(self.frame_ts_ms + 1, now_ms)

                # Pose + Face every inference frame
                pose_result = self.pose_landmarker.detect_for_video(mp_image, self.frame_ts_ms)
                face_result = self.face_landmarker.detect_for_video(mp_image, self.frame_ts_ms)

                self._last_pose_lm = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                self._last_face_lm = face_result.face_landmarks[0] if face_result.face_landmarks else None
                self._last_blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else None

                # Hand model is heaviest — run less often
                if run_hand:
                    hand_result = self.hand_landmarker.detect_for_video(mp_image, self.frame_ts_ms)
                    self._last_hand_lms = hand_result.hand_landmarks if hand_result.hand_landmarks else []

            # Use cached landmarks (either fresh or from last inference)
            pose_lm = self._last_pose_lm
            face_lm = self._last_face_lm
            blendshapes = self._last_blendshapes
            hand_lms_list = self._last_hand_lms

            # Run analysis + scoring only on inference frames
            if run_inference:
                posture_data = self.posture_analyzer.analyze(pose_lm)
                face_data = self.face_analyzer.analyze(
                    face_lm, blendshapes, (frame.shape[1], frame.shape[0])
                )

                nose_pos = None
                if pose_lm and getattr(pose_lm[0], "visibility", 1.0) > 0.5:
                    nose_pos = (pose_lm[0].x, pose_lm[0].y)
                elif face_lm:
                    nose_pos = (face_lm[1].x, face_lm[1].y)
                hand_data = self.hand_analyzer.analyze(hand_lms_list, nose_pos)

                scores = self.scorer.update(posture_data, face_data, hand_data)

                session_time = time.time() - self.session_start_time

                warnings = []
                if not pose_lm:
                    warnings.append("Body landmarks lost or occluded")
                if not face_lm:
                    warnings.append("Face not detected")
                if scores.get("reliability", 100.0) < 60:
                    warnings.append("Low visual detection reliability")

                # Update cached snapshot
                self._latest_metrics = {
                    "scores": scores,
                    "posture": posture_data,
                    "face": face_data,
                    "hand": hand_data,
                    "fps": round(self.current_fps, 1),
                    "session_time": round(session_time, 1),
                    "is_calibrated": self.is_calibrated,
                    "calibration_progress": 100.0 if self.is_calibrated else self._latest_metrics.get("calibration_progress", 0.0),
                    "reliability": scores.get("reliability", 100.0),
                    "warnings": warnings,
                }

            # FPS calculation (every frame)
            self.fps_count += 1
            fps_elapsed = time.time() - self.fps_timer
            if fps_elapsed >= 0.5:
                self.current_fps = self.fps_count / fps_elapsed
                self.fps_count = 0
                self.fps_timer = time.time()

            # Always render green mesh overlay (fast OpenCV path)
            if draw_dashboard and not mesh_only:
                rendered_frame = self.visualizer.render(
                    frame, self._latest_metrics.get("scores", {}),
                    self._latest_metrics.get("posture", {}),
                    self._latest_metrics.get("face", {}),
                    self._latest_metrics.get("hand", {}),
                    self.current_fps,
                    time.time() - self.session_start_time,
                    pose_lm, face_lm, hand_lms_list
                )
            else:
                rendered_frame = self.visualizer.draw_mesh_only(
                    frame, pose_lm, face_lm, hand_lms_list
                )

            return rendered_frame, self._latest_metrics

    def get_current_metrics(self) -> Dict[str, Any]:
        """Thread-safe getter for latest metric snapshot. Lock-free read of cached dict."""
        # _latest_metrics is replaced atomically (dict assignment is atomic in CPython).
        # Avoid grabbing the heavy lock on every telemetry poll.
        return dict(self._latest_metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Thread-safe getter for full session visual summary."""
        with self.lock:
            if not self.scorer:
                return {}
            
            raw_summary = self.scorer.get_session_summary()
            
            # Format comprehensive summary dict for reporting
            report = {
                "raw_text": raw_summary,
                "averages": {
                    "overall": self.scorer.get_average("overall"),
                    "posture": self.scorer.get_average("posture"),
                    "eye_contact": self.scorer.get_average("eye_contact"),
                    "head_position": self.scorer.get_average("head_position"),
                    "expression": self.scorer.get_average("expression"),
                    "stability": self.scorer.get_average("stability"),
                    "gestures": self.scorer.get_average("gestures"),
                    "reliability": self.scorer.get_average("reliability"),
                },
                "events": {
                    "good_posture_frames": getattr(self.scorer, "good_posture_frames", 0),
                    "poor_posture_frames": getattr(self.scorer, "poor_posture_frames", 0),
                    "posture_warnings": getattr(self.scorer, "posture_warnings", 0),
                    "posture_corrections": getattr(self.scorer, "posture_corrections", 0),
                    "sustained_gaze_away_events": getattr(self.scorer, "sustained_gaze_away_events", 0),
                    "distraction_events": getattr(self.scorer, "distraction_events", 0),
                    "low_reliability_frames": getattr(self.scorer, "low_reliability_frames", 0),
                    "shoulder_tension_events": getattr(self.scorer, "shoulder_tension_events", 0),
                },
                "strengths": self.scorer.get_strengths() if hasattr(self.scorer, "get_strengths") else [],
                "improvements": self.scorer.get_improvements() if hasattr(self.scorer, "get_improvements") else [],
                "is_calibrated": self.is_calibrated,
            }
            return report

    def stop(self) -> None:
        """Release MediaPipe resources."""
        with self.lock:
            if self.pose_landmarker:
                try:
                    self.pose_landmarker.close()
                except Exception:
                    pass
                self.pose_landmarker = None

            if self.face_landmarker:
                try:
                    self.face_landmarker.close()
                except Exception:
                    pass
                self.face_landmarker = None

            if self.hand_landmarker:
                try:
                    self.hand_landmarker.close()
                except Exception:
                    pass
                self.hand_landmarker = None

            self.is_running = False
