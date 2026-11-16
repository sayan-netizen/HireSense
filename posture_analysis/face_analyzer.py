"""
Face Analyzer Module

Uses MediaPipe Face Mesh landmarks to estimate head orientation,
eye-contact consistency, and facial expression.

Head pose is estimated via ``cv2.solvePnP`` with a generic 3-D face
model and six key landmarks.  Eye contact combines head orientation
with iris position within each eye opening.  Expression detection is
conservative and limited to Positive / Neutral / Tense.
"""

import math
import time
import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict
try:
    from . import config
except (ImportError, ValueError):
    import config

# ─── Face Mesh Landmark Indices ──────────────────────────────────

# 6-point model for solvePnP head-pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# Eye landmarks (person's perspective)
LEFT_EYE_INNER = 133
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_INNER = 362
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374

# Additional eye contour aliases for vertical iris tracking
LEFT_EYE_TOP = 159     # same as LEFT_EYE_UPPER
LEFT_EYE_BOTTOM = 145  # same as LEFT_EYE_LOWER
RIGHT_EYE_TOP = 386    # same as RIGHT_EYE_UPPER
RIGHT_EYE_BOTTOM = 374 # same as RIGHT_EYE_LOWER

# Iris landmarks (available with refine_landmarks=True, 478 total)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Expression landmarks
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# Face boundary for width normalisation
LEFT_FACE = 234
RIGHT_FACE = 454

POSE_2D_INDICES = [NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose Tip
    (0.0, 50.0, 10.0),           # Chin
    (-30.0, -30.0, 30.0),        # Left Eye Outer (viewer left)
    (30.0, -30.0, 30.0),         # Right Eye Outer (viewer right)
    (-20.0, 30.0, 20.0),         # Left Mouth Corner
    (20.0, 30.0, 20.0)          # Right Mouth Corner
], dtype=np.float64)

KEY_LANDMARKS = POSE_2D_INDICES + [
    LEFT_EYE_INNER, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
    RIGHT_EYE_INNER, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_FACE, RIGHT_FACE,
    LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
]


class Point3D:
    """Simple 3D point container for smoothed coordinates."""
    __slots__ = ('x', 'y')
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class FaceAnalyzer:
    """Estimates head orientation, eye contact, and facial expression."""

    def __init__(self) -> None:
        # Calibration
        self.calibrated: bool = False
        self._baseline_yaw: float = 0.0
        self._baseline_pitch: float = 0.0
        self._baseline_roll: float = 0.0
        self._baseline_face_size: float = 0.05 # default fallback
        
        self._std_yaw: float = 0.0
        self._std_pitch: float = 0.0
        self._std_roll: float = 0.0
        self._baseline_iris_ratio: float = 0.5
        self._std_iris_ratio: float = 0.05
        self._baseline_iris_v_ratio: float = 0.5  # vertical iris baseline
        self._std_iris_v_ratio: float = 0.05

        self._cal_yaw: list[float] = []
        self._cal_pitch: list[float] = []
        self._cal_roll: list[float] = []
        self._cal_face_size: list[float] = []
        self._cal_iris: list[float] = []
        self._cal_iris_v: list[float] = []  # vertical iris ratio calibration
        self._cal_frame_count: int = 0  # Track for warmup skip

        # Coordinate EMA state
        self._ema_landmarks: Dict[int, Point3D] = {}

        # EMA-smoothed head pose
        self._ema_yaw: float = 0.0
        self._ema_pitch: float = 0.0
        self._ema_roll: float = 0.0
        self._ema_initialized: bool = False

        # Eye-contact tracking
        self._eye_contact_history: deque = deque(maxlen=config.EYE_CONTACT_WINDOW)
        self._gaze_loss_counter = 0
        self._current_gaze_state = "Direct"
        self._total_frames: int = 0
        self._contact_frames: int = 0
        self._away_frames: int = 0
        self._blink_hold_counter: int = 0  # Blink hold: maintain gaze after blink

        # Score smoothing
        self._ema_head_score: float = 85.0
        self._ema_eye_score: float = 85.0
        self._ema_expression_score: float = 85.0

        # solvePnP warm-start state
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None
        
        # Expression tracking for slower updates
        self._expression_history = deque(maxlen=20)  # was 15: longer window for more stable expression
        self._current_expression = "Neutral"
        
        # Gaze consensus voting
        self._gaze_vote_history: deque = deque(maxlen=config.GAZE_CONSENSUS_WINDOW)

        # Blink rate tracking
        self._start_time: float = time.time()
        self._in_blink: bool = False
        self._blink_start_time: float = 0.0
        self._blink_timestamps: deque = deque(maxlen=300)
        self._total_blinks: int = 0
        self._blink_rate: float = 0.0
        self._blink_status: str = "Normal"

    # ─── Coordinate Smoothing ────────────────────────────────────

    def _get_smoothed_landmarks(self, landmarks) -> Dict[int, Point3D]:
        """Apply EMA to raw coordinates to prevent jitter."""
        result = {}
        for idx in KEY_LANDMARKS:
            try:
                lm = landmarks[idx]
                if idx not in self._ema_landmarks:
                    self._ema_landmarks[idx] = Point3D(lm.x, lm.y)
                else:
                    self._ema_landmarks[idx].x += config.FACE_COORD_EMA_ALPHA * (lm.x - self._ema_landmarks[idx].x)
                    self._ema_landmarks[idx].y += config.FACE_COORD_EMA_ALPHA * (lm.y - self._ema_landmarks[idx].y)
                result[idx] = self._ema_landmarks[idx]
            except (IndexError, KeyError):
                continue
        return result

    # ─── Calibration ─────────────────────────────────────────────

    def calibrate_frame(self, landmarks, image_shape: Tuple[int, int]) -> None:
        """Collect one calibration frame (skips warmup frames)."""
        if landmarks is None:
            return

        self._cal_frame_count += 1
            
        smoothed = self._get_smoothed_landmarks(landmarks)
        yaw, pitch, roll = self._estimate_head_pose(smoothed, image_shape)
        
        # Skip warmup frames (noisy EMA initialization)
        if self._cal_frame_count <= config.CALIBRATION_WARMUP_SKIP:
            return
        
        # Measure face size
        try:
            face_size = abs(smoothed[RIGHT_FACE].x - smoothed[LEFT_FACE].x)
        except KeyError:
            face_size = 0.05
            
        if yaw is not None:
            self._cal_yaw.append(yaw)
            self._cal_pitch.append(pitch)
            self._cal_roll.append(roll)
            self._cal_face_size.append(face_size)

        try:
            left_r = self._iris_horizontal_ratio(smoothed, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_IRIS_CENTER)
            right_r = self._iris_horizontal_ratio(smoothed, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS_CENTER)
            self._cal_iris.append((left_r + right_r) / 2.0)
            # Vertical iris ratio
            left_vr = self._iris_vertical_ratio(smoothed, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER)
            right_vr = self._iris_vertical_ratio(smoothed, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER)
            self._cal_iris_v.append((left_vr + right_vr) / 2.0)
        except (KeyError, IndexError):
            pass

    @staticmethod
    def _trimmed_stats(data: list[float]) -> tuple[float, float]:
        """Compute trimmed median and std, removing outlier percentiles."""
        if len(data) < 5:
            return float(np.median(data)), float(np.std(data))
        trim_pct = config.CALIBRATION_OUTLIER_TRIM
        arr = np.array(data)
        lo = np.percentile(arr, trim_pct * 100)
        hi = np.percentile(arr, (1.0 - trim_pct) * 100)
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if len(trimmed) < 3:
            trimmed = arr
        return float(np.median(trimmed)), float(np.std(trimmed))

    def finish_calibration(self) -> None:
        """Compute head-pose and iris baselines from trimmed calibration data."""
        if not self._cal_yaw and not self._cal_iris:
            return

        if self._cal_yaw:
            self._baseline_yaw, self._std_yaw = self._trimmed_stats(self._cal_yaw)
            self._baseline_pitch, self._std_pitch = self._trimmed_stats(self._cal_pitch)
            self._baseline_roll, self._std_roll = self._trimmed_stats(self._cal_roll)
            self._baseline_face_size = float(np.median(self._cal_face_size))
        if self._cal_iris:
            self._baseline_iris_ratio, self._std_iris_ratio = self._trimmed_stats(self._cal_iris)
        if self._cal_iris_v:
            self._baseline_iris_v_ratio, self._std_iris_v_ratio = self._trimmed_stats(self._cal_iris_v)
        
        self.calibrated = True

    # ─── Per-Frame Analysis ──────────────────────────────────────

    def analyze(self, landmarks, blendshapes, image_shape: Tuple[int, int]) -> dict:
        """Analyze face cues for the current frame."""
        reliability = self._calculate_reliability(landmarks, image_shape)
        
        empty = self._empty_result(reliability, insufficient_data=(reliability < 40.0))

        if landmarks is None or reliability < 40.0:
            self._total_frames += 1
            self._eye_contact_history.append(False)
            return empty

        smoothed = self._get_smoothed_landmarks(landmarks)

        # ── Head pose ─────────────────────────────────────────
        yaw, pitch, roll = self._estimate_head_pose(smoothed, image_shape)
        pose_valid = yaw is not None

        if pose_valid:
            # EMA smoothing with per-frame jump rejection
            if not self._ema_initialized:
                self._ema_yaw, self._ema_pitch, self._ema_roll = yaw, pitch, roll
                self._ema_initialized = True
            else:
                # Slew-rate limit per-frame delta to suppress solvePnP 1-frame spikes
                # while allowing fast, continuous tracking of real head movements
                max_jump = config.HEAD_POSE_MAX_JUMP
                dy_step = float(np.clip(yaw - self._ema_yaw, -max_jump, max_jump))
                dp_step = float(np.clip(pitch - self._ema_pitch, -max_jump, max_jump))
                dr_step = float(np.clip(roll - self._ema_roll, -max_jump, max_jump))

                pose_alpha = config.FACE_POSE_EMA_ALPHA
                self._ema_yaw += pose_alpha * dy_step
                self._ema_pitch += pose_alpha * dp_step
                self._ema_roll += pose_alpha * dr_step
        elif not self._ema_initialized:
            self._ema_yaw, self._ema_pitch, self._ema_roll = 0.0, 0.0, 0.0
            self._ema_initialized = True

        # Deltas from baseline
        b_yaw = self._baseline_yaw if self.calibrated else 0.0
        b_pitch = self._baseline_pitch if self.calibrated else 0.0
        b_roll = self._baseline_roll if self.calibrated else 0.0

        dy = abs(self._ema_yaw - b_yaw)
        dp = abs(self._ema_pitch - b_pitch)
        dr = abs(self._ema_roll - b_roll)

        result: dict = {
            "detected": True,
            "reliability": reliability,
            "head_yaw": round(self._ema_yaw, 1),
            "head_pitch": round(self._ema_pitch, 1),
            "head_roll": round(self._ema_roll, 1),
        }

        # Classify head position
        if not pose_valid and not self.calibrated:
            head_raw = 40.0
        else:
            # Dynamic thresholds if calibrated
            if self.calibrated and self._std_yaw > 1.0:
                t_yaw_s = max(config.YAW_SLIGHT, self._std_yaw * config.STD_MULTIPLIER_SLIGHT)
                t_yaw_sig = max(config.YAW_SIGNIFICANT, self._std_yaw * config.STD_MULTIPLIER_MODERATE)
                t_pitch_s = max(config.PITCH_SLIGHT, self._std_pitch * config.STD_MULTIPLIER_SLIGHT)
                t_pitch_sig = max(config.PITCH_SIGNIFICANT, self._std_pitch * config.STD_MULTIPLIER_MODERATE)
                t_roll_s = max(config.ROLL_SLIGHT, self._std_roll * config.STD_MULTIPLIER_SLIGHT)
                t_roll_sig = max(config.ROLL_SIGNIFICANT, self._std_roll * config.STD_MULTIPLIER_MODERATE)
            else:
                t_yaw_s, t_yaw_sig = config.YAW_SLIGHT, config.YAW_SIGNIFICANT
                t_pitch_s, t_pitch_sig = config.PITCH_SLIGHT, config.PITCH_SIGNIFICANT
                t_roll_s, t_roll_sig = config.ROLL_SLIGHT, config.ROLL_SIGNIFICANT

            # Continuous gradient scoring based on angular deviation
            # Each axis contributes a penalty proportional to how far past the "slight" threshold it is
            # Yaw (left-right) and pitch (up-down) are weighted more than roll (head tilt)
            def _axis_score(delta: float, t_slight: float, t_sig: float, weight: float) -> float:
                if delta < t_slight * 0.5:
                    return 100.0 * weight  # well within tolerance
                elif delta < t_slight:
                    # Slight penalty: 100 → 85 linearly
                    frac = (delta - t_slight * 0.5) / (t_slight * 0.5)
                    return (100.0 - frac * 15.0) * weight
                elif delta < t_sig:
                    # Moderate penalty: 85 → 35 linearly
                    frac = (delta - t_slight) / max(1e-6, t_sig - t_slight)
                    return (85.0 - frac * 50.0) * weight
                else:
                    # Beyond significant: 35 → 10 with diminishing returns
                    over = delta - t_sig
                    return max(10.0, 35.0 - over * 1.5) * weight

            # Weights: yaw 40%, pitch 40%, roll 20%
            head_raw = (
                _axis_score(dy, t_yaw_s, t_yaw_sig, 0.4)
                + _axis_score(dp, t_pitch_s, t_pitch_sig, 0.4)
                + _axis_score(dr, t_roll_s, t_roll_sig, 0.2)
            )
            head_raw = float(np.clip(head_raw, 5.0, 100.0))

        # ── Eye contact ───────────────────────────────────────
        left_ear = self._eye_aspect_ratio(landmarks, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_UPPER, LEFT_EYE_LOWER)
        right_ear = self._eye_aspect_ratio(landmarks, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
        is_blinking = (left_ear + right_ear) / 2.0 < config.BLINK_EAR_THRESHOLD
        
        raw_looking, gaze_type = self._estimate_eye_contact(smoothed, dy, dp)
        
        # ── Blink rate tracking ───────────────────────────────
        now = time.time()
        if is_blinking:
            if not self._in_blink:
                self._in_blink = True
                self._blink_start_time = now
            self._blink_hold_counter = config.BLINK_HOLD_FRAMES
            raw_looking = (self._current_gaze_state in ("Direct", "Near-Camera"))
        else:
            if self._in_blink:
                self._in_blink = False
                blink_dur = now - self._blink_start_time
                if config.BLINK_MIN_DURATION <= blink_dur <= config.BLINK_MAX_DURATION:
                    self._total_blinks += 1
                    self._blink_timestamps.append(now)

            if self._blink_hold_counter > 0:
                self._blink_hold_counter -= 1
                raw_looking = (self._current_gaze_state in ("Direct", "Near-Camera"))

        # Rolling blink rate (blinks per minute over config.BLINK_WINDOW_SEC)
        while self._blink_timestamps and (now - self._blink_timestamps[0] > config.BLINK_WINDOW_SEC):
            self._blink_timestamps.popleft()

        elapsed = max(5.0, min(config.BLINK_WINDOW_SEC, now - self._start_time))
        self._blink_rate = round((len(self._blink_timestamps) / elapsed) * 60.0, 1)

        if self._blink_rate >= config.BLINK_ELEVATED:
            self._blink_status = "Elevated (Stress)"
        elif self._blink_rate <= config.BLINK_LOW and (now - self._start_time) >= 20.0:
            self._blink_status = "Low (Staring)"
        else:
            self._blink_status = "Normal"
            
        # Gaze consensus voting: require N consistent frames before state change
        self._gaze_vote_history.append(raw_looking)
        
        if raw_looking:
            self._gaze_loss_counter = 0
            self._away_frames = 0
            self._current_gaze_state = gaze_type
        else:
            self._gaze_loss_counter += 1
            self._away_frames += 1
            # Only change state if majority of recent votes agree
            recent_away_ratio = 1.0 - (sum(self._gaze_vote_history) / max(1, len(self._gaze_vote_history)))
            if self._gaze_loss_counter >= config.GAZE_HYSTERESIS_FRAMES and recent_away_ratio >= 0.7:
                if self._away_frames >= config.SUSTAINED_AWAY_FRAMES:
                    self._current_gaze_state = "Sustained Away"
                else:
                    self._current_gaze_state = "Occasional Distraction"
                
        self._total_frames += 1
        is_contact = self._current_gaze_state in ("Direct", "Near-Camera")
        if is_contact:
            self._contact_frames += 1
        self._eye_contact_history.append(is_contact)

        recent = sum(self._eye_contact_history) / max(1, len(self._eye_contact_history))
        overall = self._contact_frames / max(1, self._total_frames)
        eye_pct = round(recent * 100, 1)

        result["eye_contact_pct"] = eye_pct
        result["eye_contact_overall_pct"] = round(overall * 100, 1)

        eye_raw = float(np.clip(eye_pct, 0, 100))

        # ── Expression ────────────────────────────────────────
        expression_raw, expr_score_raw = self._detect_expression(blendshapes)
        self._expression_history.append(expression_raw)
        
        # Only change expression if it's the majority in recent history
        if len(self._expression_history) == self._expression_history.maxlen:
            counts = {
                "Neutral": self._expression_history.count("Neutral"),
                "Positive": self._expression_history.count("Positive"),
                "Tense": self._expression_history.count("Tense"),
                "Engaged": self._expression_history.count("Engaged")
            }
            majority_expr = max(counts, key=counts.get)
            # Require a strong majority to switch state from Neutral
            if counts[majority_expr] > 9:
                self._current_expression = majority_expr
                
        result["expression"] = self._current_expression

        # ── Score smoothing ───────────────────────────────────
        self._ema_head_score += config.FACE_SCORE_EMA_ALPHA * (head_raw - self._ema_head_score)
        self._ema_eye_score += config.FACE_SCORE_EMA_ALPHA * (eye_raw - self._ema_eye_score)
        self._ema_expression_score += config.EXPRESSION_SCORE_EMA_ALPHA * (expr_score_raw - self._ema_expression_score)

        result["head_score"] = round(float(np.clip(self._ema_head_score, 0, 100)), 1)
        result["eye_score"] = round(float(np.clip(self._ema_eye_score, 0, 100)), 1)
        result["expression_score"] = round(float(np.clip(self._ema_expression_score, 0, 100)), 1)
        
        # Textual status strings pass-through (hysteresis is applied in scoring.py)
        result["head_position"] = "Centered" if result["head_score"] >= 80 else "Tilted" if result["head_score"] >= 40 else "Looking Away"
        result["eye_contact"] = self._current_gaze_state
        result["blink_rate"] = self._blink_rate
        result["blink_count"] = self._total_blinks
        result["blink_status"] = self._blink_status
        
        return result

    # ─── Helpers ─────────────────────────────────────────────────

    def _calculate_reliability(self, landmarks, image_shape: Tuple[int, int]) -> float:
        """Calculate face detection reliability based on presence, tracking bounds, and face scale."""
        if landmarks is None or len(landmarks) == 0:
            return 0.0
            
        try:
            lx = landmarks[LEFT_FACE].x
            rx = landmarks[RIGHT_FACE].x
            face_width = abs(rx - lx)
            
            # Check if nose and face corners are within image bounds
            nose_p = landmarks[NOSE_TIP]
            if not (0.05 <= nose_p.x <= 0.95 and 0.05 <= nose_p.y <= 0.95):
                return 50.0  # Face partially cut off by camera frame edge
                
            expected_width = self._baseline_face_size if self.calibrated else 0.05
            if face_width < expected_width * 0.4:
                return max(30.0, 100.0 * (face_width / max(1e-4, expected_width * 0.4)))
                
            return 100.0
        except (IndexError, KeyError):
            return 85.0

    # ─── Head Pose Estimation ────────────────────────────────────

    def _estimate_head_pose(
        self, lm: Dict[int, Point3D], image_shape: Tuple[int, int]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        w, h = image_shape
        try:
            image_points = np.array(
                [(lm[idx].x * w, lm[idx].y * h) for idx in POSE_2D_INDICES],
                dtype=np.float64,
            )
        except KeyError:
            return None, None, None

        # Approximate camera intrinsics
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Adaptive 3D model points: in mirrored feed, landmark 33 (person's right eye) has larger X than 263
        is_mirrored = image_points[2, 0] > image_points[3, 0]
        if is_mirrored:
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose Tip (1)
                (0.0, 50.0, -10.0),          # Chin (152)
                (30.0, -30.0, -30.0),        # 33 on screen right
                (-30.0, -30.0, -30.0),       # 263 on screen left
                (20.0, 30.0, -20.0),         # 61 on screen right
                (-20.0, 30.0, -20.0)         # 291 on screen left
            ], dtype=np.float64)
        else:
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose Tip (1)
                (0.0, 50.0, -10.0),          # Chin (152)
                (-30.0, -30.0, -30.0),       # 33 on screen left
                (30.0, -30.0, -30.0),        # 263 on screen right
                (-20.0, 30.0, -20.0),        # 61 on screen left
                (20.0, 30.0, -20.0)          # 291 on screen right
            ], dtype=np.float64)

        try:
            use_guess = self._rvec is not None
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                rvec=self._rvec, tvec=self._tvec,
                useExtrinsicGuess=use_guess, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            self._rvec = None
            self._tvec = None
            return None, None, None

        if not success:
            self._rvec = None
            self._tvec = None
            return None, None, None

        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles from the rotation matrix
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = 0.0
            
        # Reject unrealistic outlier jumps (solvePnP instability)
        if abs(yaw) > 85.0 or abs(pitch) > 85.0 or abs(roll) > 85.0:
            self._rvec = None
            self._tvec = None
            return None, None, None

        # Cache valid extrinsic guess for next frame
        self._rvec = rvec
        self._tvec = tvec
        return yaw, pitch, roll

    # ─── Eye Contact Estimation ──────────────────────────────────

    def _estimate_eye_contact(
        self, lm: Dict[int, Point3D], yaw_delta: float, pitch_delta: float
    ) -> Tuple[bool, str]:
        """Estimate eye contact using head pose + iris position (horizontal and vertical)."""
        # If head is turned significantly, that alone means no eye contact
        if yaw_delta > config.YAW_SIGNIFICANT or pitch_delta > config.PITCH_SIGNIFICANT:
            return False, "Sustained Away" if yaw_delta > config.YAW_SIGNIFICANT * 1.5 else "Occasional Distraction"

        try:
            left_ratio = self._iris_horizontal_ratio(
                lm, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_IRIS_CENTER
            )
            right_ratio = self._iris_horizontal_ratio(
                lm, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS_CENTER
            )
        except KeyError:
            # Without iris data we CANNOT confirm eye contact — be conservative
            # Only consider it "Direct" if head is nearly perfectly centered
            head_very_centered = yaw_delta < config.YAW_SLIGHT * 0.5 and pitch_delta < config.PITCH_SLIGHT * 0.5
            return head_very_centered, "Direct" if head_very_centered else "Occasional Distraction"

        avg_ratio = (left_ratio + right_ratio) / 2.0
        b_iris = self._baseline_iris_ratio if self.calibrated else 0.5
        iris_h_delta = abs(avg_ratio - b_iris)
        iris_h_centered = iris_h_delta < config.IRIS_CENTER_TOLERANCE

        # Vertical iris ratio: detect looking up/down
        try:
            left_v = self._iris_vertical_ratio(lm, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER)
            right_v = self._iris_vertical_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER)
            avg_v_ratio = (left_v + right_v) / 2.0
            b_iris_v = self._baseline_iris_v_ratio if self.calibrated else 0.5
            iris_v_delta = abs(avg_v_ratio - b_iris_v)
            iris_v_centered = iris_v_delta < config.IRIS_CENTER_TOLERANCE * 1.2  # slightly more lenient for vertical
        except KeyError:
            iris_v_centered = True  # if vertical data unavailable, don't penalize
            iris_v_delta = 0.0

        iris_centered = iris_h_centered and iris_v_centered

        # Near camera: pitch slightly down, but iris centered on camera
        is_near = (config.PITCH_SLIGHT <= pitch_delta < config.PITCH_SIGNIFICANT) and iris_centered
        is_direct = (yaw_delta < config.YAW_SLIGHT and pitch_delta < config.PITCH_SLIGHT) and iris_centered
        
        if is_direct:
            return True, "Direct"
        elif is_near:
            return True, "Near-Camera"
        elif iris_centered and yaw_delta < config.YAW_SLIGHT * 1.5:
            # Iris is centered but head is slightly turned — still acceptable
            return True, "Near-Camera"
        else:
            return False, "Occasional Distraction"

    @staticmethod
    def _eye_aspect_ratio(lm: Dict[int, Point3D], p_outer: int, p_inner: int, p_upper: int, p_lower: int) -> float:
        try:
            h = math.dist((lm[p_upper].x, lm[p_upper].y), (lm[p_lower].x, lm[p_lower].y))
            w = math.dist((lm[p_outer].x, lm[p_outer].y), (lm[p_inner].x, lm[p_inner].y))
            if w < 1e-6:
                return 0.3
            return h / w
        except KeyError:
            return 0.3

    @staticmethod
    def _iris_horizontal_ratio(
        lm: Dict[int, Point3D], corner_a_idx: int, corner_b_idx: int, iris_idx: int
    ) -> float:
        ax = lm[corner_a_idx].x
        bx = lm[corner_b_idx].x
        ix = lm[iris_idx].x
        width = abs(bx - ax)
        if width < 1e-6:
            return 0.5
        return (ix - min(ax, bx)) / width

    @staticmethod
    def _iris_vertical_ratio(
        lm: Dict[int, Point3D], top_idx: int, bottom_idx: int, iris_idx: int
    ) -> float:
        """Vertical iris position: 0.0 = top of eye, 1.0 = bottom of eye, 0.5 = centered."""
        ty = lm[top_idx].y
        by = lm[bottom_idx].y
        iy = lm[iris_idx].y
        height = abs(by - ty)
        if height < 1e-6:
            return 0.5
        return (iy - min(ty, by)) / height

    # ─── Expression Detection ────────────────────────────────────

    def _detect_expression(self, blendshapes) -> Tuple[str, float]:
        if not blendshapes:
            return "Neutral", 70.0
            
        # MediaPipe blendshapes is a list of categories
        b_dict = {cat.category_name: cat.score for cat in blendshapes}
        
        # Positive / Engaged indicators
        smile_l = b_dict.get("mouthSmileLeft", 0.0)
        smile_r = b_dict.get("mouthSmileRight", 0.0)
        smile_score = (smile_l + smile_r) / 2.0
        
        # Cheek raise strongly correlates with genuine (Duchenne) smile
        cheek_l = b_dict.get("cheekSquintLeft", 0.0)
        cheek_r = b_dict.get("cheekSquintRight", 0.0)
        cheek_raise = (cheek_l + cheek_r) / 2.0
        
        # Combined smile score with cheek raise for better accuracy
        genuine_smile = smile_score * 0.7 + cheek_raise * 0.3
        
        # Tense / Stressed indicators
        frown_l = b_dict.get("mouthFrownLeft", 0.0)
        frown_r = b_dict.get("mouthFrownRight", 0.0)
        brow_down_l = b_dict.get("browDownLeft", 0.0)
        brow_down_r = b_dict.get("browDownRight", 0.0)
        mouth_press_l = b_dict.get("mouthPressLeft", 0.0)
        mouth_press_r = b_dict.get("mouthPressRight", 0.0)
        # Weighted combination: frowning + brow furrowing + lip pressing
        tense_score = max(
            (frown_l + frown_r) / 2.0,
            (brow_down_l + brow_down_r) / 2.0,
            (mouth_press_l + mouth_press_r) / 2.0 * 0.6  # lip press is less reliable
        )
        
        # Surprised / Alert indicators (engaged in conversation)
        brow_inner_up = b_dict.get("browInnerUp", 0.0)
        brow_outer_up_l = b_dict.get("browOuterUpLeft", 0.0)
        brow_outer_up_r = b_dict.get("browOuterUpRight", 0.0)
        brow_up = max(brow_inner_up, (brow_outer_up_l + brow_outer_up_r) / 2.0)
        
        # Classification with higher thresholds to avoid false positives
        if genuine_smile > 0.15:  # was 0.2 for plain smile: cheek-raise combo is more reliable
            return "Positive", min(100.0, 85.0 + genuine_smile * 30.0)
        elif tense_score > 0.35:  # was 0.3: higher threshold reduces false tense detections
            return "Tense", max(30.0, 70.0 - tense_score * 35.0)
        elif brow_up > 0.35:  # was 0.3: require stronger evidence of engagement
            return "Engaged", 90.0
        else:
            return "Neutral", 85.0

    @staticmethod
    def _empty_result(reliability: float, insufficient_data: bool = False) -> dict:
        return {
            "detected": False,
            "insufficient_data": insufficient_data,
            "reliability": reliability,
            "head_position": "Unknown",
            "eye_contact": "Unknown",
            "eye_contact_pct": 0.0,
            "eye_contact_overall_pct": 0.0,
            "expression": "Unknown",
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "head_score": 0.0,
            "eye_score": 0.0,
            "expression_score": 0.0,
            "blink_rate": 0.0,
            "blink_count": 0,
            "blink_status": "Unknown",
        }
