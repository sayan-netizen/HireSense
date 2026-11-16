"""
Posture Analyzer Module

Analyzes body posture using MediaPipe Pose landmarks.
Detects sitting posture quality, shoulder alignment, body lean,
head alignment, and movement stability.

All geometric calculations use smoothed, normalized landmark coordinates
so the system is resolution-independent and jitter-free.
"""

import math
import numpy as np
from collections import deque
from typing import Optional, Dict

try:
    from . import config
except (ImportError, ValueError):
    import config

# ─── MediaPipe Pose Landmark Indices ─────────────────────────────
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

KEY_LANDMARKS = (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
RELIABILITY_LANDMARKS = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)

VISIBILITY_THRESHOLD = 0.5      # minimum landmark visibility to consider detected


class Point3D:
    """Simple 3D point container for smoothed coordinates."""
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class PostureAnalyzer:
    """Analyzes body posture from MediaPipe Pose landmarks."""

    def __init__(self) -> None:
        # Calibration state
        self.calibrated: bool = False
        
        # Baselines
        self._baseline_shoulder_angle: float = 0.0
        self._baseline_lateral_angle: float = 0.0
        self._baseline_head_offset: float = 0.0
        self._baseline_shoulder_width: float = 0.0
        self._baseline_neck_ratio: float = 0.42
        self._baseline_torso_ratio: Optional[float] = None
        self._baseline_nose_y: float = 0.0
        self._baseline_shoulder_z: float = 0.0
        self._baseline_forward_head_z: float = 0.0
        self._baseline_ear_shoulder_ratio: float = 0.35
        self._baseline_sagittal_angle: Optional[float] = None
        
        # Standard deviations from calibration
        self._std_shoulder_angle: float = 0.0
        self._std_lateral_angle: float = 0.0
        self._std_head_offset: float = 0.0
        self._std_shoulder_width: float = 0.0
        self._std_neck_ratio: float = 0.0
        self._std_nose_y: float = 0.0
        self._std_shoulder_z: float = 0.0
        self._std_ear_shoulder_ratio: float = 0.0

        # Calibration buffers
        self._cal_shoulder: list[float] = []
        self._cal_lateral: list[float] = []
        self._cal_head: list[float] = []
        self._cal_shoulder_width: list[float] = []
        self._cal_neck_ratio: list[float] = []
        self._cal_torso_ratio: list[float] = []
        self._cal_nose_y: list[float] = []
        self._cal_shoulder_z: list[float] = []
        self._cal_forward_head: list[float] = []
        self._cal_ear_shoulder: list[float] = []
        self._cal_sagittal: list[float] = []
        self._cal_frame_count: int = 0
        self._tension_counter: int = 0

        # Coordinate EMA state
        self._ema_landmarks: Dict[int, Point3D] = {}

        # Output score EMA state
        self._ema_posture_score: float = 85.0
        self._ema_shoulder_score: float = 90.0

        # Stability & persistence tracking
        self._position_history: deque = deque(maxlen=config.POSTURE_STABILITY_WINDOW)
        self._lean_history: deque = deque(maxlen=config.POSTURE_LEAN_HISTORY_FRAMES) 
        
        # Posture score history for multi-frame consensus
        self._posture_label_history: deque = deque(maxlen=config.POSTURE_CONSENSUS_WINDOW)

        # Flag for rising-edge posture correction detection
        self._in_poor_posture: bool = False

    # ─── Coordinate Smoothing ────────────────────────────────────

    def _get_smoothed_landmarks(self, landmarks) -> Optional[Dict[int, Point3D]]:
        """Apply EMA to raw coordinates to prevent jitter."""
        if not self._key_landmarks_visible(landmarks):
            return None

        result = {}
        for idx in KEY_LANDMARKS:
            lm = landmarks[idx]
            
            # Reject invalid coordinates (NaN/Inf)
            if math.isnan(lm.x) or math.isinf(lm.x) or math.isnan(lm.y) or math.isinf(lm.y):
                return None
                
            if idx not in self._ema_landmarks:
                self._ema_landmarks[idx] = Point3D(lm.x, lm.y, lm.z)
            else:
                self._ema_landmarks[idx].x += config.POSTURE_COORD_EMA_ALPHA * (lm.x - self._ema_landmarks[idx].x)
                self._ema_landmarks[idx].y += config.POSTURE_COORD_EMA_ALPHA * (lm.y - self._ema_landmarks[idx].y)
                self._ema_landmarks[idx].z += config.POSTURE_COORD_EMA_ALPHA * (lm.z - self._ema_landmarks[idx].z)
            result[idx] = self._ema_landmarks[idx]
        
        return result

    # ─── Calibration ─────────────────────────────────────────────

    def calibrate_frame(self, landmarks) -> None:
        """Collect one frame of calibration data (skips warmup frames)."""
        if landmarks is None:
            return
        
        self._cal_frame_count += 1
        smoothed = self._get_smoothed_landmarks(landmarks)
        if smoothed is None:
            return

        # Skip initial warmup frames (noisy EMA initialization)
        if self._cal_frame_count <= config.CALIBRATION_WARMUP_SKIP:
            return

        sh_w = self._shoulder_width(smoothed)
        if sh_w < 1e-4:
            return

        self._cal_shoulder.append(self._shoulder_angle(smoothed))
        self._cal_lateral.append(self._lateral_angle(smoothed))
        self._cal_head.append(self._head_offset(smoothed))
        self._cal_shoulder_width.append(sh_w)
        self._cal_neck_ratio.append(self._neck_ratio(smoothed, sh_w))
        self._cal_nose_y.append(smoothed[NOSE].y)
        self._cal_shoulder_z.append((smoothed[LEFT_SHOULDER].z + smoothed[RIGHT_SHOULDER].z) / 2.0)
        self._cal_forward_head.append(self._forward_head(smoothed, sh_w))
        self._cal_ear_shoulder.append(self._ear_shoulder_ratio(smoothed, sh_w))

        if self._hips_usable(smoothed):
            sag = self._sagittal_angle(smoothed)
            if sag is not None:
                self._cal_sagittal.append(sag)
            tr = self._torso_ratio(smoothed, sh_w)
            if tr is not None:
                self._cal_torso_ratio.append(tr)

    @staticmethod
    def _trimmed_stats(data: list[float]) -> tuple[float, float]:
        """Compute trimmed median and std, removing top/bottom outlier percentiles."""
        if len(data) < 5:
            return float(np.median(data)), float(np.std(data))
        trim_pct = config.CALIBRATION_OUTLIER_TRIM
        arr = np.array(data)
        lo = np.percentile(arr, trim_pct * 100)
        hi = np.percentile(arr, (1.0 - trim_pct) * 100)
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if len(trimmed) < 3:
            trimmed = arr  # Fallback if too aggressively trimmed
        return float(np.median(trimmed)), float(np.std(trimmed))

    def finish_calibration(self) -> None:
        """Compute baseline values and standard deviations from trimmed calibration data."""
        if not self._cal_shoulder_width:
            return

        self._baseline_shoulder_angle, self._std_shoulder_angle = self._trimmed_stats(self._cal_shoulder)
        self._baseline_lateral_angle, self._std_lateral_angle = self._trimmed_stats(self._cal_lateral)
        self._baseline_head_offset, self._std_head_offset = self._trimmed_stats(self._cal_head)
        self._baseline_shoulder_width, self._std_shoulder_width = self._trimmed_stats(self._cal_shoulder_width)
        self._baseline_neck_ratio, self._std_neck_ratio = self._trimmed_stats(self._cal_neck_ratio)
        self._baseline_nose_y, self._std_nose_y = self._trimmed_stats(self._cal_nose_y)
        self._baseline_shoulder_z, self._std_shoulder_z = self._trimmed_stats(self._cal_shoulder_z)
        self._baseline_forward_head_z, _ = self._trimmed_stats(self._cal_forward_head)
        if self._cal_ear_shoulder:
            self._baseline_ear_shoulder_ratio, self._std_ear_shoulder_ratio = self._trimmed_stats(self._cal_ear_shoulder)

        if self._cal_sagittal:
            self._baseline_sagittal_angle, _ = self._trimmed_stats(self._cal_sagittal)
        if self._cal_torso_ratio:
            self._baseline_torso_ratio, _ = self._trimmed_stats(self._cal_torso_ratio)

        self.calibrated = True

    # ─── Per-Frame Analysis ──────────────────────────────────────

    def analyze(self, landmarks) -> dict:
        """Analyze posture for the current frame with multi-cue forward slouch & lean detection."""
        if landmarks is None:
            return self._empty_result(0.0)

        reliability = self._calculate_reliability(landmarks)
        if reliability < 40.0:
            return self._empty_result(reliability, insufficient_data=True)
            
        smoothed = self._get_smoothed_landmarks(landmarks)
        if smoothed is None:
            return self._empty_result(reliability, insufficient_data=True)

        # ── Current geometric measurements ──────────────────
        sh_w = self._shoulder_width(smoothed)
        if sh_w < 1e-4:
            return self._empty_result(reliability, insufficient_data=True)

        shoulder_angle = self._shoulder_angle(smoothed)
        lat_angle = self._lateral_angle(smoothed)
        head_offset = self._head_offset(smoothed)
        neck_r = self._neck_ratio(smoothed, sh_w)
        cur_nose_y = smoothed[NOSE].y
        cur_sh_z = (smoothed[LEFT_SHOULDER].z + smoothed[RIGHT_SHOULDER].z) / 2.0
        fwd_head_z = self._forward_head(smoothed, sh_w)

        # ── Shoulder alignment (Coronal Tilt) ─────────────────
        if self.calibrated:
            shoulder_delta = abs(shoulder_angle - self._baseline_shoulder_angle)
            t_sh_slight = max(config.SHOULDER_TILT_SLIGHT, self._std_shoulder_angle * config.STD_MULTIPLIER_SLIGHT)
            t_sh_sig = max(config.SHOULDER_TILT_SIGNIFICANT, self._std_shoulder_angle * config.STD_MULTIPLIER_MODERATE)
        else:
            shoulder_delta = abs(shoulder_angle)
            t_sh_slight = config.SHOULDER_TILT_SLIGHT
            t_sh_sig = config.SHOULDER_TILT_SIGNIFICANT

        if shoulder_delta < t_sh_slight:
            shoulder_status = "Balanced"
            sh_score = 100.0
        elif shoulder_delta < t_sh_sig:
            shoulder_status = "Slight Tilt"
            frac = (shoulder_delta - t_sh_slight) / max(1e-4, t_sh_sig - t_sh_slight)
            sh_score = 100.0 - frac * 40.0
        else:
            shoulder_status = "Significant Tilt"
            sh_score = max(20.0, 60.0 - (shoulder_delta - t_sh_sig) * 2.5)

        # ── Lateral (Side-to-Side) Lean ───────────────────────
        if self.calibrated:
            lat_delta = lat_angle - self._baseline_lateral_angle
            t_lat_slight = max(config.LATERAL_LEAN_SLIGHT, self._std_lateral_angle * config.STD_MULTIPLIER_SLIGHT)
            t_lat_mod = max(config.LATERAL_LEAN_MODERATE, self._std_lateral_angle * config.STD_MULTIPLIER_MODERATE)
            t_lat_exc = max(config.LATERAL_LEAN_EXCESSIVE, self._std_lateral_angle * config.STD_MULTIPLIER_EXCESSIVE)
        else:
            lat_delta = lat_angle
            t_lat_slight = config.LATERAL_LEAN_SLIGHT
            t_lat_mod = config.LATERAL_LEAN_MODERATE
            t_lat_exc = config.LATERAL_LEAN_EXCESSIVE

        abs_lat = abs(lat_delta)

        # ── Forward Lean & Proximity Detection ────────────────
        if self.calibrated and self._baseline_shoulder_width > 1e-4:
            prox_ratio = sh_w / self._baseline_shoulder_width
            delta_sh_z = cur_sh_z - self._baseline_shoulder_z
            if self._baseline_sagittal_angle is not None and self._hips_usable(smoothed):
                sag = self._sagittal_angle(smoothed)
                delta_sag = (sag - self._baseline_sagittal_angle) if sag is not None else 0.0
            else:
                delta_sag = 0.0
        else:
            prox_ratio = 1.0
            delta_sh_z = 0.0
            delta_sag = 0.0

        # Proximity expansion contribution
        f_prox = max(0.0, (prox_ratio - config.FORWARD_PROXIMITY_SLIGHT) / max(1e-4, config.FORWARD_PROXIMITY_EXCESSIVE - config.FORWARD_PROXIMITY_SLIGHT))
        # Z-depth forward shift contribution (negative delta = closer)
        f_z = max(0.0, (-delta_sh_z - (-config.FORWARD_SHOULDER_Z_SLIGHT)) / max(1e-4, (-config.FORWARD_SHOULDER_Z_MODERATE) - (-config.FORWARD_SHOULDER_Z_SLIGHT)))
        # Sagittal pitch angle contribution
        f_sag = max(0.0, (delta_sag - 7.0) / 16.0) if delta_sag > 0 else 0.0

        if self._hips_usable(smoothed) and self._baseline_sagittal_angle is not None:
            fwd_lean_level = f_prox * 0.50 + f_z * 0.25 + f_sag * 0.25
        else:
            fwd_lean_level = f_prox * 0.65 + f_z * 0.35

        # Backward lean (retreating / leaning back)
        b_prox = max(0.0, (config.BACKWARD_PROXIMITY_SLIGHT - prox_ratio) / max(1e-4, config.BACKWARD_PROXIMITY_SLIGHT - config.BACKWARD_PROXIMITY_MODERATE))
        b_z = max(0.0, (delta_sh_z - 0.04) / 0.10)
        bwd_lean_level = b_prox * 0.70 + b_z * 0.30

        # ── Slouching & Spinal Hunching Detection ─────────────
        # 1. Neck vertical compression: when slouching, head sinks / shoulders roll up
        if self.calibrated and self._baseline_neck_ratio > 1e-4:
            neck_comp = (self._baseline_neck_ratio - neck_r) / self._baseline_neck_ratio
            head_drop = (cur_nose_y - self._baseline_nose_y) / max(1e-4, self._baseline_shoulder_width)
            delta_fwd_head = fwd_head_z - self._baseline_forward_head_z
            
            if self._baseline_torso_ratio is not None and self._hips_usable(smoothed):
                tr = self._torso_ratio(smoothed, sh_w)
                torso_comp = (self._baseline_torso_ratio - tr) / max(1e-4, self._baseline_torso_ratio) if tr is not None else 0.0
            else:
                torso_comp = 0.0
        else:
            # Anatomical fallback before calibration: typical neck ratio is ~0.42
            neck_comp = max(0.0, (0.42 - neck_r) / 0.42)
            head_drop = 0.0
            delta_fwd_head = 0.0
            torso_comp = 0.0

        # Sub-scores for slouching components
        s_neck = max(0.0, (neck_comp - config.SLOUCH_NECK_SLIGHT) / max(1e-4, config.SLOUCH_NECK_EXCESSIVE - config.SLOUCH_NECK_SLIGHT))
        s_drop = max(0.0, (head_drop - config.HEAD_SLUMP_SLIGHT) / max(1e-4, config.HEAD_SLUMP_MODERATE - config.HEAD_SLUMP_SLIGHT))
        s_head = max(0.0, (-delta_fwd_head - (-config.FORWARD_HEAD_Z_THRESHOLD)) / 0.08)
        s_torso = max(0.0, (torso_comp - config.SLOUCH_TORSO_SLIGHT) / max(1e-4, config.SLOUCH_TORSO_MODERATE - config.SLOUCH_TORSO_SLIGHT))

        slouch_level = s_neck * 0.55 + s_drop * 0.20 + s_head * 0.15 + s_torso * 0.10

        # ── Stance & Leaning Classification ───────────────────
        # Priority: Slouching > Forward Lean > Backward Lean > Lateral Lean > None
        is_slouching = slouch_level >= 0.40 or neck_comp >= config.SLOUCH_NECK_MODERATE
        is_slight_slouch = slouch_level >= 0.15 or neck_comp >= config.SLOUCH_NECK_SLIGHT
        is_forward = fwd_lean_level >= 0.12 or prox_ratio >= config.FORWARD_PROXIMITY_SLIGHT

        if is_slouching:
            if is_forward and (fwd_lean_level >= 0.25 or prox_ratio >= config.FORWARD_PROXIMITY_MODERATE):
                lean_str = "Slouching (Forward)"
                lean_penalty = 38.0 + max(slouch_level, fwd_lean_level) * 26.0
            else:
                lean_str = "Slouching"
                lean_penalty = 32.0 + slouch_level * 25.0
        elif is_slight_slouch:
            if is_forward and fwd_lean_level >= 0.20:
                lean_str = "Slight Slouch (Forward)"
                lean_penalty = 20.0 + max(slouch_level, fwd_lean_level) * 16.0
            else:
                lean_str = "Slight Slouch"
                lean_penalty = 16.0 + slouch_level * 14.0
        elif fwd_lean_level >= 0.60 or prox_ratio >= config.FORWARD_PROXIMITY_EXCESSIVE:
            lean_str = "Excessive (Forward)"
            lean_penalty = 42.0 + fwd_lean_level * 22.0
        elif fwd_lean_level >= 0.30 or prox_ratio >= config.FORWARD_PROXIMITY_MODERATE:
            lean_str = "Moderate (Forward)"
            lean_penalty = 24.0 + fwd_lean_level * 18.0
        elif fwd_lean_level >= 0.10 or prox_ratio >= config.FORWARD_PROXIMITY_SLIGHT:
            lean_str = "Slight (Forward)"
            lean_penalty = 12.0 + fwd_lean_level * 10.0
        elif bwd_lean_level >= 0.50:
            lean_str = "Moderate (Backward)"
            lean_penalty = 22.0 + bwd_lean_level * 16.0
        elif bwd_lean_level >= 0.15:
            lean_str = "Slight (Backward)"
            lean_penalty = 10.0
        elif abs_lat >= t_lat_exc:
            side = "Right" if lat_delta > 0 else "Left"
            lean_str = f"Excessive ({side})"
            lean_penalty = 38.0 + (abs_lat - t_lat_exc) * 3.0
        elif abs_lat >= t_lat_mod:
            side = "Right" if lat_delta > 0 else "Left"
            lean_str = f"Moderate ({side})"
            lean_penalty = 20.0 + (abs_lat - t_lat_mod) * 2.5
        elif abs_lat >= t_lat_slight:
            side = "Right" if lat_delta > 0 else "Left"
            lean_str = f"Slight ({side})"
            lean_penalty = 8.0
        else:
            lean_str = "None"
            lean_penalty = 0.0

        # Persistence tracking: momentary vs sustained posture deviation
        is_deviating = lean_penalty > 8.0
        self._lean_history.append(is_deviating)
        deviation_ratio = sum(self._lean_history) / max(1, len(self._lean_history))
        is_persistent = deviation_ratio > 0.4

        # For slight deviations, reduce penalty if not sustained; full penalty if persistent
        if not is_persistent and lean_penalty <= 20.0:
            lean_penalty = lean_penalty * 0.6

        # ── Shoulder Tension & Shrugging ──────────────────────
        ear_sh_ratio = self._ear_shoulder_ratio(smoothed, sh_w)
        if self.calibrated and self._baseline_ear_shoulder_ratio > 0.05:
            tension_drop = (self._baseline_ear_shoulder_ratio - ear_sh_ratio) / self._baseline_ear_shoulder_ratio
            is_shrugging = (tension_drop >= config.SHOULDER_TENSION_RATIO_DROP) and (head_drop < config.HEAD_SLUMP_SLIGHT)
            if is_shrugging:
                self._tension_counter += 1
            else:
                self._tension_counter = max(0, self._tension_counter - 1)
            shoulder_tension = (self._tension_counter >= config.SHOULDER_TENSION_FRAMES)
        else:
            shoulder_tension = False

        # ── Posture Score Calculation ─────────────────────────
        sh_tilt_penalty = max(0.0, shoulder_delta - t_sh_slight) * 1.8
        tension_penalty = 12.0 if shoulder_tension else 0.0
        raw_posture = max(0.0, min(100.0, 100.0 - lean_penalty - sh_tilt_penalty - tension_penalty))

        self._ema_posture_score += config.POSTURE_SCORE_EMA_ALPHA * (raw_posture - self._ema_posture_score)
        posture_score = round(float(np.clip(self._ema_posture_score, 0, 100)), 1)

        # ── Strict Label Guardrails ───────────────────────────
        # CRITICAL FIX: If slouching or leaning in front is detected, NEVER allow "Good" or "Excellent"
        lean_lower = lean_str.lower()
        has_slouch = "slouch" in lean_lower
        has_mod_or_exc = any(w in lean_lower for w in ("moderate", "excessive"))
        has_slight = "slight" in lean_lower

        # Track posture state for reliable correction event detection
        if posture_score < 70.0 or has_slouch or has_mod_or_exc:
            self._in_poor_posture = True
            correction_event = False
        elif self._in_poor_posture and posture_score >= 75.0 and lean_str == "None":
            correction_event = True
            self._in_poor_posture = False
        else:
            correction_event = False

        if has_slouch or has_mod_or_exc:
            if posture_score < 45.0 or "excessive" in lean_lower or lean_str in ("Slouching", "Slouching (Forward)"):
                raw_label = "Poor"
            else:
                raw_label = "Fair"
        elif has_slight:
            raw_label = "Good" if posture_score >= 75.0 else "Fair"
        else:
            raw_label = "Excellent" if posture_score >= 85.0 else "Good" if posture_score >= 70.0 else "Fair" if posture_score >= 40.0 else "Poor"

        # Multi-frame consensus for posture label to avoid label flicker
        self._posture_label_history.append(raw_label)
        if len(self._posture_label_history) >= config.POSTURE_CONSENSUS_WINDOW:
            from collections import Counter
            counts = Counter(self._posture_label_history)
            consensus_label = counts.most_common(1)[0][0]
        else:
            consensus_label = raw_label

        # ── Head horizontal alignment ─────────────────────────
        if self.calibrated:
            head_delta = abs(head_offset - self._baseline_head_offset)
        else:
            head_delta = abs(head_offset)

        if head_delta < config.HEAD_OFFSET_SLIGHT:
            head_status = "Centered"
        elif head_delta < config.HEAD_OFFSET_SIGNIFICANT:
            head_status = "Slightly Off-Center"
        else:
            head_status = "Off-Center"

        # ── Stability ─────────────────────────────────────────
        mid_sh = self._mid_shoulder(smoothed)
        self._position_history.append(mid_sh)
        stability_score = self._stability_score()

        self._ema_shoulder_score += config.POSTURE_SCORE_EMA_ALPHA * (sh_score - self._ema_shoulder_score)

        return {
            "detected": True,
            "insufficient_data": False,
            "reliability": reliability,
            "posture": consensus_label,
            "shoulder_alignment": shoulder_status,
            "shoulder_tension": shoulder_tension,
            "shoulder_status": "Tense / Elevated" if shoulder_tension else "Relaxed",
            "leaning": lean_str,
            "head_alignment": head_status,
            "posture_score": posture_score,
            "shoulder_score": round(float(np.clip(self._ema_shoulder_score, 0, 100)), 1),
            "stability_score": stability_score,
            "correction_event": correction_event,
        }

    # ─── Geometry Helpers ────────────────────────────────────────

    @staticmethod
    def _calculate_reliability(landmarks) -> float:
        """Calculate detection reliability based on presence and frame bounds of core upper body landmarks."""
        if landmarks is None or len(landmarks) == 0:
            return 0.0

        in_bounds = 0
        for idx in RELIABILITY_LANDMARKS:
            try:
                lm = landmarks[idx]
                if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                    in_bounds += 1
            except (IndexError, KeyError, AttributeError):
                pass
                
        if in_bounds == 3:
            return 100.0
        elif in_bounds == 2:
            return 80.0
        else:
            return 30.0

    @staticmethod
    def _key_landmarks_visible(landmarks) -> bool:
        """Check that the core upper-body landmarks are within frame bounds."""
        if landmarks is None:
            return False
        present_count = 0
        for idx in RELIABILITY_LANDMARKS:
            try:
                lm = landmarks[idx]
                if -0.1 <= lm.x <= 1.1 and -0.1 <= lm.y <= 1.1:
                    present_count += 1
            except (IndexError, KeyError, AttributeError):
                pass
        return present_count >= 2

    @staticmethod
    def _shoulder_width(lm: Dict[int, Point3D]) -> float:
        """2D Euclidean distance between shoulders in normalized coordinates."""
        ls = lm[LEFT_SHOULDER]
        rs = lm[RIGHT_SHOULDER]
        return math.dist((ls.x, ls.y), (rs.x, rs.y))

    @staticmethod
    def _shoulder_angle(lm: Dict[int, Point3D]) -> float:
        """Angle of shoulder line from horizontal (degrees)."""
        ls = lm[LEFT_SHOULDER]
        rs = lm[RIGHT_SHOULDER]
        return math.degrees(math.atan2(rs.y - ls.y, rs.x - ls.x))

    def _lateral_angle(self, lm: Dict[int, Point3D]) -> float:
        """Torso side-to-side lean from vertical (degrees) in coronal plane."""
        msx = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2.0
        msy = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        
        if self._hips_usable(lm):
            mhx = (lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2.0
            mhy = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2.0
            dx = msx - mhx
            dy = mhy - msy
        else:
            # Fallback to upper torso / head vertical alignment
            dx = lm[NOSE].x - msx
            dy = msy - lm[NOSE].y

        if dy < 1e-4:
            return 0.0
        return math.degrees(math.atan2(dx, dy))

    def _sagittal_angle(self, lm: Dict[int, Point3D]) -> Optional[float]:
        """Torso forward/backward lean from vertical (degrees) in sagittal plane."""
        if not self._hips_usable(lm):
            return None
        msy = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        msz = (lm[LEFT_SHOULDER].z + lm[RIGHT_SHOULDER].z) / 2.0
        mhy = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2.0
        mhz = (lm[LEFT_HIP].z + lm[RIGHT_HIP].z) / 2.0
        
        dy = mhy - msy
        dz = mhz - msz  # positive when shoulders are closer to camera than hips
        if dy < 1e-4:
            return None
        return math.degrees(math.atan2(dz, dy))

    @staticmethod
    def _neck_ratio(lm: Dict[int, Point3D], shoulder_width: float) -> float:
        """Vertical distance from mid-shoulder to nose normalized by shoulder width.
        In normal upright sitting posture, this is maximal; decreases during slouching/hunching."""
        msy = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        neck_h = msy - lm[NOSE].y  # nose is above mid-shoulder (smaller y)
        return neck_h / max(1e-4, shoulder_width)

    def _torso_ratio(self, lm: Dict[int, Point3D], shoulder_width: float) -> Optional[float]:
        """Vertical distance from hips to shoulders normalized by shoulder width."""
        if not self._hips_usable(lm):
            return None
        msy = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        mhy = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2.0
        torso_h = mhy - msy
        return torso_h / max(1e-4, shoulder_width)

    @staticmethod
    def _hips_usable(lm: Dict[int, Point3D]) -> bool:
        """Check if hip landmarks are inside image bounds and vertically distinct from shoulders."""
        lh = lm.get(LEFT_HIP)
        rh = lm.get(RIGHT_HIP)
        if lh is None or rh is None:
            return False
        # If hips are past bottom edge (y > 1.05) or inverted, they are not reliable
        if lh.y > 1.05 or rh.y > 1.05:
            return False
        sh_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        hip_y = (lh.y + rh.y) / 2.0
        return (hip_y - sh_y) > 0.12

    @staticmethod
    def _head_offset(lm: Dict[int, Point3D]) -> float:
        """Horizontal offset of nose from mid-shoulder, normalized by shoulder width."""
        ls = lm[LEFT_SHOULDER]
        rs = lm[RIGHT_SHOULDER]
        msx = (ls.x + rs.x) / 2.0
        shoulder_width = math.dist((ls.x, ls.y), (rs.x, rs.y))
        if shoulder_width < 1e-4:
            return 0.0
        return (lm[NOSE].x - msx) / shoulder_width

    @staticmethod
    def _forward_head(lm: Dict[int, Point3D], shoulder_width: float) -> float:
        """Estimate forward head protrusion by comparing Z-depth of ears to shoulders."""
        try:
            ls = lm[LEFT_SHOULDER]
            rs = lm[RIGHT_SHOULDER]
            le = lm[LEFT_EAR]
            re = lm[RIGHT_EAR]
        except KeyError:
            return 0.0
            
        mid_shoulder_z = (ls.z + rs.z) / 2.0
        mid_ear_z = (le.z + re.z) / 2.0
        return (mid_ear_z - mid_shoulder_z) / max(1e-4, shoulder_width)

    @staticmethod
    def _ear_shoulder_ratio(lm: Dict[int, Point3D], shoulder_width: float) -> float:
        """Vertical distance from mid-ear to mid-shoulder normalized by shoulder width.
        In upright posture, ears are well above shoulders; during shoulder shrugging/tension,
        this distance shortens significantly."""
        try:
            ls_y = lm[LEFT_SHOULDER].y
            rs_y = lm[RIGHT_SHOULDER].y
            le_y = lm[LEFT_EAR].y
            re_y = lm[RIGHT_EAR].y
            mid_sh_y = (ls_y + rs_y) / 2.0
            mid_ear_y = (le_y + re_y) / 2.0
            return (mid_sh_y - mid_ear_y) / max(1e-4, shoulder_width)
        except (KeyError, AttributeError):
            return 0.35

    @staticmethod
    def _mid_shoulder(lm: Dict[int, Point3D]) -> np.ndarray:
        """Return mid-shoulder position as a NumPy array [x, y]."""
        return np.array([
            (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2.0,
            (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0,
        ])

    def _stability_score(self) -> float:
        """Compute stability from the variance of recent mid-shoulder positions."""
        if len(self._position_history) < 5:
            return 100.0
        pts = np.array(self._position_history)
        variance = float(np.var(pts[:, 0]) + np.var(pts[:, 1]))
        score = max(0.0, 100.0 - variance * config.POSTURE_STABILITY_VARIANCE_SCALE)
        return round(float(np.clip(score, 0, 100)), 1)

    @staticmethod
    def _empty_result(reliability: float, insufficient_data: bool = False) -> dict:
        """Return a default result when landmarks are unavailable."""
        return {
            "detected": False,
            "insufficient_data": insufficient_data,
            "reliability": reliability,
            "posture": "Unknown",
            "shoulder_alignment": "Unknown",
            "shoulder_tension": False,
            "shoulder_status": "Normal",
            "leaning": "Unknown",
            "head_alignment": "Unknown",
            "posture_score": 0.0,
            "shoulder_score": 0.0,
            "stability_score": 0.0,
            "correction_event": False,
        }

