"""
Hand Analyzer Module

Uses MediaPipe HandLandmarker to track hand visibility, movement stability
(fidgeting), and face-touching behavior.
"""

import math
import time
import numpy as np
from collections import deque
from typing import Tuple, List

try:
    from . import config
except (ImportError, ValueError):
    import config

# Smoothing configuration
VARIANCE_WINDOW = config.HAND_VARIANCE_WINDOW  # ~3 sec for gesture analysis

# Landmark indices
WRIST = 0
INDEX_FINGER_PIP = 6
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_TIP = 12
RING_FINGER_PIP = 14
RING_FINGER_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20


class Point3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class HandAnalyzer:
    def __init__(self) -> None:
        self._ema_initialized = False
        # Track up to 2 hands (left and right)
        self._ema_wrists = [Point3D(), Point3D()]
        self._ema_indices = [Point3D(), Point3D()]
        
        # Calibration
        self.calibrated = False
        self._cal_variance = []
        self._baseline_variance = 0.0
        
        # History for stability (fidgeting)
        self._wrist_history = deque(maxlen=config.HAND_VARIANCE_WINDOW)
        self._index_history = deque(maxlen=config.HAND_VARIANCE_WINDOW)
        self._face_touch_history = deque(maxlen=30)  # 1 second for persistence
        
        self._ema_gesture_score = 100.0
        
        # Multi-frame consensus for fidgeting
        self._fidget_history: deque = deque(maxlen=config.FIDGET_CONSENSUS_FRAMES)
        
        # Multi-frame persistence for obscene gesture
        self._obscene_history: deque = deque(maxlen=config.OBSCENE_GESTURE_PERSISTENCE)
        
    def calibrate_frame(self, hand_landmarks_list: List[List[Point3D]]) -> None:
        if not hand_landmarks_list:
            return
            
        current_wrists = []
        for i, hand_lm in enumerate(hand_landmarks_list):
            if i > 1: break
            try:
                w = hand_lm[WRIST]
                current_wrists.append((w.x, w.y, time.time()))
            except IndexError:
                continue
                
        if current_wrists:
            self._wrist_history.append(current_wrists[0])
            if len(self._wrist_history) > 15:
                var, _, _ = self._analyze_movement()
                self._cal_variance.append(var)
                
    def finish_calibration(self) -> None:
        if self._cal_variance:
            self._baseline_variance = float(np.median(self._cal_variance))
        self.calibrated = True

    def analyze(self, hand_landmarks_list: List[List[Point3D]], nose_pos: Tuple[float, float]) -> dict:
        """
        Analyze hands for the current frame.
        hand_landmarks_list: List of hands, where each hand is a list of landmarks.
        nose_pos: (x, y) normalized coordinates of the nose (to detect face touching).
        """
        if not hand_landmarks_list:
            # No hands visible (maybe hidden under desk)
            return self._empty_result()
            
        hands_visible = len(hand_landmarks_list)
        
        current_wrists = []
        current_indices = []
        
        face_touching = False
        obscene_gesture = False
        
        for i, hand_lm in enumerate(hand_landmarks_list):
            if i > 1: break # Only process up to 2 hands
            
            try:
                w = hand_lm[WRIST]
                idx = hand_lm[INDEX_FINGER_TIP]
                
                # Check for face touching (hand very close to nose)
                if nose_pos:
                    dist_to_nose = math.dist((w.x, w.y), nose_pos)
                    idx_dist = math.dist((idx.x, idx.y), nose_pos)
                    face_touch_dist = config.FACE_TOUCH_DISTANCE
                    if dist_to_nose < face_touch_dist or idx_dist < face_touch_dist:
                        face_touching = True
                        
                # Check for obscene gesture (middle finger) with persistence
                if self._is_middle_finger(hand_lm):
                    self._obscene_history.append(True)
                else:
                    self._obscene_history.append(False)

                if not self._ema_initialized:
                    self._ema_wrists[i].x, self._ema_wrists[i].y = w.x, w.y
                    self._ema_indices[i].x, self._ema_indices[i].y = idx.x, idx.y
                else:
                    self._ema_wrists[i].x += config.HAND_COORD_EMA_ALPHA * (w.x - self._ema_wrists[i].x)
                    self._ema_wrists[i].y += config.HAND_COORD_EMA_ALPHA * (w.y - self._ema_wrists[i].y)
                    self._ema_indices[i].x += config.HAND_COORD_EMA_ALPHA * (idx.x - self._ema_indices[i].x)
                    self._ema_indices[i].y += config.HAND_COORD_EMA_ALPHA * (idx.y - self._ema_indices[i].y)
                
                current_wrists.append((self._ema_wrists[i].x, self._ema_wrists[i].y, time.time()))
                current_indices.append((self._ema_indices[i].x, self._ema_indices[i].y))
                
            except IndexError:
                continue
                
        self._ema_initialized = True
        
        self._face_touch_history.append(face_touching)
        face_touching_persistent = sum(self._face_touch_history) >= config.FACE_TOUCH_PERSISTENCE
        
        # Obscene gesture requires sustained persistence (not a single frame)
        obscene_gesture = (sum(self._obscene_history) >= config.OBSCENE_GESTURE_PERSISTENCE * 0.7)
        
        # Stability (Fidgeting) calculation
        if current_wrists:
            self._wrist_history.append(current_wrists[0]) # Just track primary hand for variance
        
        variance, amplitude, freq = self._analyze_movement()
            
        # Score calculation
        # Base score 100
        # -10 for only 1 hand visible
        # -50 for persistent face touching
        # Penalize fidgeting only if high frequency and low amplitude
        
        raw_score = 100.0
        if hands_visible == 1:
            raw_score -= 10.0
            
        if face_touching_persistent:
            raw_score -= 40.0
            
        is_fidgeting = False
        is_gesturing = False
        fidget_penalty = 0.0
        
        if len(self._wrist_history) > 30:
            # Factor in calibrated variance (fidgeting should be significantly above rest variance)
            threshold_freq = config.FIDGET_FREQ_THRESHOLD
            if self.calibrated and self._baseline_variance > 0:
                # If they are naturally a bit more active, raise the frequency threshold
                pass
                
            if amplitude > config.GESTURE_AMPLITUDE_THRESHOLD:
                # Large movements = Natural Gestures
                is_gesturing = True
            elif freq > threshold_freq and config.FIDGET_AMPLITUDE_MIN < amplitude < config.FIDGET_AMPLITUDE_MAX:
                # Small, rapid, repetitive movements = Fidgeting (requires consensus)
                self._fidget_history.append(True)
            else:
                self._fidget_history.append(False)
            
            # Fidget requires consensus over N frames
            if len(self._fidget_history) >= config.FIDGET_CONSENSUS_FRAMES:
                fidget_ratio = sum(self._fidget_history) / len(self._fidget_history)
                if fidget_ratio >= 0.6:  # 60% of recent frames show fidgeting
                    is_fidgeting = True
                    fidget_penalty = min(30.0, freq * 4.0)
                
        raw_score -= fidget_penalty
        
        raw_score = float(np.clip(raw_score, 0, 100))
        
        # Smooth the score
        self._ema_gesture_score += config.HAND_SCORE_EMA_ALPHA * (raw_score - self._ema_gesture_score)
        final_score = round(float(np.clip(self._ema_gesture_score, 0, 100)), 1)
        
        # Determine status
        if obscene_gesture:
            status = "Obscene Gesture"
            raw_score = 0.0 # Heavy penalty
            self._ema_gesture_score = 0.0
            final_score = 0.0
        elif face_touching_persistent:
            status = "Touching Face"
        elif is_fidgeting:
            status = "Possible Fidgeting"
        elif is_gesturing:
            status = "Natural Gesture"
        elif hands_visible == 1:
            status = "One Hand Hidden"
        else:
            status = "Normal Movement" if amplitude > 0.05 else "Resting"
            
        return {
            "detected": True,
            "insufficient_data": False,
            "hands_visible": hands_visible,
            "face_touching": face_touching_persistent,
            "obscene_gesture": obscene_gesture,
            "gesture_score": final_score,
            "status": status
        }
        
    def _analyze_movement(self) -> Tuple[float, float, float]:
        """Return (variance, amplitude, frequency) of wrist movement over the window."""
        if len(self._wrist_history) < 15:
            return 0.0, 0.0, 0.0
            
        xs = np.array([p[0] for p in self._wrist_history])
        ys = np.array([p[1] for p in self._wrist_history])
        
        variance = float(np.var(xs) + np.var(ys))
        amplitude = float((np.max(xs) - np.min(xs)) + (np.max(ys) - np.min(ys)))
        
        # Count direction changes (frequency) with noise rejection
        # Apply a simple 3-point median filter before computing direction changes
        if len(xs) >= 3:
            xs_filtered = np.array([np.median(xs[max(0,i-1):i+2]) for i in range(len(xs))])
            ys_filtered = np.array([np.median(ys[max(0,i-1):i+2]) for i in range(len(ys))])
        else:
            xs_filtered = xs
            ys_filtered = ys
        
        dx = np.diff(xs_filtered)
        dy = np.diff(ys_filtered)
        
        # Only consider significant movements (reject sub-pixel jitter)
        jitter_threshold = 0.002  # normalized coords
        dir_x = np.sign(dx)
        dir_y = np.sign(dy)
        
        # Zero out near-zero movements
        dir_x[np.abs(dx) < jitter_threshold] = 0
        dir_y[np.abs(dy) < jitter_threshold] = 0
        
        dir_x = dir_x[dir_x != 0]
        dir_y = dir_y[dir_y != 0]
        
        changes_x = np.sum(dir_x[:-1] != dir_x[1:]) if len(dir_x) > 1 else 0
        changes_y = np.sum(dir_y[:-1] != dir_y[1:]) if len(dir_y) > 1 else 0
        
        # Real-time elapsed duration between first and latest sample in the window
        duration_sec = self._wrist_history[-1][2] - self._wrist_history[0][2]
        freq = float((changes_x + changes_y) / max(0.5, duration_sec))
        
        return variance, amplitude, freq
        
    def _is_middle_finger(self, hand_lm) -> bool:
        def dist(p1, p2):
            return math.dist((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z))
            
        try:
            w = hand_lm[WRIST]
            
            # MCP (Metacarpophalangeal) joint indices
            INDEX_MCP = 5
            MIDDLE_MCP = 9
            RING_MCP = 13
            PINKY_MCP = 17
            
            # A finger is extended if its TIP is significantly further from the wrist than its MCP
            # When curled, the TIP usually comes closer to the palm (wrist/MCP).
            def is_extended(tip_idx, mcp_idx):
                return dist(w, hand_lm[tip_idx]) > dist(w, hand_lm[mcp_idx]) * 1.3
                
            index_ext = is_extended(INDEX_FINGER_TIP, INDEX_MCP)
            middle_ext = is_extended(MIDDLE_FINGER_TIP, MIDDLE_MCP)
            ring_ext = is_extended(RING_FINGER_TIP, RING_MCP)
            pinky_ext = is_extended(PINKY_TIP, PINKY_MCP)
            
            return middle_ext and not index_ext and not ring_ext and not pinky_ext
        except IndexError:
            return False
            
    def _empty_result(self) -> dict:
        """When hands are resting on the lap or desk below the camera view."""
        raw_score = 90.0  # Composed resting posture
        if not self._ema_initialized:
            self._ema_gesture_score = raw_score
            self._ema_initialized = True
        else:
            self._ema_gesture_score += config.HAND_SCORE_EMA_ALPHA * (raw_score - self._ema_gesture_score)
            
        final_score = round(float(np.clip(self._ema_gesture_score, 0, 100)), 1)
        
        return {
            "detected": False,
            "insufficient_data": False,
            "hands_visible": 0,
            "face_touching": False,
            "obscene_gesture": False,
            "gesture_score": final_score,
            "status": "Resting"
        }
