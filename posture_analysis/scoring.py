"""
Scoring Module

Transparent, explainable scoring system for interview readiness.

Combines five sub-scores (posture, eye contact, head position,
expression, stability) into an overall Interview Readiness Score.
All weights are configurable constants; scoring uses exponential
moving average smoothing so that brief anomalies don't cause
jarring score swings.

The overall score is labelled as an *estimated* interview-readiness
indicator, not a psychological assessment.
"""

import time
import numpy as np
try:
    from . import config
except (ImportError, ValueError):
    import config

# ─── Status Thresholds ──────────────────────────────────────────
EXCELLENT_THRESHOLD = 90
GOOD_THRESHOLD = 75
AVERAGE_THRESHOLD = 60
NEEDS_IMPROVEMENT_THRESHOLD = 40

# ─── Smoothing ──────────────────────────────────────────────────
STATUS_DEBOUNCE_FRAMES = 12  # ~0.4s at 30fps: fast responsiveness without flickering


class StatusDebouncer:
    """Provides hysteresis (debouncing) for string statuses to prevent UI flickering."""
    def __init__(self, initial_state="Unknown", frames_required=STATUS_DEBOUNCE_FRAMES):
        self.current_state = initial_state
        self.pending_state = initial_state
        self.pending_count = 0
        self.frames_required = frames_required
        
    def update(self, new_state: str) -> str:
        # Immediately adopt first valid state rather than staying 'Unknown' for 15 frames
        if self.current_state in ("Unknown", "N/A", "Average") and new_state not in ("Unknown", "N/A"):
            self.current_state = new_state
            self.pending_state = new_state
            self.pending_count = 0
            return self.current_state

        if new_state == self.current_state:
            self.pending_count = 0
            return self.current_state
            
        if new_state == self.pending_state:
            self.pending_count += 1
            if self.pending_count >= self.frames_required:
                self.current_state = new_state
                self.pending_count = 0
        else:
            self.pending_state = new_state
            self.pending_count = 1
            
        return self.current_state


class InterviewScorer:
    """Calculates and tracks interview-readiness scores."""

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._frame_count: int = 0

        # Smoothed current scores
        self._scores: dict = {
            "posture": 85.0,
            "eye_contact": 85.0,
            "head_position": 85.0,
            "expression": 85.0,
            "stability": 90.0,
            "gestures": 90.0,
            "overall": 85.0,
            "reliability": 100.0,
        }

        # History for session averages (one entry per frame)
        self._history: dict = {k: [] for k in self._scores}

        # Event duration tracking
        self._fidgeting_frames: int = 0
        self._gaze_away_frames: int = 0

        # Event counters
        self.good_posture_frames: int = 0
        self.poor_posture_frames: int = 0
        self.posture_warnings: int = 0
        self.sustained_gaze_away_events: int = 0
        self.distraction_events: int = 0
        self.low_reliability_frames: int = 0
        self.posture_corrections: int = 0
        self.shoulder_tension_events: int = 0

        # Edge-detection flags (rising-edge tracking)
        self._was_poor_posture: bool = False
        self._was_sustained_away: bool = False
        self._was_distracted: bool = False
        self._was_fidgeting: bool = False
        self._was_shoulder_tension: bool = False
        
        # Latest behavioral metrics
        self._last_blink_rate: float = 0.0
        self._last_blink_count: int = 0
        self._last_blink_status: str = "Normal"
        
        # Chronological Event Log
        self.event_log: list[str] = []
        
        # Debouncers for text fields
        self._debouncers = {
            "overall_status": StatusDebouncer("Average"),
            "posture": StatusDebouncer("Unknown"),
            "shoulder_alignment": StatusDebouncer("Unknown"),
            "leaning": StatusDebouncer("Unknown"),
            "head_alignment": StatusDebouncer("Unknown"),
            "head_position": StatusDebouncer("Unknown"),
            "eye_contact": StatusDebouncer("Unknown"),
            "expression": StatusDebouncer("Unknown"),
            "gestures": StatusDebouncer("Unknown"),
        }

    # ─── Properties ──────────────────────────────────────────────

    @property
    def session_duration(self) -> float:
        """Elapsed seconds since the scorer was created."""
        return time.time() - self._start_time

    # ─── Per-Frame Update ────────────────────────────────────────

    def update(self, posture_data: dict, face_data: dict, hand_data: dict) -> dict:
        """Ingest one frame of analyser data and return current scores.

        Args:
            posture_data: Output of ``PostureAnalyzer.analyze()``. (Modified in-place)
            face_data:    Output of ``FaceAnalyzer.analyze()``. (Modified in-place)
            hand_data:    Output of ``HandAnalyzer.analyze()``. (Modified in-place)

        Returns:
            A dict with rounded sub-scores, ``overall``, ``reliability``, and ``status``.
        """
        self._frame_count += 1
        
        # ── Debounce statuses in the data dicts in-place ──
        if "posture" in posture_data:
            posture_data["posture"] = self._debouncers["posture"].update(posture_data["posture"])
        if "shoulder_alignment" in posture_data:
            posture_data["shoulder_alignment"] = self._debouncers["shoulder_alignment"].update(posture_data["shoulder_alignment"])
        if "leaning" in posture_data:
            posture_data["leaning"] = self._debouncers["leaning"].update(posture_data["leaning"])
        if "head_alignment" in posture_data:
            posture_data["head_alignment"] = self._debouncers["head_alignment"].update(posture_data["head_alignment"])
            
        if "head_position" in face_data:
            face_data["head_position"] = self._debouncers["head_position"].update(face_data["head_position"])
        if "eye_contact" in face_data:
            face_data["eye_contact"] = self._debouncers["eye_contact"].update(face_data["eye_contact"])
        if "expression" in face_data:
            face_data["expression"] = self._debouncers["expression"].update(face_data["expression"])
            
        if "status" in hand_data:
            hand_data["status"] = self._debouncers["gestures"].update(hand_data["status"])

        # ── Raw scores from analysers ──
        raw = {
            "posture": posture_data.get("posture_score", self._scores["posture"]),
            "eye_contact": face_data.get("eye_score", self._scores["eye_contact"]),
            "head_position": face_data.get("head_score", self._scores["head_position"]),
            "expression": face_data.get("expression_score", self._scores["expression"]),
            "stability": posture_data.get("stability_score", self._scores["stability"]),
            "gestures": hand_data.get("gesture_score", self._scores["gestures"]),
        }

        # If a detector didn't find its target or reliability is too low, hold the previous score
        if not posture_data.get("detected", False) or posture_data.get("insufficient_data", False):
            raw["posture"] = self._scores["posture"]
            raw["stability"] = self._scores["stability"]
        if not face_data.get("detected", False) or face_data.get("insufficient_data", False):
            raw["eye_contact"] = self._scores["eye_contact"]
            raw["head_position"] = self._scores["head_position"]
            raw["expression"] = self._scores["expression"]

        # ── EMA smoothing for scores ──
        for key in raw:
            # Different alpha depending on the type of score
            if key == "posture" or key == "stability":
                alpha = config.POSTURE_SCORE_EMA_ALPHA
            elif key == "gestures":
                alpha = config.HAND_SCORE_EMA_ALPHA
            elif key == "expression":
                alpha = config.EXPRESSION_SCORE_EMA_ALPHA
            else:
                alpha = config.FACE_SCORE_EMA_ALPHA
                
            self._scores[key] += alpha * (raw[key] - self._scores[key])
            self._scores[key] = float(np.clip(self._scores[key], 0, 100))
            
        # ── Detection Reliability ──
        posture_rel = posture_data.get("reliability", 0.0) if posture_data.get("detected") else 0.0
        face_rel = face_data.get("reliability", 0.0) if face_data.get("detected") else 0.0
        
        # Primary tracking targets are upper-body posture and face
        if posture_data.get("detected") and face_data.get("detected"):
            current_rel = (posture_rel + face_rel) / 2.0
        elif face_data.get("detected"):
            current_rel = face_rel * 0.85
        elif posture_data.get("detected"):
            current_rel = posture_rel * 0.85
        else:
            current_rel = 0.0

        self._scores["reliability"] += config.RELIABILITY_EMA_ALPHA * (current_rel - self._scores["reliability"])
        self._scores["reliability"] = float(np.clip(self._scores["reliability"], 0, 100))
        
        if current_rel < 40.0:
            self.low_reliability_frames += 1

        # ── Weighted overall score ──
        overall = (
            self._scores["posture"] * config.WEIGHT_POSTURE
            + self._scores["eye_contact"] * config.WEIGHT_EYE_CONTACT
            + self._scores["head_position"] * config.WEIGHT_HEAD_POSITION
            + self._scores["expression"] * config.WEIGHT_EXPRESSION
            + self._scores["stability"] * config.WEIGHT_STABILITY
            + self._scores["gestures"] * config.WEIGHT_GESTURES
        )
        self._scores["overall"] += config.OVERALL_SCORE_EMA_ALPHA * (overall - self._scores["overall"])
        self._scores["overall"] = float(np.clip(self._scores["overall"], 0, 100))
        
        raw_status = self._status_label(self._scores["overall"])
        self._scores["status"] = self._debouncers["overall_status"].update(raw_status)

        # ── Record history ──
        for key in self._history:
            self._history[key].append(self._scores[key])

        # ── Capture latest blink metrics ──
        if "blink_rate" in face_data:
            self._last_blink_rate = face_data.get("blink_rate", 0.0)
            self._last_blink_count = face_data.get("blink_count", 0)
            self._last_blink_status = face_data.get("blink_status", "Normal")

        # ── Event detection ──
        self._detect_events(posture_data, face_data, hand_data)

        result = {
            k: round(v, 1) if isinstance(v, float) else v
            for k, v in self._scores.items()
        }
        
        # Include the latest 4 events for the UI
        result["recent_events"] = self.event_log[-4:]
        
        return result

    # ─── Event Detection ─────────────────────────────────────────
    
    def _log_event(self, event_str: str) -> None:
        """Add a chronological event with a timestamp."""
        duration = self.session_duration
        mins, secs = divmod(int(duration), 60)
        ts_str = f"{mins:02d}:{secs:02d}"
        self.event_log.append(f"{ts_str} — {event_str}")

    def _detect_events(self, posture_data: dict, face_data: dict, hand_data: dict) -> None:
        """Track rising-edge warnings, corrections, and overall durations."""
        
        # ── Posture ──
        posture_label = posture_data.get("posture", "Unknown")
        lean_str = posture_data.get("leaning", "None")
        lean_lower = lean_str.lower()
        has_slouch_or_lean = "slouch" in lean_lower or "moderate" in lean_lower or "excessive" in lean_lower

        is_good = posture_label in ("Excellent", "Good") and not has_slouch_or_lean
        is_poor = posture_label in ("Poor", "Very Poor") or has_slouch_or_lean

        if is_good:
            self.good_posture_frames += 1
        if is_poor:
            self.poor_posture_frames += 1
            if not self._was_poor_posture:
                self.posture_warnings += 1
                self._log_event(f"Posture Warning ({lean_str})")
        
        # Posture Correction: transition from poor posture back to good posture
        if is_good and self._was_poor_posture:
            self.posture_corrections += 1
            self._log_event("Posture corrected")
            
        self._was_poor_posture = is_poor

        # ── Shoulder Tension & Shrugging ──
        is_tense = posture_data.get("shoulder_tension", False)
        if is_tense and not self._was_shoulder_tension:
            self.shoulder_tension_events += 1
            self._log_event("Shoulder tension / shrugging detected")
        self._was_shoulder_tension = is_tense

        # ── Eye Contact ──
        gaze_label = face_data.get("eye_contact", "Unknown")
        is_sustained = gaze_label == "Sustained Away"
        is_distracted = gaze_label == "Occasional Distraction"
        
        if is_sustained or is_distracted:
            self._gaze_away_frames += 1
            
        if is_sustained and not self._was_sustained_away:
            self.sustained_gaze_away_events += 1
            self._log_event("Sustained Gaze-Away detected")
        self._was_sustained_away = is_sustained
        
        if is_distracted and not self._was_distracted:
            self.distraction_events += 1
            self._log_event("Gaze distracted")
        self._was_distracted = is_distracted
        
        # ── Hands ──
        hand_status = hand_data.get("status", "Unknown")
        is_fidgeting = hand_status == "Possible Fidgeting"
        if is_fidgeting:
            self._fidgeting_frames += 1
            if not self._was_fidgeting:
                self._log_event("Fidgeting detected")
        self._was_fidgeting = is_fidgeting
        
        if hand_status == "Touching Face" and hand_data.get("face_touching"):
            # We don't have a specific _was_face_touch flag, but it's debounced heavily
            # To avoid spamming, we will just use the debouncer change.
            if getattr(self, '_last_hand_status', None) != "Touching Face":
                self._log_event("Face-touching detected")
        self._last_hand_status = hand_status

    # ─── Status Label ────────────────────────────────────────────

    @staticmethod
    def _status_label(score: float) -> str:
        """Map a 0-100 score to a human-readable status."""
        if score >= EXCELLENT_THRESHOLD:
            return "Excellent"
        if score >= GOOD_THRESHOLD:
            return "Good"
        if score >= AVERAGE_THRESHOLD:
            return "Average"
        if score >= NEEDS_IMPROVEMENT_THRESHOLD:
            return "Needs Improvement"
        return "Poor"

    # ─── Session Metrics & Summary ───────────────────────────────

    def get_averages(self) -> dict:
        """Return dict of session average sub-scores."""
        avg = {}
        for key in ("posture", "eye_contact", "head_position",
                    "expression", "stability", "gestures", "overall", "reliability"):
            vals = self._history.get(key, [])
            avg[key] = round(float(np.mean(vals)), 1) if vals else 0.0
        return avg

    def get_average(self, key: str) -> float:
        """Return average for a single metric key."""
        vals = self._history.get(key, [])
        return round(float(np.mean(vals)), 1) if vals else 0.0

    def get_strengths(self) -> list[str]:
        """Compute observable strengths based on session averages."""
        avg = self.get_averages()
        strengths = []
        if avg.get("posture", 0) >= 75:
            strengths.append("Good sitting posture")
        if avg.get("eye_contact", 0) >= 75:
            strengths.append("Consistent eye contact")
        if avg.get("head_position", 0) >= 75:
            strengths.append("Good head alignment")
        if avg.get("stability", 0) >= 80:
            strengths.append("Stable body position")
        if avg.get("expression", 0) >= 75:
            strengths.append("Calm, attentive facial expression")
        if avg.get("gestures", 0) >= 75:
            strengths.append("Good hand posture and composure")
        if self.shoulder_tension_events == 0 and self.good_posture_frames > 20:
            strengths.append("Relaxed upper body with no shoulder tension")
        return strengths

    def get_improvements(self) -> list[str]:
        """Compute actionable improvements based on session averages."""
        avg = self.get_averages()
        improvements = []
        if avg.get("posture", 0) < 75:
            improvements.append("Work on maintaining upright posture")
        if avg.get("eye_contact", 0) < 75:
            improvements.append("Maintain eye contact more consistently")
        if avg.get("head_position", 0) < 75:
            improvements.append("Keep head centered and facing forward")
        if avg.get("stability", 0) < 80:
            improvements.append("Reduce unnecessary body movement")
        if avg.get("expression", 0) < 75:
            improvements.append("Try to relax facial tension and maintain open engagement")
        if avg.get("gestures", 0) < 75:
            if self._fidgeting_frames > 30:
                improvements.append("Minimize repetitive hand fidgeting")
            else:
                improvements.append("Maintain relaxed, natural hand gestures")
        if self.shoulder_tension_events >= 1:
            improvements.append("Periodically drop and relax your shoulders to relieve upper-body tension")
        if getattr(self, "_last_blink_status", "") == "Elevated (Stress)":
            improvements.append("Elevated blink rate detected — practice steady, calm breathing to ease nervousness")
        return improvements

    def get_session_summary(self) -> str:
        """Return a formatted multi-line session summary string."""
        duration = self.session_duration
        mins, secs = divmod(int(duration), 60)

        # Compute session averages
        avg: dict = {}
        for key in ("posture", "eye_contact", "head_position",
                     "expression", "stability", "gestures", "overall", "reliability"):
            vals = self._history.get(key, [])
            avg[key] = round(float(np.mean(vals)), 1) if vals else 0.0

        total_classified = self.good_posture_frames + self.poor_posture_frames
        good_pct = round(
            self.good_posture_frames / max(1, total_classified) * 100, 1
        ) if total_classified > 0 else 0.0
        poor_pct = round(
            self.poor_posture_frames / max(1, total_classified) * 100, 1
        ) if total_classified > 0 else 0.0

        # Determine strengths and areas to improve
        strengths: list[str] = []
        improvements: list[str] = []

        if avg["posture"] >= 75:
            strengths.append("Good sitting posture")
        else:
            improvements.append("Work on maintaining upright posture")

        if avg["eye_contact"] >= 75:
            strengths.append("Consistent eye contact")
        else:
            improvements.append("Maintain eye contact more consistently")

        if avg["head_position"] >= 75:
            strengths.append("Good head alignment")
        else:
            improvements.append("Keep head centered and facing forward")

        if avg["stability"] >= 80:
            strengths.append("Stable body position")
        else:
            improvements.append("Reduce unnecessary body movement")

        if avg["expression"] >= 75:
            strengths.append("Calm, attentive facial expression")
        else:
            improvements.append("Try to relax facial tension and maintain open engagement")
            
        if avg["gestures"] >= 75:
            strengths.append("Good hand posture and composure")
        elif self._fidgeting_frames > 30:
            improvements.append("Minimize repetitive hand fidgeting")
        else:
            improvements.append("Maintain relaxed, natural hand gestures")

        if self.shoulder_tension_events == 0 and self.good_posture_frames > 20:
            strengths.append("Relaxed upper body with no shoulder tension")
        elif self.shoulder_tension_events >= 1:
            improvements.append("Periodically drop and relax your shoulders to relieve upper-body tension")

        if self._last_blink_status == "Elevated (Stress)":
            improvements.append("Elevated blink rate detected — practice steady, calm breathing to ease nervousness")

        # Build the summary string
        lines = [
            "",
            "========================================",
            "       INTERVIEW SESSION SUMMARY",
            "========================================",
            "",
            f"Duration:              {mins:02d}:{secs:02d}",
            f"Frames Analysed:       {self._frame_count}",
            "",
            f"Average Posture:       {avg['posture']:.0f}/100",
            f"Average Eye Contact:   {avg['eye_contact']:.0f}/100",
            f"Average Head Position: {avg['head_position']:.0f}/100",
            f"Average Expression:    {avg['expression']:.0f}/100",
            f"Average Stability:     {avg['stability']:.0f}/100",
            f"Average Gestures:      {avg['gestures']:.0f}/100",
            "",
            f"Overall Readiness:     {avg['overall']:.0f}/100",
            f"Status:                {self._status_label(avg['overall'])}",
            f"Average Reliability:   {avg['reliability']:.0f}%",
            "",
            f"Good Posture:          {good_pct}% of the time",
            f"Poor Posture:          {poor_pct}% of the time",
            f"Posture Warnings:      {self.posture_warnings}",
            f"Posture Corrections:   {self.posture_corrections}",
            f"Shoulder Tension Warnings: {self.shoulder_tension_events}",
            f"Total Blinks:          {self._last_blink_count}",
            f"Blink Rate:            {self._last_blink_rate:.0f} blinks/min ({self._last_blink_status})",
            f"Occasional Distractions: {self.distraction_events}",
            f"Sustained Gaze-Away:   {self.sustained_gaze_away_events}",
            f"Total Time Looking Away: {self._gaze_away_frames / 30.0:.1f}s",
            f"Total Fidgeting Time:  {self._fidgeting_frames / 30.0:.1f}s",
            f"Low Reliability Time:  {self.low_reliability_frames / 30.0:.1f}s",
        ]

        if strengths:
            lines.append("")
            lines.append("Strengths:")
            for s in strengths:
                lines.append(f"  \u2713 {s}")

        if improvements:
            lines.append("")
            lines.append("Areas to Improve:")
            for i in improvements:
                lines.append(f"  \u2022 {i}")
                
        if self.event_log:
            lines.append("")
            lines.append("========================================")
            lines.append("       CHRONOLOGICAL EVENT LOG")
            lines.append("========================================")
            for event in self.event_log:
                lines.append(f"  {event}")

        lines.extend(
            [
                "",
                "========================================",
                "",
                "Note: This tool analyzes visual behavioral cues only.",
                "It does NOT evaluate subjective psychological states",
                "such as internal confidence, intelligence, or honesty.",
                "Use these metrics solely for behavioral awareness.",
                "",
            ]
        )

        return "\n".join(lines)
