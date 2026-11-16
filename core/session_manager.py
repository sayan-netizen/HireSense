"""
Central Interview Session Manager for HireSense.
Coordinates the state across Interview, Speech, and Visual Behaviour engines.
Thread-safe and persistent across Streamlit reruns.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
import threading


class InterviewSessionManager:
    """Manages interview lifecycle, timing, question records, and multi-engine metrics."""

    def __init__(self):
        self.lock = threading.Lock()
        self.session_data: Dict[str, Any] = self._create_empty_session()

    def _create_empty_session(self) -> Dict[str, Any]:
        return {
            "session_id": str(uuid.uuid4())[:8],
            "candidate_name": "Candidate",
            "started_at": None,
            "ended_at": None,
            "duration_seconds": 0.0,
            "is_active": False,
            "is_completed": False,
            "settings": {
                "category": "Technical",
                "subject": "All Technical",
                "difficulty": "Easy",
                "answer_mode": "Written",
                "adaptive_followups": False,
                "webcam_enabled": True,
                "num_questions": 5,
            },
            "questions": [],          # list of question dicts
            "current_index": 0,
            "answers": {},            # question_index -> answer record dict
            "question_intervals": {}, # question_index -> {"start_time": ..., "end_time": ...}
            "speech_metrics": {},     # aggregated session speech metrics
            "visual_metrics": {},     # aggregated session visual summary
            "final_report": None,     # compiled final evaluation report
        }

    def start_session(
        self,
        candidate_name: str,
        settings: Dict[str, Any],
        questions: List[Dict[str, Any]],
    ) -> None:
        """Initialize and start a new interview session."""
        with self.lock:
            self.session_data = self._create_empty_session()
            self.session_data["candidate_name"] = candidate_name or "Candidate"
            self.session_data["started_at"] = time.time()
            self.session_data["is_active"] = True
            self.session_data["is_completed"] = False
            self.session_data["settings"] = settings
            self.session_data["questions"] = questions
            self.session_data["current_index"] = 0
            if questions:
                self.session_data["question_intervals"][0] = {
                    "start_time": time.time(),
                    "end_time": None,
                }

    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """Return the current active question dictionary."""
        with self.lock:
            idx = self.session_data.get("current_index", 0)
            qs = self.session_data.get("questions", [])
            if 0 <= idx < len(qs):
                return qs[idx]
            return None

    def record_question_start(self, index: int) -> None:
        """Mark start timestamp for question interval."""
        with self.lock:
            if index not in self.session_data["question_intervals"]:
                self.session_data["question_intervals"][index] = {
                    "start_time": time.time(),
                    "end_time": None,
                }

    def record_answer(
        self,
        question_index: int,
        question_dict: Dict[str, Any],
        user_answer: str,
        evaluation: Dict[str, Any],
        speech_data: Optional[Dict[str, Any]] = None,
        visual_snapshot: Optional[Dict[str, Any]] = None,
        followup_question: Optional[str] = None,
    ) -> None:
        """Store evaluation and metric data for a submitted answer."""
        with self.lock:
            now = time.time()
            if question_index in self.session_data["question_intervals"]:
                self.session_data["question_intervals"][question_index]["end_time"] = now

            record = {
                "question_index": question_index,
                "question": question_dict.get("question", ""),
                "category": question_dict.get("category", ""),
                "subject": question_dict.get("subject", ""),
                "difficulty": question_dict.get("difficulty", ""),
                "reference_answer": question_dict.get("reference_answer", ""),
                "key_concepts": question_dict.get("key_concepts", ""),
                "user_answer": user_answer,
                "submitted_at": now,
                "evaluation": evaluation,
                "speech_data": speech_data,
                "visual_snapshot": visual_snapshot,
                "followup_question": followup_question,
            }
            self.session_data["answers"][question_index] = record

    def advance_question(self) -> bool:
        """
        Advance to the next question.
        Returns True if advanced, False if reached the end of questions.
        """
        with self.lock:
            curr = self.session_data.get("current_index", 0)
            next_idx = curr + 1
            if next_idx < len(self.session_data.get("questions", [])):
                self.session_data["current_index"] = next_idx
                self.session_data["question_intervals"][next_idx] = {
                    "start_time": time.time(),
                    "end_time": None,
                }
                return True
            else:
                self.session_data["is_active"] = False
                self.session_data["is_completed"] = True
                self.session_data["ended_at"] = time.time()
                started = self.session_data.get("started_at") or time.time()
                self.session_data["duration_seconds"] = self.session_data["ended_at"] - started
                return False

    def finish_session(
        self,
        speech_summary: Optional[Dict[str, Any]] = None,
        visual_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mark interview as finished and attach global engine summaries."""
        with self.lock:
            self.session_data["is_active"] = False
            self.session_data["is_completed"] = True
            if not self.session_data.get("ended_at"):
                self.session_data["ended_at"] = time.time()
            started = self.session_data.get("started_at") or self.session_data["ended_at"]
            self.session_data["duration_seconds"] = self.session_data["ended_at"] - started

            if speech_summary:
                self.session_data["speech_metrics"] = speech_summary
            if visual_summary:
                self.session_data["visual_metrics"] = visual_summary

            return dict(self.session_data)

    def get_session_data(self) -> Dict[str, Any]:
        """Return a copy of the entire session data dictionary."""
        with self.lock:
            return dict(self.session_data)
