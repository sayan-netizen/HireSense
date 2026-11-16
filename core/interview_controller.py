"""
Interview Controller: Orchestrates interview progression, evaluation dispatch,
adaptive questioning, and cross-engine coordination.
"""

from typing import Dict, Any, Optional
from core.session_manager import InterviewSessionManager
from core.question_manager import QuestionManager
from core.event_manager import EventManager
from ai_interviewer.answer_evaluator import AnswerEvaluator
from ai_interviewer.question_generator import QuestionGenerator
from speech_analysis.speech_metrics import SpeechMetricsAggregator


class InterviewController:
    """Coordinates the full interview session flow."""

    def __init__(
        self,
        session_manager: Optional[InterviewSessionManager] = None,
        question_manager: Optional[QuestionManager] = None,
        evaluator: Optional[AnswerEvaluator] = None,
        question_generator: Optional[QuestionGenerator] = None,
    ):
        self.session_manager = session_manager or InterviewSessionManager()
        self.question_manager = question_manager or QuestionManager()
        self.evaluator = evaluator or AnswerEvaluator()
        self.question_generator = question_generator or QuestionGenerator()
        self.speech_aggregator = SpeechMetricsAggregator()
        self.event_manager = EventManager()

    def setup_interview(
        self,
        candidate_name: str,
        category: str = "Technical",
        subject: str = "All Technical",
        difficulty: str = "Easy",
        answer_mode: str = "Written",
        adaptive_followups: bool = False,
        webcam_enabled: bool = True,
        num_questions: int = 5,
    ) -> None:
        """Filter questions and initialize session."""
        questions = self.question_manager.filter_questions(
            interview_type=category,
            subject=subject,
            difficulty=difficulty,
            num_questions=num_questions,
            randomize=True,
        )

        settings = {
            "category": category,
            "subject": subject,
            "difficulty": difficulty,
            "answer_mode": answer_mode,
            "adaptive_followups": adaptive_followups,
            "webcam_enabled": webcam_enabled,
            "num_questions": len(questions),
        }

        self.session_manager.start_session(
            candidate_name=candidate_name,
            settings=settings,
            questions=questions,
        )

        self.event_manager.log(
            "SESSION",
            f"Interview started for {candidate_name}: {category} - {subject} ({difficulty})",
        )

    def submit_answer(
        self,
        user_answer: str,
        speech_data: Optional[Dict[str, Any]] = None,
        visual_snapshot: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the answer for the current question and stores the record.
        """
        curr_q = self.session_manager.get_current_question()
        if not curr_q:
            raise RuntimeError("No active question to evaluate.")

        curr_idx = self.session_manager.session_data.get("current_index", 0)
        mode = self.session_manager.session_data["settings"].get("answer_mode", "Written")

        # Evaluate Answer Intelligence
        eval_result = self.evaluator.evaluate(
            question=curr_q["question"],
            reference_answer=curr_q["reference_answer"],
            key_concepts=curr_q["key_concepts"],
            user_answer=user_answer,
            mode=mode,
            model=model_name,
        )

        # Aggregate speech data if oral answer
        if speech_data and speech_data.get("filler_analysis"):
            self.speech_aggregator.add_record(
                question_id=curr_q.get("id", curr_idx + 1),
                transcript=user_answer,
                filler_analysis=speech_data["filler_analysis"],
                pronunciation=speech_data.get("pronunciation"),
            )

        # Generate optional follow-up question
        followup = None
        if self.session_manager.session_data["settings"].get("adaptive_followups", False):
            followup = self.question_generator.generate_followup(
                question=curr_q["question"],
                user_answer=user_answer,
                evaluation=eval_result,
                category=curr_q.get("category", "Technical"),
                model=model_name,
            )

        # Record into session manager
        self.session_manager.record_answer(
            question_index=curr_idx,
            question_dict=curr_q,
            user_answer=user_answer,
            evaluation=eval_result,
            speech_data=speech_data,
            visual_snapshot=visual_snapshot,
            followup_question=followup,
        )

        self.event_manager.log(
            "EVALUATION",
            f"Evaluated answer for Q{curr_idx + 1}: Correctness {eval_result.get('correctness')}/10",
        )

        return eval_result
