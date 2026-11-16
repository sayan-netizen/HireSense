"""
Answer Evaluator: Evaluates candidate answers via local Ollama LLM with
schema validation and graceful fallback handling.
"""

from typing import Dict, Any, Optional
from .ollama_client import OllamaClient
from .evaluation_prompts import get_evaluation_system_prompt, build_evaluation_prompt
from .response_validator import validate_evaluation_json


class AnswerEvaluator:
    """Evaluates written and transcribed answers against reference concepts."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        key_concepts: str,
        user_answer: str,
        mode: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate response.
        Attempts local LLM evaluation via Ollama; falls back gracefully to concept heuristic evaluation
        if Ollama is offline.
        """
        if not user_answer or not user_answer.strip():
            raise ValueError("Student answer cannot be empty.")

        prompt = build_evaluation_prompt(
            question=question,
            reference_answer=reference_answer,
            key_concepts=key_concepts,
            user_answer=user_answer,
            mode=mode,
        )

        messages = [
            {"role": "system", "content": get_evaluation_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            raw_response = self.client.chat(messages=messages, model=model, json_format=True)
            evaluation = validate_evaluation_json(raw_response)
            evaluation["evaluation_source"] = "Ollama Local LLM"
            return evaluation

        except Exception as e:
            # Check if it was an Ollama connection error; provide a graceful fallback report
            fallback_eval = self._generate_fallback_evaluation(
                question=question,
                reference_answer=reference_answer,
                key_concepts=key_concepts,
                user_answer=user_answer,
                error_msg=str(e),
            )
            return fallback_eval

    def _generate_fallback_evaluation(
        self,
        question: str,
        reference_answer: str,
        key_concepts: str,
        user_answer: str,
        error_msg: str,
    ) -> Dict[str, Any]:
        """
        Heuristic fallback when Ollama is offline or unavailable.
        Uses key concepts coverage and length analysis to provide actionable feedback.
        """
        concepts = [c.strip().lower() for c in key_concepts.split(";") if c.strip()]
        lower_ans = user_answer.lower()
        matched = [c for c in concepts if c in lower_ans]

        ratio = len(matched) / max(1, len(concepts))
        word_count = len(user_answer.split())

        # Estimated scores based on conceptual keyword coverage
        correctness = min(10.0, max(2.0, round(ratio * 9.0 + 1.0, 1)))
        depth = min(10.0, max(2.0, round(min(1.0, word_count / 40.0) * 8.0 + 2.0, 1)))
        completeness = min(10.0, max(1.0, round(ratio * 10.0, 1)))

        strengths = []
        if matched:
            strengths.append(f"Successfully incorporated key concepts: {', '.join(matched)}")
        if word_count >= 25:
            strengths.append("Provided a substantive explanation with decent elaboration.")
        if not strengths:
            strengths.append("Attempted to address the question prompt.")

        improvements = []
        missing = [c for c in concepts if c not in lower_ans]
        if missing:
            improvements.append(f"Consider integrating expected core concepts: {', '.join(missing)}")
        if word_count < 25:
            improvements.append("Elaborate further with specific examples, mechanics, or reasoning.")

        return {
            "correctness": correctness,
            "depth": depth,
            "completeness": completeness,
            "explanation": (
                f"Evaluation generated via fallback concept analysis because local Ollama LLM was unavailable ({error_msg}). "
                f"Candidate answered with {word_count} words and addressed {len(matched)} of {len(concepts)} expected concepts."
            ),
            "grammar_spelling": "Automatic LLM grammar inspection unavailable in fallback mode.",
            "overall_feedback": (
                "Your answer captured several essential points. To strengthen your response in an interview, "
                "clearly define core terminology first and provide a practical real-world use case or code example."
            ),
            "strengths": strengths,
            "areas_to_improve": improvements,
            "reference_answer": reference_answer,
            "evaluation_source": "Offline Fallback Analysis (Ollama Offline)",
            "ollama_error": error_msg,
        }
