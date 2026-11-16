"""
Adaptive Question Generator for HireSense.
Generates follow-up and clarification questions using local LLM based on interview state.
"""

from typing import Optional, Dict, Any
from .ollama_client import OllamaClient
from .evaluation_prompts import build_followup_prompt


class QuestionGenerator:
    """Generates dynamic follow-up interview questions."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    def generate_followup(
        self,
        question: str,
        user_answer: str,
        evaluation: Dict[str, Any],
        category: str = "Technical",
        model: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate an adaptive follow-up question.
        Returns a question string or None if generation failed.
        """
        prompt = build_followup_prompt(
            question=question,
            user_answer=user_answer,
            evaluation=evaluation,
            category=category,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a professional mock interviewer. Ask one concise follow-up question.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat(
                messages=messages,
                model=model,
                json_format=False,
                temperature=0.3,
                num_predict=150,
            )
            # Clean up response
            clean_q = response.strip().strip('"').strip("'")
            if clean_q and len(clean_q) > 10:
                return clean_q
            return None
        except Exception:
            return None
