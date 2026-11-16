"""
Question Manager: Loads, validates, and filters interview questions from questions.csv.
Normalizes subject aliases (e.g. OS vs Operating Systems) and prevents session repetition.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
from config.settings import (
    QUESTIONS_CSV_PATH,
    SUBJECT_NORMALIZATION,
    AVAILABLE_TECHNICAL_SUBJECTS,
    AVAILABLE_HR_SUBJECTS,
)


class QuestionManager:
    """Manages interview question repository and session filtering."""

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or str(QUESTIONS_CSV_PATH)
        self.df = self._load_and_validate(self.csv_path)

    def _load_and_validate(self, path: str) -> pd.DataFrame:
        """Loads and normalizes questions DataFrame."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Questions database not found at: {path}")

        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        required = [
            "id",
            "category",
            "subject",
            "difficulty",
            "question",
            "reference_answer",
            "key_concepts",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in questions.csv: {', '.join(missing)}")

        for col in required:
            df[col] = df[col].fillna("").astype(str).str.strip()

        # Normalize subjects
        df["subject_normalized"] = df["subject"].apply(
            lambda s: SUBJECT_NORMALIZATION.get(s.lower(), s)
        )

        return df

    def get_available_subjects(self, interview_type: str) -> List[str]:
        """Return valid subject filters for the given interview category."""
        if interview_type == "Technical":
            return ["All Technical"] + AVAILABLE_TECHNICAL_SUBJECTS
        elif interview_type == "HR":
            return ["General"]
        else: # Mixed
            return ["All Subjects"] + AVAILABLE_TECHNICAL_SUBJECTS + AVAILABLE_HR_SUBJECTS

    def filter_questions(
        self,
        interview_type: str = "Technical",
        subject: str = "All Subjects",
        difficulty: str = "Easy",
        num_questions: Optional[int] = None,
        randomize: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Filter questions matching criteria and return as a list of dicts.
        """
        filtered = self.df.copy()

        # Category filter
        if interview_type.lower() == "technical":
            filtered = filtered[filtered["category"].str.casefold() == "technical"]
        elif interview_type.lower() == "hr":
            filtered = filtered[filtered["category"].str.casefold() == "hr"]

        # Subject filter
        if subject not in ("All Subjects", "All Technical", "All"):
            normalized_target = SUBJECT_NORMALIZATION.get(subject.lower(), subject.lower())
            filtered = filtered[
                filtered["subject_normalized"].str.casefold() == normalized_target.casefold()
            ]

        # Difficulty filter
        if difficulty not in ("All", "Any"):
            filtered = filtered[filtered["difficulty"].str.casefold() == difficulty.casefold()]

        if filtered.empty:
            # Fallback if no exact difficulty match: match category and subject without difficulty restriction
            fallback = self.df.copy()
            if interview_type.lower() == "technical":
                fallback = fallback[fallback["category"].str.casefold() == "technical"]
            elif interview_type.lower() == "hr":
                fallback = fallback[fallback["category"].str.casefold() == "hr"]

            # Preserve subject filter during fallback
            if subject not in ("All Subjects", "All Technical", "All"):
                normalized_target = SUBJECT_NORMALIZATION.get(subject.lower(), subject.lower())
                subject_match = fallback[
                    fallback["subject_normalized"].str.casefold() == normalized_target.casefold()
                ]
                if not subject_match.empty:
                    fallback = subject_match

            if not fallback.empty:
                filtered = fallback

        if randomize:
            filtered = filtered.sample(frac=1.0)

        if num_questions and num_questions > 0:
            filtered = filtered.head(num_questions)

        records = filtered.to_dict(orient="records")
        return records
