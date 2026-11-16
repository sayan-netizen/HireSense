"""
Data models and aggregators for speech evaluation metrics.
"""

from typing import Dict, Any, List


class SpeechMetricsAggregator:
    """Aggregates speech metrics across all answered questions in an interview session."""

    def __init__(self):
        self.question_speech_records: List[Dict[str, Any]] = []

    def add_record(
        self,
        question_id: int,
        transcript: str,
        filler_analysis: Dict[str, Any],
        pronunciation: Dict[str, Any] = None,
    ) -> None:
        record = {
            "question_id": question_id,
            "transcript": transcript,
            "filler_analysis": filler_analysis,
            "pronunciation": pronunciation,
        }
        # Update existing record if question was re-submitted to prevent duplicate aggregation
        for i, existing in enumerate(self.question_speech_records):
            if existing.get("question_id") == question_id:
                self.question_speech_records[i] = record
                return
        self.question_speech_records.append(record)

    def get_summary(self) -> Dict[str, Any]:
        """Compute session-wide speech totals and averages."""
        if not self.question_speech_records:
            return {
                "total_words": 0,
                "total_fillers": 0,
                "overall_filler_rate": 0.0,
                "top_filler_words": {},
                "pronunciation_available": False,
                "average_pronunciation_score": None,
                "flagged_words_count": 0,
            }

        total_words = sum(r["filler_analysis"].get("word_count", 0) for r in self.question_speech_records)
        total_fillers = sum(r["filler_analysis"].get("total", 0) for r in self.question_speech_records)
        overall_filler_rate = round(total_fillers / total_words * 100.0, 1) if total_words > 0 else 0.0

        all_filler_counts = {}
        for r in self.question_speech_records:
            for word, cnt in r["filler_analysis"].get("counts", {}).items():
                all_filler_counts[word] = all_filler_counts.get(word, 0) + cnt

        # Pronunciation scores if present
        pron_scores = [
            r["pronunciation"]["score"]
            for r in self.question_speech_records
            if r.get("pronunciation") and isinstance(r["pronunciation"].get("score"), (int, float))
        ]
        avg_pron = round(sum(pron_scores) / len(pron_scores), 1) if pron_scores else None

        flagged_words = 0
        for r in self.question_speech_records:
            if r.get("pronunciation") and isinstance(r["pronunciation"].get("differences"), dict):
                flagged_words += len(r["pronunciation"]["differences"].get("errors", []))

        return {
            "total_words": total_words,
            "total_fillers": total_fillers,
            "overall_filler_rate": overall_filler_rate,
            "top_filler_words": all_filler_counts,
            "pronunciation_available": len(pron_scores) > 0,
            "average_pronunciation_score": avg_pron,
            "flagged_words_count": flagged_words,
        }
