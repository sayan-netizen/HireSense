"""
Unit tests for Speech Analysis: filler words, audio utilities, and metric aggregation.
"""

import unittest
from speech_analysis.filler_analyzer import analyze_fillers
from speech_analysis.speech_metrics import SpeechMetricsAggregator


class TestSpeechEngine(unittest.TestCase):

    def test_filler_word_detection(self):
        transcript = "Um, so basically, like, a stack is you know a LIFO structure actually."
        result = analyze_fillers(transcript)

        self.assertTrue(result["fillers_detected"])
        self.assertGreater(result["total"], 0)
        self.assertIn("um", result["counts"])
        self.assertIn("basically", result["counts"])
        self.assertIn("like", result["counts"])
        self.assertIn("you know", result["counts"])
        self.assertIn("actually", result["counts"])
        self.assertGreater(result["rate_per_100_words"], 0.0)

    def test_empty_transcript(self):
        result = analyze_fillers("")
        self.assertFalse(result["fillers_detected"])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["word_count"], 0)

    def test_clean_speech_no_fillers(self):
        clean_text = "Binary search repeatedly divides the search interval in half."
        result = analyze_fillers(clean_text)
        self.assertFalse(result["fillers_detected"])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["rate_per_100_words"], 0.0)

    def test_speech_aggregator(self):
        aggregator = SpeechMetricsAggregator()
        analysis1 = analyze_fillers("Um, this is basically the first response.")
        analysis2 = analyze_fillers("Actually, like, this is the second response.")

        aggregator.add_record(1, "Um, this is basically the first response.", analysis1)
        aggregator.add_record(2, "Actually, like, this is the second response.", analysis2)

        summary = aggregator.get_summary()
        self.assertGreater(summary["total_words"], 0)
        self.assertGreater(summary["total_fillers"], 0)
        self.assertIn("um", summary["top_filler_words"])
        self.assertIn("basically", summary["top_filler_words"])

    def test_speech_aggregator_re_evaluation_no_duplicate(self):
        aggregator = SpeechMetricsAggregator()
        analysis1 = analyze_fillers("Um, this is basically the first response.")
        analysis1_updated = analyze_fillers("This is a clean updated first response with no fillers.")

        aggregator.add_record(1, "Um, this is basically the first response.", analysis1)
        self.assertEqual(len(aggregator.question_speech_records), 1)
        self.assertEqual(aggregator.question_speech_records[0]["filler_analysis"]["total"], 2)

        # Re-submitting for question 1 should replace, not append
        aggregator.add_record(1, "This is a clean updated first response with no fillers.", analysis1_updated)
        self.assertEqual(len(aggregator.question_speech_records), 1)
        self.assertEqual(aggregator.question_speech_records[0]["filler_analysis"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
