"""
Unit tests for TelemetryHTTPHandler security and metric handling.
"""

import json
import unittest
from unittest.mock import MagicMock
from core.telemetry_server import TelemetryHTTPHandler


class TestTelemetrySecurity(unittest.TestCase):

    def test_cors_restriction_local_origin(self):
        handler = TelemetryHTTPHandler.__new__(TelemetryHTTPHandler)
        handler.headers = {"Origin": "http://localhost:8501"}
        allowed = handler._get_allowed_origin()
        self.assertEqual(allowed, "http://localhost:8501")

        handler.headers = {"Origin": "http://127.0.0.1:8501"}
        allowed = handler._get_allowed_origin()
        self.assertEqual(allowed, "http://127.0.0.1:8501")

    def test_cors_restriction_rejects_external_origin(self):
        handler = TelemetryHTTPHandler.__new__(TelemetryHTTPHandler)
        handler.headers = {"Origin": "https://evil-hacker-site.com"}
        allowed = handler._get_allowed_origin()
        self.assertEqual(allowed, "http://localhost:8501")

    def test_inactive_tracking_zero_metrics(self):
        # When engine has not processed frames, tracking_active is False and scores default to 0
        TelemetryHTTPHandler.get_metrics_callback = lambda: {
            "fps": 0.0,
            "session_time": 0.0,
            "is_calibrated": False,
            "scores": {},
            "posture": {},
            "face": {},
            "hand": {},
        }
        TelemetryHTTPHandler._last_cache_time = 0.0  # Invalidate cache

        # Call do_GET logic via mock
        handler = TelemetryHTTPHandler.__new__(TelemetryHTTPHandler)
        handler.headers = {"Origin": "http://localhost:8501"}
        handler.path = "/telemetry"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()

        handler.do_GET()

        handler.send_response.assert_called_with(200)
        # Check written json
        written_bytes = handler.wfile.write.call_args[0][0]
        data = json.loads(written_bytes.decode("utf-8"))
        self.assertFalse(data["tracking_active"])
        self.assertEqual(data["posture"], 0)
        self.assertEqual(data["eye_contact"], 0)
        self.assertEqual(data["reliability"], 0)


if __name__ == "__main__":
    unittest.main()
