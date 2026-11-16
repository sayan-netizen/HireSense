"""
Telemetry Server for HireSense

Provides a lightweight, zero-latency in-process HTTP endpoint on 127.0.0.1:8502
serving real-time visual biomechanical telemetry from PostureEngine.
Allows client-side DOM components in Streamlit to poll and update live metrics
at 4-10 FPS without triggering Streamlit script reruns or interrupting typing.
"""

import http.server
import json
import socketserver
import threading
import time
from typing import Any, Callable, Dict, Optional

DEFAULT_TELEMETRY_PORT = 8502


class TelemetryHTTPHandler(http.server.BaseHTTPRequestHandler):
    get_metrics_callback: Optional[Callable[[], Dict[str, Any]]] = None
    _last_cache_time: float = 0.0
    _cached_body: bytes = b"{}"

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/telemetry"):
            now = time.time()
            if now - TelemetryHTTPHandler._last_cache_time > 0.08:
                data: Dict[str, Any] = {}
                if TelemetryHTTPHandler.get_metrics_callback:
                    try:
                        data = TelemetryHTTPHandler.get_metrics_callback() or {}
                    except Exception:
                        data = {}

                scores = data.get("scores", {})
                posture = data.get("posture", {})
                face = data.get("face", {})
                hand = data.get("hand", {})

                # Clean payload formatted for the frontend cards
                gesture_val = scores.get("gesture_score")
                if gesture_val is None:
                    gesture_val = scores.get("gestures", 85)

                payload = {
                    "posture": round(float(scores.get("posture", 75))),
                    "eye_contact": round(float(scores.get("eye_contact", 85))),
                    "gaze_direction": str(face.get("gaze_direction", "CENTERED")).upper(),
                    "head_position": round(float(scores.get("head_position", 90))),
                    "gesture_score": round(float(gesture_val)),
                    "reliability": round(float(scores.get("reliability", 100))),
                    "lean_direction": str(posture.get("lean_direction", "Balanced")),
                    "is_calibrated": bool(data.get("is_calibrated", False)),
                    "calibration_progress": round(float(data.get("calibration_progress", 0.0))),
                    "fps": round(float(data.get("fps", 0.0)), 1),
                    "timestamp": now,
                }
                TelemetryHTTPHandler._cached_body = json.dumps(payload).encode("utf-8")
                TelemetryHTTPHandler._last_cache_time = now

            body = TelemetryHTTPHandler._cached_body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/health"):
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence console log spam for sub-second telemetry polling
        pass


class FastHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


_server_instance: Optional[FastHTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_server_lock = threading.Lock()


def start_telemetry_server(
    get_metrics_fn: Callable[[], Dict[str, Any]],
    port: int = DEFAULT_TELEMETRY_PORT,
) -> bool:
    """Start the telemetry background server if not already running."""
    global _server_instance, _server_thread

    with _server_lock:
        TelemetryHTTPHandler.get_metrics_callback = get_metrics_fn

        if _server_instance is not None and _server_thread and _server_thread.is_alive():
            return True

        try:
            try:
                server = FastHTTPServer(("0.0.0.0", port), TelemetryHTTPHandler)
            except OSError:
                server = FastHTTPServer(("127.0.0.1", port), TelemetryHTTPHandler)
            thread = threading.Thread(
                target=server.serve_forever,
                name="HireSenseTelemetryServer",
                daemon=True,
            )
            thread.start()
            _server_instance = server
            _server_thread = thread
            return True
        except OSError:
            return False
