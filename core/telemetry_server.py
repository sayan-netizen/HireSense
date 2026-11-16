"""
Telemetry Server for HireSense

Provides a lightweight, zero-latency in-process HTTP endpoint on 127.0.0.1:8502
serving real-time visual biomechanical telemetry from PostureEngine.
Allows client-side DOM components in Streamlit to poll and update live metrics
at 4-10 FPS without triggering Streamlit script reruns or interrupting typing.
"""

import http.server
import json
import os
import socketserver
import threading
import time
from typing import Any, Callable, Dict, Optional

DEFAULT_TELEMETRY_PORT = 8502


class TelemetryHTTPHandler(http.server.BaseHTTPRequestHandler):
    get_metrics_callback: Optional[Callable[[], Dict[str, Any]]] = None
    _last_cache_time: float = 0.0
    _cached_body: bytes = b"{}"

    def _get_allowed_origin(self) -> str:
        """Restrict CORS to local Streamlit origins only, preventing malicious third-party scraping."""
        origin = self.headers.get("Origin", "")
        if origin and ("localhost" in origin or "127.0.0.1" in origin):
            return origin
        return "http://localhost:8501"

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", self._get_allowed_origin())
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        allowed_origin = self._get_allowed_origin()
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
                fps_val = float(data.get("fps", 0.0))
                is_calibrated = bool(data.get("is_calibrated", False))

                # Tracking is active only when camera is actively processing frames with detections
                tracking_active = bool(scores and (fps_val > 0.0 or data.get("session_time", 0.0) > 0.0))

                posture_val = scores.get("posture")
                eye_val = scores.get("eye_contact")
                head_val = scores.get("head_position")
                gesture_val = scores.get("gesture_score")
                if gesture_val is None:
                    gesture_val = scores.get("gestures")
                rel_val = scores.get("reliability")

                payload = {
                    "posture": round(float(posture_val)) if (posture_val is not None and tracking_active) else 0,
                    "eye_contact": round(float(eye_val)) if (eye_val is not None and tracking_active) else 0,
                    "gaze_direction": str(face.get("gaze_direction", "CENTERED")).upper() if tracking_active else "N/A",
                    "head_position": round(float(head_val)) if (head_val is not None and tracking_active) else 0,
                    "gesture_score": round(float(gesture_val)) if (gesture_val is not None and tracking_active) else 0,
                    "reliability": round(float(rel_val)) if (rel_val is not None and tracking_active) else 0,
                    "lean_direction": str(posture.get("lean_direction", "Balanced")) if tracking_active else "N/A",
                    "is_calibrated": is_calibrated,
                    "calibration_progress": round(float(data.get("calibration_progress", 0.0))),
                    "fps": round(fps_val, 1),
                    "tracking_active": tracking_active,
                    "timestamp": now,
                }
                TelemetryHTTPHandler._cached_body = json.dumps(payload).encode("utf-8")
                TelemetryHTTPHandler._last_cache_time = now

            body = TelemetryHTTPHandler._cached_body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/health"):
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
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
    """Start the telemetry background server bound strictly to localhost (127.0.0.1) for local security."""
    global _server_instance, _server_thread

    with _server_lock:
        TelemetryHTTPHandler.get_metrics_callback = get_metrics_fn

        if _server_instance is not None and _server_thread and _server_thread.is_alive():
            return True

        # Default bind to 127.0.0.1 for local isolation. Opt-in to 0.0.0.0 via HIRESENSE_BIND_ALL=1.
        bind_host = "0.0.0.0" if os.environ.get("HIRESENSE_BIND_ALL", "0") == "1" else "127.0.0.1"

        try:
            server = FastHTTPServer((bind_host, port), TelemetryHTTPHandler)
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
