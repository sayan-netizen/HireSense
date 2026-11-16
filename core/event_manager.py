"""
Event Manager: Central event logger for interview milestones and engine notifications.
"""

import time
from typing import List, Dict, Any


class EventManager:
    """Logs system events, errors, and milestone occurrences."""

    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def log(self, category: str, message: str, level: str = "INFO", metadata: Dict[str, Any] = None) -> None:
        event = {
            "timestamp": time.time(),
            "category": category, # e.g. "SESSION", "VISION", "SPEECH", "LLM"
            "level": level,       # "INFO", "WARNING", "ERROR"
            "message": message,
            "metadata": metadata or {},
        }
        self._events.append(event)

    def get_events(self, category: str = None) -> List[Dict[str, Any]]:
        if category:
            return [e for e in self._events if e["category"] == category]
        return list(self._events)
