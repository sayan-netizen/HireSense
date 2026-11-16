"""
Speech Analysis Package for HireSense.
"""

from .audio_capture import convert_audio_to_wav, is_ffmpeg_available
from .speech_to_text import transcribe_audio
from .filler_analyzer import analyze_fillers
from .pronunciation_analyzer import analyze_pronunciation, is_openpronounce_available
from .speech_metrics import SpeechMetricsAggregator

__all__ = [
    "convert_audio_to_wav",
    "is_ffmpeg_available",
    "transcribe_audio",
    "analyze_fillers",
    "analyze_pronunciation",
    "is_openpronounce_available",
    "SpeechMetricsAggregator",
]
