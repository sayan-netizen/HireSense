"""
Optional audio pronunciation analysis using OpenPronounce.
Preserves the exact OpenPronounce integration logic from app.py.
"""

import os
from typing import Dict, Any, Tuple, Optional
from .audio_capture import convert_audio_to_wav


def is_openpronounce_available() -> bool:
    """Check if openpronounce package is importable."""
    try:
        import openpronounce
        return True
    except ImportError:
        return False


def analyze_pronunciation(audio_bytes: bytes, transcript: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Attempt audio-based pronunciation analysis with OpenPronounce.
    Returns:
        (result_dict, error_message)
    """
    try:
        from openpronounce import load_audio, compare_audio_with_text
    except ImportError:
        return None, (
            "OpenPronounce is not installed in this Python environment. "
            "To enable pronunciation assessment, run: pip install openpronounce"
        )

    if not audio_bytes:
        return None, "No recorded audio available."

    if not transcript or not transcript.strip():
        return None, "A valid transcript is required for pronunciation comparison."

    wav_path, err = convert_audio_to_wav(audio_bytes)
    if err:
        return None, err

    try:
        sound = load_audio(wav_path)
        result = compare_audio_with_text(sound, transcript)
        if not isinstance(result, dict):
            return None, "Pronunciation tool returned an unexpected non-dict result."
        return result, None
    except Exception as e:
        return None, f"Pronunciation analysis failed: {str(e)}"
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
