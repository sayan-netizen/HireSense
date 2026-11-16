"""
Audio Capture and Conversion Utilities.
Converts browser WebM audio to 16 kHz mono WAV for speech recognition and pronunciation analysis.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Tuple, Optional


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg executable is available on PATH."""
    return shutil.which("ffmpeg") is not None


def convert_audio_to_wav(audio_bytes: bytes, target_sample_rate: int = 16000) -> Tuple[str, Optional[str]]:
    """
    Saves raw audio bytes to a temporary file and converts it to a standard WAV file.
    Returns:
        (wav_file_path, error_message)
    Caller is responsible for removing the returned file when finished.
    """
    if not audio_bytes:
        return "", "No audio bytes provided."

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_src:
        temp_src.write(audio_bytes)
        source_path = temp_src.name

    wav_path = source_path + ".wav"
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        try:
            cmd = [
                ffmpeg, "-y", "-i", source_path,
                "-ac", "1", "-ar", str(target_sample_rate), wav_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if res.returncode == 0 and os.path.exists(wav_path):
                # Clean up source webm
                try:
                    os.remove(source_path)
                except OSError:
                    pass
                return wav_path, None
            else:
                return "", f"FFmpeg audio conversion failed: {res.stderr}"
        except Exception as e:
            return "", f"FFmpeg execution error: {str(e)}"
        finally:
            if os.path.exists(source_path):
                try:
                    os.remove(source_path)
                except OSError:
                    pass
    elif source_path.lower().endswith(".wav"):
        return source_path, None
    else:
        # Clean up source
        if os.path.exists(source_path):
            try:
                os.remove(source_path)
            except OSError:
                pass
        return "", (
            "FFmpeg is not installed or not in PATH. "
            "Please install FFmpeg or add it to system PATH to process browser WebM microphone recordings."
        )
