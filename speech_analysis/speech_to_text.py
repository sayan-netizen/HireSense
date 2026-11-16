"""
Speech-to-Text conversion module using SpeechRecognition and Google Web Speech API.
"""

import os
from typing import Tuple, Optional
import speech_recognition as sr
from .audio_capture import convert_audio_to_wav
from config.settings import SPEECH_RECOGNITION_LANGUAGE


def transcribe_audio(
    audio_bytes: bytes,
    language: str = SPEECH_RECOGNITION_LANGUAGE,
) -> Tuple[str, Optional[str]]:
    """
    Convert recorded audio bytes into text transcript.
    Returns:
        (transcript, error_message)
    """
    if not audio_bytes:
        return "", "No audio recorded."

    wav_path, err = convert_audio_to_wav(audio_bytes)
    if err:
        return "", err

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        transcript = recognizer.recognize_google(audio_data, language=language)
        return transcript.strip(), None

    except sr.UnknownValueError:
        return "", "Speech could not be understood. Please record your answer clearly."
    except sr.RequestError as e:
        return "", f"Speech recognition service error (requires active internet connection): {e}"
    except Exception as e:
        return "", f"Speech transcription error: {str(e)}"
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
