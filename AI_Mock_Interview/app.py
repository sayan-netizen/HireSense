
import os
import re
import json
import socket
import urllib.request
import urllib.error
import tempfile
import time
from collections import Counter

import streamlit as st
import pandas as pd
import speech_recognition as sr


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

st.set_page_config(
    page_title="AI Mock Interview",
    page_icon="🎓",
    layout="wide"
)

st.title("AI-Based Mock Interview System")
st.write(
    "Practice technical and HR interviews with written and oral feedback."
)


# ==================================================
# LOAD QUESTIONS
# ==================================================

@st.cache_data
def load_questions():
    df = pd.read_csv("questions.csv")
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = [
        "category",
        "subject",
        "difficulty",
        "question",
        "reference_answer",
        "key_concepts"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing columns in questions.csv: " + ", ".join(missing)
        )

    for column in required:
        df[column] = (
            df[column].fillna("").astype(str).str.strip()
        )

    return df


try:
    questions_df = load_questions()
except Exception as e:
    st.error(f"Could not load questions.csv: {e}")
    st.stop()


# ==================================================
# INTERVIEW SETTINGS
# ==================================================

st.sidebar.header("Interview Settings")

interview_type = st.sidebar.selectbox(
    "Interview Type", ["Technical", "HR", "Mixed"]
)

all_subjects = [
    "DSA", "DBMS", "Operating Systems",
    "Computer Networks", "OOP", "General"
]

if interview_type == "Technical":
    available_subjects = all_subjects[:-1]
elif interview_type == "HR":
    available_subjects = ["General"]
else:
    available_subjects = all_subjects

subject = st.sidebar.selectbox("Subject", available_subjects)

difficulty = st.sidebar.selectbox(
    "Difficulty", ["Easy", "Medium", "Hard"]
)

answer_mode = st.sidebar.radio("Answer Mode", ["Written", "Oral"])


# ==================================================
# FILTER QUESTIONS
# ==================================================

df = questions_df.copy()

if interview_type == "Technical":
    df = df[df["category"].str.casefold() == "technical"]
elif interview_type == "HR":
    df = df[df["category"].str.casefold() == "hr"]

if subject != "General":
    df = df[df["subject"].str.casefold() == subject.casefold()]

df = df[df["difficulty"].str.casefold() == difficulty.casefold()]

if df.empty:
    st.warning(
        "No matching questions found. Check category, subject and "
        "difficulty in questions.csv."
    )
    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "question_index": 0,
    "evaluation": None,
    "transcript": "",
    "audio_bytes": None,
    "audio_duration": None,
    "pronunciation": None,
    "settings": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

settings = (interview_type, subject, difficulty, answer_mode)

if st.session_state.settings != settings:
    st.session_state.settings = settings
    st.session_state.question_index = 0
    st.session_state.evaluation = None
    st.session_state.transcript = ""
    st.session_state.audio_bytes = None
    st.session_state.audio_duration = None
    st.session_state.pronunciation = None

if st.session_state.question_index >= len(df):
    st.session_state.question_index = 0

question_row = df.iloc[st.session_state.question_index]


# ==================================================
# QUESTION DISPLAY
# ==================================================

st.subheader(
    f"Question {st.session_state.question_index + 1} of {len(df)}"
)
st.info(question_row["question"])
st.caption(
    f"Subject: {question_row['subject']} | "
    f"Difficulty: {question_row['difficulty']}"
)


# ==================================================
# QWEN EVALUATION
# ==================================================

def evaluate_with_llm(
    question,
    reference_answer,
    key_concepts,
    user_answer,
    mode
):
    prompt = f"""
You are a fair AI mock interview evaluator.

Evaluate the student's answer against the question, reference answer,
and key concepts. Accept correct explanations written in different words.
Do not require the student to copy the reference answer verbatim.
Do not invent facts or penalize an answer for information the question
does not require.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

KEY CONCEPTS:
{key_concepts}

STUDENT ANSWER:
{user_answer}

MODE: {mode}

Return ONLY a valid JSON object with exactly these keys:
{{
  "correctness": 0,
  "depth": 0,
  "completeness": 0,
  "explanation": "",
  "grammar_spelling": "",
  "overall_feedback": "",
  "strengths": [],
  "areas_to_improve": [],
  "reference_answer": ""
}}

Rules:
- correctness, depth, completeness must be scores from 0 to 10.
- correctness measures factual accuracy and relevance.
- depth measures reasoning, examples, and explanation.
- completeness measures coverage of the key concepts.
- grammar_spelling identifies actual spelling, grammar, or sentence
  clarity issues. Quote examples only if present in the student's answer.
- For oral mode, the input is an automatically recognized transcript.
  Do not infer pronunciation, voice quality, or fluency from text.
- Give specific, constructive feedback.
- Include a reference answer based on the supplied reference answer.
- Do not claim a fact is missing if it is not needed for the question.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You evaluate interview answers fairly. "
                    "Return only valid JSON."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 700,
            "num_ctx": 4096
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

        content = result.get("message", {}).get("content", "").strip()

        if not content:
            raise RuntimeError("Qwen returned an empty response.")

        evaluation = json.loads(content)

        required_keys = [
            "correctness", "depth", "completeness", "explanation",
            "grammar_spelling", "overall_feedback", "strengths",
            "areas_to_improve", "reference_answer"
        ]

        missing = [k for k in required_keys if k not in evaluation]
        if missing:
            raise RuntimeError(
                "Evaluation is missing fields: " + ", ".join(missing)
            )

        for key in ["correctness", "depth", "completeness"]:
            evaluation[key] = max(
                0.0, min(10.0, float(evaluation[key]))
            )

        for key in ["strengths", "areas_to_improve"]:
            if not isinstance(evaluation[key], list):
                evaluation[key] = [str(evaluation[key])]

        return evaluation

    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP error {e.code}: {details}"
        ) from e

    except urllib.error.URLError as e:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from e

    except (TimeoutError, socket.timeout) as e:
        raise RuntimeError(
            "Qwen took too long to respond. Please try again."
        ) from e

    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            "Qwen did not return a valid evaluation. Please retry."
        ) from e


# ==================================================
# FILLER WORD ANALYSIS
# ==================================================

def analyze_fillers(transcript):
    patterns = {
        "um": r"\bum+\b",
        "uh": r"\buh+\b",
        "erm": r"\berm+\b",
        "like": r"\blike\b",
        "you know": r"\byou know\b",
        "actually": r"\bactually\b",
        "basically": r"\bbasically\b",
        "I mean": r"\bi mean\b",
        "sort of": r"\bsort of\b",
        "kind of": r"\bkind of\b"
    }

    counts = {
        word: len(re.findall(pattern, transcript, re.I))
        for word, pattern in patterns.items()
    }
    counts = {word: n for word, n in counts.items() if n > 0}

    words = re.findall(r"\b[\w']+\b", transcript)

    return {
        "counts": counts,
        "total": sum(counts.values()),
        "word_count": len(words)
    }


def show_filler_report(transcript):
    report = analyze_fillers(transcript)

    st.subheader("Filler-Word Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Detected filler words", report["total"])
    col2.metric("Transcript words", report["word_count"])

    if report["counts"]:
        for word, count in report["counts"].items():
            st.write(f"- **{word}**: {count}")
    else:
        st.success("No common filler words were detected.")

    if report["word_count"]:
        rate = report["total"] / report["word_count"] * 100
        st.caption(f"Detected fillers per 100 words: {rate:.1f}")

    st.caption(
        "This count is based on the recognized transcript. "
        "Speech recognition may omit or misrecognize words. "
        "Words such as 'like' and 'so' can be meaningful, not fillers."
    )


# ==================================================
# PRONUNCIATION ANALYSIS
# ==================================================

def analyze_pronunciation(audio_bytes, transcript):
    """
    Attempt audio-based pronunciation analysis with OpenPronounce.
    Returns a result dictionary or raises a descriptive error.
    """
    try:
        from openpronounce import load_audio, compare_audio_with_text
    except ImportError as e:
        raise RuntimeError(
            "OpenPronounce is not available in this Python environment. "
            "Activate your project venv and check the installation."
        ) from e

    if not audio_bytes:
        raise RuntimeError("No recorded audio is available.")

    if not transcript.strip():
        raise RuntimeError("A transcript is needed for comparison.")

    temp_path = None

    try:
        # Streamlit recordings may be WebM or WAV. OpenPronounce expects
        # a supported audio file, so FFmpeg is used when conversion is needed.
        with tempfile.NamedTemporaryFile(
            suffix=".webm", delete=False
        ) as temp:
            temp.write(audio_bytes)
            source_path = temp.name

        temp_path = source_path

        # Convert to WAV if FFmpeg is available.
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")

        if ffmpeg:
            wav_path = source_path + ".wav"
            completed = subprocess.run(
                [
                    ffmpeg, "-y", "-i", source_path,
                    "-ac", "1", "-ar", "16000", wav_path
                ],
                capture_output=True,
                text=True,
                timeout=90
            )

            if completed.returncode == 0 and os.path.exists(wav_path):
                temp_path = wav_path
            else:
                raise RuntimeError(
                    "Audio conversion failed. Please check FFmpeg "
                    "and record the answer again."
                )

        elif source_path.lower().endswith(".wav"):
            temp_path = source_path
        else:
            raise RuntimeError(
                "FFmpeg was not found in this terminal's PATH. "
                "Restart VS Code after adding FFmpeg, then try again."
            )

        sound = load_audio(temp_path)
        result = compare_audio_with_text(sound, transcript)

        if not isinstance(result, dict):
            raise RuntimeError("Pronunciation tool returned an unexpected result.")

        return result

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            "Pronunciation analysis could not complete. "
            "The model may need its first-time downloads, or the audio "
            f"format may not be supported. Details: {e}"
        ) from e

    finally:
        # Clean up temporary files after analysis.
        for path in {temp_path}:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def show_pronunciation_report(result):
    st.subheader("Pronunciation Assessment")

    score = result.get("score")

    if isinstance(score, (int, float)):
        st.metric("Estimated pronunciation score", f"{score:.1f}/100")
        st.progress(max(0, min(100, int(score))))

    else:
        st.info("The pronunciation tool did not return a numeric score.")

    differences = result.get("differences", {})
    errors = differences.get("errors", []) if isinstance(differences, dict) else []

    if errors:
        st.write("**Words the tool flagged for review:**")

        for item in errors[:20]:
            if not isinstance(item, dict):
                continue

            word = item.get("word", "Unknown word")
            expected = item.get("expected", "")
            actual = item.get("actual", "")
            confidence = item.get("confidence")

            message = f"- **{word}**"
            if expected:
                message += f" — expected: `{expected}`"
            if actual:
                message += f", detected: `{actual}`"
            else:
                message += ", possibly not recognized"

            if isinstance(confidence, (int, float)):
                message += f" (confidence: {confidence:.2f})"

            st.write(message)

    else:
        st.write(
            "No word-level pronunciation differences were returned. "
            "This does not guarantee perfect pronunciation."
        )

    st.caption(
        "This score is an automated estimate. Accent, noise, transcript "
        "errors, and model limitations can cause false positives. "
        "Review flagged words before treating them as mistakes."
    )


# ==================================================
# ANSWER INPUT
# ==================================================

user_answer = ""

if answer_mode == "Written":
    user_answer = st.text_area(
        "Write your answer below",
        height=180,
        key=f"written_{st.session_state.question_index}"
    )

else:
    st.write("Record your answer using your microphone.")
    audio_file = st.audio_input("Record your answer")

    if audio_file is not None:
        # Keep the actual audio bytes for pronunciation assessment.
        st.session_state.audio_bytes = audio_file.getvalue()

        if st.button("Convert Speech to Text"):
            try:
                recognizer = sr.Recognizer()

                # Streamlit's audio input may not be a WAV file.
                # Save the recording, then convert it to WAV with FFmpeg.
                import subprocess
                import shutil

                ffmpeg = shutil.which("ffmpeg")

                with tempfile.NamedTemporaryFile(
                    suffix=".webm", delete=False
                ) as temp:
                    temp.write(st.session_state.audio_bytes)
                    source_path = temp.name

                wav_path = source_path + ".wav"

                try:
                    if not ffmpeg:
                        st.error(
                            "FFmpeg was not found in this terminal's PATH. "
                            "Restart VS Code or add FFmpeg to PATH."
                        )
                        st.stop()

                    conversion = subprocess.run(
                        [
                            ffmpeg, "-y", "-i", source_path,
                            "-ac", "1", "-ar", "16000", wav_path
                        ],
                        capture_output=True,
                        text=True,
                        timeout=90
                    )

                    if conversion.returncode != 0:
                        st.error(
                            "Could not convert the recording to WAV. "
                            "Please record again."
                        )
                        st.stop()

                    with sr.AudioFile(wav_path) as source:
                        audio_data = recognizer.record(source)

                    with st.spinner("Converting speech to text..."):
                        transcript = recognizer.recognize_google(
                            audio_data,
                            language="en-IN"
                        )

                    st.session_state.transcript = transcript
                    st.session_state.evaluation = None
                    st.session_state.pronunciation = None

                finally:
                    for path in [source_path, wav_path]:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass

            except sr.UnknownValueError:
                st.error(
                    "Speech could not be understood. "
                    "Please record your answer again."
                )
            except sr.RequestError as e:
                st.error(
                    "Speech recognition service error. Check your "
                    f"internet connection. Details: {e}"
                )
            except Exception as e:
                st.error(f"Audio processing failed: {e}")

    if st.session_state.transcript:
        st.subheader("Recognized Answer")
        st.write(st.session_state.transcript)

        st.caption(
            "Check the transcript for speech-recognition mistakes before "
            "evaluating it. You can correct it below if necessary."
        )

        corrected_transcript = st.text_area(
            "Edit transcript if it is incorrect",
            value=st.session_state.transcript,
            key=f"transcript_edit_{st.session_state.question_index}",
            height=120
        )

        if corrected_transcript != st.session_state.transcript:
            st.session_state.transcript = corrected_transcript
            st.session_state.evaluation = None
            st.session_state.pronunciation = None

        user_answer = st.session_state.transcript


# ==================================================
# SUBMIT ANSWER
# ==================================================

if st.button("Evaluate Answer", type="primary"):
    if not user_answer.strip():
        st.warning("Please enter or record an answer first.")
    else:
        try:
            with st.spinner(
                "Qwen is evaluating your answer. Please wait..."
            ):
                result = evaluate_with_llm(
                    question=question_row["question"],
                    reference_answer=question_row["reference_answer"],
                    key_concepts=question_row["key_concepts"],
                    user_answer=user_answer,
                    mode=answer_mode
                )

            st.session_state.evaluation = result
            st.session_state.pronunciation = None

        except Exception as e:
            st.error(f"Evaluation failed: {e}")


# ==================================================
# DISPLAY EVALUATION
# ==================================================

evaluation = st.session_state.evaluation

if evaluation:
    st.divider()
    st.header("Interview Feedback")

    scores = {
        "Correctness": evaluation["correctness"],
        "Depth": evaluation["depth"],
        "Completeness": evaluation["completeness"]
    }

    cols = st.columns(3)
    for col, (name, score) in zip(cols, scores.items()):
        with col:
            st.metric(name, f"{score:g}/10")
            st.progress(int(score * 10))

    average = sum(scores.values()) / len(scores)
    st.subheader(f"Overall Content Score: {average:.1f}/10")

    st.subheader("Explanation")
    st.write(evaluation["explanation"])

    st.subheader("Grammar, Spelling and Sentence Clarity")
    st.write(evaluation["grammar_spelling"])

    st.subheader("Overall Feedback")
    st.write(evaluation["overall_feedback"])

    st.subheader("Strengths")
    for item in evaluation["strengths"]:
        st.write(f"- {item}")

    st.subheader("Areas to Improve")
    for item in evaluation["areas_to_improve"]:
        st.write(f"- {item}")

    if answer_mode == "Oral" and st.session_state.transcript:
        show_filler_report(st.session_state.transcript)

        st.subheader("Pronunciation Check")
        st.write(
            "This check analyzes the recorded audio against the recognized "
            "transcript. It is separate from the content score."
        )

        if st.button("Analyze Pronunciation"):
            if not st.session_state.audio_bytes:
                st.warning(
                    "The original recording is unavailable. "
                    "Please record your answer again."
                )
            else:
                try:
                    with st.spinner(
                        "Analyzing pronunciation. The first run may take "
                        "longer while model files download..."
                    ):
                        pronunciation_result = analyze_pronunciation(
                            st.session_state.audio_bytes,
                            st.session_state.transcript
                        )

                    st.session_state.pronunciation = pronunciation_result

                except Exception as e:
                    st.error(str(e))

        if st.session_state.pronunciation:
            show_pronunciation_report(st.session_state.pronunciation)

    with st.expander("View Reference Answer"):
        st.write(
            evaluation["reference_answer"]
            or question_row["reference_answer"]
        )


# ==================================================
# NAVIGATION
# ==================================================

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("Next Question"):
        st.session_state.question_index = (
            st.session_state.question_index + 1
        ) % len(df)

        st.session_state.evaluation = None
        st.session_state.transcript = ""
        st.session_state.audio_bytes = None
        st.session_state.audio_duration = None
        st.session_state.pronunciation = None
        st.rerun()

with col2:
    if st.button("Restart Interview"):
        st.session_state.question_index = 0
        st.session_state.evaluation = None
        st.session_state.transcript = ""
        st.session_state.audio_bytes = None
        st.session_state.audio_duration = None
        st.session_state.pronunciation = None
        st.rerun()
