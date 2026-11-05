# AI Mock Interview System

An interactive Streamlit application for practicing technical and HR
interviews. Users can answer questions in writing or by voice and receive
AI-generated feedback on correctness, depth, completeness, grammar, and
overall quality.

## Features

- Technical, HR, or mixed interview practice.
- Subject filtering for DSA, DBMS, operating systems, computer networks,
  OOP, and general HR questions.
- Easy, medium, and hard difficulty levels.
- Written answers through a text area.
- Oral answers through microphone recording.
- Speech-to-text conversion using Google Speech Recognition.
- Local answer evaluation through Ollama and a configurable Qwen model.
- Separate filler-word analysis for oral answers.
- Optional audio-based pronunciation analysis with OpenPronounce.
- Session-state handling so the current question, transcript, audio, and
  evaluation survive Streamlit reruns.

## Project architecture

```text
AI_Mock_Interview/
├── README.md
├── .gitignore
└── AI_Mock_Interview/
    ├── app.py
    └── questions.csv
```

### Runtime components

```text
Browser
  │
  ▼
Streamlit UI (app.py)
  ├── Loads and validates questions.csv
  ├── Filters questions by interview type, subject, and difficulty
  ├── Collects written text or microphone audio
  ├── Converts oral audio to text with SpeechRecognition
  ├── Sends the answer and question context to Ollama
  ├── Calculates deterministic filler-word statistics
  └── Optionally compares audio pronunciation with OpenPronounce
```

### Request and evaluation flow

1. `questions.csv` is loaded with `pandas` and cached with
   `st.cache_data`.
2. Sidebar selections filter the question set.
3. Streamlit session state tracks the question index, answer, transcript,
   recorded audio, settings, and evaluation.
4. For a written answer, the text is sent directly to the evaluator.
5. For an oral answer:
   - Streamlit captures the recording.
   - FFmpeg converts the browser recording to 16 kHz mono WAV.
   - Google Speech Recognition creates a transcript.
   - The transcript can be corrected before evaluation.
6. `evaluate_with_llm()` sends the question, reference answer, key concepts,
   and user answer to the local Ollama HTTP API.
7. The model must return JSON containing three scores from 0 to 10,
   explanations, strengths, improvement areas, and a reference answer.
8. Oral feedback additionally includes filler-word counts and, when
   configured, pronunciation results.

### Data model

Each row in `questions.csv` represents one question:

| Column | Description |
|---|---|
| `id` | Unique question identifier |
| `category` | `Technical` or `HR` |
| `subject` | Topic such as `DSA` or `General` |
| `difficulty` | `Easy`, `Medium`, or `Hard` |
| `question` | Prompt shown to the user |
| `reference_answer` | Baseline answer supplied to the evaluator |
| `key_concepts` | Semicolon-separated concepts expected in a good answer |

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Ollama
- A downloaded Ollama model, preferably `qwen3.5:4b`
- FFmpeg for oral answers and pronunciation analysis
- A microphone for oral mode
- Internet access for Google Speech Recognition

Written mode only requires Python packages and Ollama. FFmpeg,
SpeechRecognition, and OpenPronounce are needed for the corresponding audio
features.

## Installation

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit pandas SpeechRecognition
```

Install Ollama from [ollama.com](https://ollama.com), then download the
default model:

```powershell
ollama pull qwen3.5:4b
```

Make sure FFmpeg is installed and available on `PATH`. Verify both tools:

```powershell
ollama list
ffmpeg -version
```

To enable pronunciation analysis, install a compatible OpenPronounce release
in the same virtual environment:

```powershell
pip install openpronounce
```

If the package has additional platform-specific requirements, follow its
installation instructions before using the pronunciation button.

## Running the application

Start Ollama if it is not already running, activate the virtual environment,
and launch Streamlit from the repository root:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run .\AI_Mock_Interview\app.py
```

Open the URL printed by Streamlit, normally
`http://localhost:8501`.

The application reads `questions.csv` using a relative path, so launch it
from the repository root as shown above.

### Selecting another Ollama model

The default model is `qwen3.5:4b`. The app reads `OLLAMA_MODEL` from the
environment:

```powershell
$env:OLLAMA_MODEL = "qwen3.5:4b"
streamlit run .\AI_Mock_Interview\app.py
```

The selected model must be available locally:

```powershell
ollama pull <model-name>
```

## How to use

1. Start Ollama and Streamlit.
2. Choose an interview type in the sidebar.
3. Choose a subject and difficulty.
4. Select `Written` or `Oral` answer mode.
5. Read the displayed question and submit an answer.
6. Select **Evaluate Answer**.
7. Review the content scores, explanation, strengths, and improvement areas.
8. In oral mode, review the transcript and correct it if necessary before
   evaluation.
9. After evaluation, review filler-word statistics.
10. Select **Analyze Pronunciation** if OpenPronounce and FFmpeg are
    configured.
11. Change the sidebar settings to restart with a new filtered question set.

## Configuration and extension points

- **Question bank:** Add or edit rows in
  [`AI_Mock_Interview/questions.csv`](./AI_Mock_Interview/questions.csv).
  Keep all required columns and use the existing category, subject, and
  difficulty values unless the filtering logic is updated.
- **Model:** Change `OLLAMA_MODEL` without changing source code.
- **Evaluator:** Update the prompt and validation in
  `evaluate_with_llm()` to change scoring criteria or feedback fields.
- **Speech recognition:** Change the language passed to
  `recognize_google()` when targeting another locale.
- **Filler analysis:** Extend the regular-expression map in
  `analyze_fillers()`.
- **Pronunciation:** Adapt `analyze_pronunciation()` if the installed
  OpenPronounce API or supported audio formats differ.

## Troubleshooting

### Ollama connection or evaluation errors

Confirm that Ollama is running and that the configured model is installed:

```powershell
ollama list
ollama run qwen3.5:4b
```

The evaluator uses `http://localhost:11434/api/chat` and allows up to five
minutes for a response.

### FFmpeg not found

Install FFmpeg, add its `bin` directory to the system `PATH`, restart VS
Code or the terminal, and verify with `ffmpeg -version`.

### Speech cannot be recognized

Check microphone permissions, record clear English audio, and confirm that
the machine has internet access. Speech recognition is an external service,
so recognition can fail or produce an imperfect transcript.

### No questions found

Check that the selected category, subject, and difficulty values match the
rows in `questions.csv`. The application stops with a warning when a filter
combination has no matching rows.

## Limitations

- Evaluations depend on the local model's quality and response time.
- Speech recognition requires an internet connection and may misrecognize
  words.
- Filler-word counts are based on the transcript, not the raw audio.
- Pronunciation scores are automated estimates and should not be treated as
  definitive accent or fluency judgments.
- There is no database or user-account system; application state is kept in
  the current Streamlit session.
