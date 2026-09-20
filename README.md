# voice-assistant — Lightweight Local Voice Assistant

A 100% local voice assistant, with no LLM required, designed to run
comfortably on a modest CPU.

## Features

- Local wake word detection (openWakeWord)
- Speech-to-Text (faster-whisper) and Text-to-Speech (Piper)
- Control of your local music library (play, pause, etc.)
- Command understanding via rules, or an optional Zero-Shot intent engine
- Optional AI agent (Ollama, Gemini, OpenAI, Anthropic)

## Requirements

- Windows 11
- [Python](https://www.python.org/downloads/) and [uv](https://docs.astral.sh/uv/)
- [VLC](https://www.videolan.org/vlc/) (audio playback)
- [FFmpeg](https://ffmpeg.org/download.html) (audio processing)

```powershell
winget install VideoLAN.VLC
winget install Gyan.FFmpeg
```

## Installation

```powershell
uv sync
```

## Usage

### 1. Index your music

Run this once, then again whenever you add new songs. The program scans your
music folder and generates a JSON file in `data/`.

```powershell
python main.py --parse
```

### 2. Start the assistant

```powershell
python main.py
```

Wait until 'Listen...' is displayed, then say **"Hey Jarvis"** (the pre-trained wake word shipped by default),
followed by your command. For example:

> "Hey Jarvis, play Billie Jean by Michael Jackson"

### 3. Quit the assistant

Say **"Hey Jarvis"** followed by **"goodbye"** or **"bye"** to exit the program.

> "Hey Jarvis, goodbye"

The assistant replies with a farewell message and shuts down cleanly.

You can also stop it at any time from the terminal with `Ctrl+C`.

The list of exit phrases is handled by the `exit_command/` component.

## Architecture

Each folder is a component that can be replaced independently
(see the docstring at the top of each file).

| Folder | Role |
|---|---|
| `activation_sound/` | Plays a melody when the wake word is triggered |
| `ai_agent/` | AI agent that answers your questions |
| `audio/` | Microphone capture + wake word detection (openWakeWord) |
| `config/` | Configuration files to customize the assistant's behavior |
| `data/` | JSON file containing all your songs |
| `exit_command/` | Command to quit the program |
| `handle/` | Executes the recognized command |
| `log/` | Log files |
| `music/` | Functions to play and control music |
| `nlp/` | Understanding: `rules.py` (default) or `sentiment.py` (optional) |
| `parse/` | Scans your music folder to generate the file stored in `data/` |
| `reply/` | Predefined sentences the assistant uses to reply |
| `speech/` | Speech-to-Text (faster-whisper) and Text-to-Speech (Piper) |
| `voice_assistant` | Entry point that assembles the voice assistant |

The shared contract is the `Command` class (`nlp/command.py`): any
understanding engine (rules, a future LLM, a future trained model) must
produce this same object.

## Configuration

Behavior is configured in `config/config.yaml`.

### NLP engine

The `zeroshot` engine is optional and recommended for more powerful machines.

```yaml
nlp:
  engine: "rules"      # "rules" (default) or "zeroshot"
```

### AI agent

After the wake word, say **"demande a Ollama"** followed by your question.  
For example:

> "Hey Jarvis, demande a ollama qelle est la météo à Paris aujourd'hui"

Available backends: `ollama`, `gemini`, `openai`, `anthropic`.

```yaml
ai_agent:
  backend: "ollama"
  model: "qwen3:1.7b"
```

## Environment variables (`.env`)

API keys for cloud backends are read from a `.env` file at the project root.
**Never commit this file**: it contains secrets.

1. Copy the example file:

```powershell
   copy .env.example .env
```

2. Fill in only the keys for the backends you use:

```env
   # Only needed if ai_agent.backend = "gemini"
   GEMINI_API_KEY=your_gemini_key

   # Only needed if ai_agent.backend = "openai"
   OPENAI_API_KEY=your_openai_key

   # Only needed if ai_agent.backend = "anthropic"
   ANTHROPIC_API_KEY=your_anthropic_key

   # Optional: only if Ollama is not running on the default address
   OLLAMA_HOST=http://localhost:11434
```

| Backend | Variable | Required |
|---|---|---|
| `ollama` | `OLLAMA_HOST` | No (defaults to `http://localhost:11434`) |
| `gemini` | `GEMINI_API_KEY` | Yes |
| `openai` | `OPENAI_API_KEY` | Yes |
| `anthropic` | `ANTHROPIC_API_KEY` | Yes |

Where to get a key:
[Google AI Studio](https://aistudio.google.com/apikey) ·
[OpenAI](https://platform.openai.com/api-keys) ·
[Anthropic Console](https://console.anthropic.com/)
