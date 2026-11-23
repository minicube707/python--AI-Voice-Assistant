# AI-Voice-Assistant — Lightweight Local Voice Assistant

A 100% local voice assistant, with no LLM required, designed to run
comfortably on a modest CPU.

## Features

- Local wake word detection (openWakeWord)
- Speech-to-Text (faster-whisper) and Text-to-Speech (Piper)
- Control of your local music library (play, pause, etc.)
- Command understanding via rules, or an optional neural / Zero-Shot / Laya engine
- Music controls (pause, volume...) extracted by rules or by a neural network, selectable in the config
- Optional AI agent (Ollama, Gemini, OpenAI, Anthropic)

## Requirements

- Windows 11
- [Python](https://www.python.org/downloads/) and [uv](https://docs.astral.sh/uv/)
- [VLC](https://www.videolan.org/vlc/) (audio playback)
- [FFmpeg](https://ffmpeg.org/download.html) (audio processing)
- **Optional**: [Ollama](https://ollama.com/download/windows) — to run a local AI agent that answers your questions and uses tools (see [Optional: Ollama](#optional-ollama)).
- **Optional**: [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) — to create an isolated sandbox environment where the model can write, execute, and test code.

## Installation

Install VLC and FFmpeg:
```powershell
winget install VideoLAN.VLC
winget install Gyan.FFmpeg
```

Install the project dependencies:
```powershell
uv sync
```
### Optional: Ollama
Ollama is only required if you want to use the AI agent with a local model (`ai_agent.backend: "ollama"`, the default). Without it, the assistant still plays and controls your music. If you prefer a cloud backend (Gemini, OpenAI, Anthropic), see [AI agent](#ai-agent) and [Environment variables](#environment-variables-env) instead.

1. Install Ollama:
```powershell
winget install Ollama.Ollama
```

2. Download the model set in `config/config.yaml` (`qwen3:1.7b` by default). Pick a model that supports **tool calling**, since the agent uses tools (web search, Wikipedia, weather, Python sandbox):
```powershell
ollama pull qwen3:1.7b
```

3. Check that Ollama is installed and the model is available:
```powershell
ollama --version
ollama list
```

4. Make sure the model name in the configuration matches the one you pulled:
```yaml
ai_agent:
  backend: "ollama"
  model: "qwen3:1.7b"
```

You do not need to start Ollama yourself: when you ask the agent something, the assistant checks that the Ollama server is running (`http://localhost:11434`) and starts `ollama serve` if needed.

Then say "Hey Jarvis, demande à Ollama ..." followed by your question (see [Available Commands](#available-commands)).

**Note**: a small model runs on a modest CPU but answers more slowly and less accurately than a larger one. Running model-generated Python code additionally requires [Docker Desktop](#optional-docker-desktop).

### Optional: Docker Desktop
Docker Desktop is only required if you want to run the model's code in an isolated sandbox.

1. Install Docker Desktop for Windows.
2. Enable the WSL 2 backend if required.
3. Start Docker Desktop and verify that Docker is running:

```powershell
docker --version
docker info
```

**Note**: Docker Desktop is not required to install or run the project unless the sandbox functionality is enabled. For security, execute model-generated code in restricted containers with appropriate resource limits and limited access to the host system.

## Usage

### 1. Index your music

Run this once, then again whenever you add new songs. The program scans your
music folder and generates a JSON file in `data/`.

```powershell
python main.py --parse
```

Run this with -f whenever you want to force a full re-parse of your music folder instead of updating the existing data. This regenerates the JSON file in data/ from scratch.

```powershell
python main.py --parse -f
```

### 2. Train the neural models (neural engine / neural extraction only)

The `neural` NLP engine and the `neural` music controls extractor rely on models
that are **not available until you train them**. If you select one of them in
`config/config.yaml` (see [Configuration](#configuration)) without training it first,
the assistant stops at startup with a `Neural model not found` error.

```powershell
python main.py --train             # trains every model
python main.py --train intents     # only the intent classifier (nlp/neural)
python main.py --train controls    # only the music controls classifier (music/music_controls)
```

Training takes a few seconds on CPU. Run it again whenever you edit a dataset
(`nlp/neural/intents.json`, `music/music_controls/intents.json`, `reply/reply.json`).
The default `rules` engine and `rules` extractor need no training.

### 3. Start the assistant

```powershell
python main.py
```

Wait until Listen... is displayed in the terminal. Then say **"Hey Jarvis"**, the default pre-trained wake word, followed by your command.

#### Available Commands
🎵 **Play Music**  
Ask Jarvis to play a specific song by providing its title and artist.

**Example:**

"Hey Jarvis, joue Billie Jean de Michael Jackson."

💿 **Play an Album**  
Ask Jarvis to play an entire album by specifying its title and artist.

**Example**:

"Hey Jarvis, joue l'album Discovery de Daft Punk."

📚 **Look Up Definitions on Wikipedia**  
Ask Jarvis to retrieve information from Wikipedia and use Ollama to help explain or summarize it. You can specify the language of the Wikipedia source.

**Example**:

"Hey Jarvis, demande à Ollama donne moi la définition d'une république en utilisant Wikipédia en français."

📰 **Search for the Latest News**  
Ask Jarvis to search the internet for recent news and use Ollama to summarize the results.

**Example**:

"Hey Jarvis, demande à Ollama recherche sur Internet les dernières actualités en France en 2026 et résumer les principaux événements."

🧮 **Solve Mathematical Problems with Python**  
Ask Jarvis to generate and execute Python code to perform calculations or solve mathematical problems.

**Example**:

"Hey Jarvis, demande à Ollama écrie et exécute un script Python pour calculer les chiffres de pi."

💬 **Ask Questions**  
Ask Jarvis general questions and let Ollama generate an answer.

**Example**:

"Hey Jarvis, demande à Ollama explique moi comment fonctionnent les trous noirs."  

#### Notes
Always start your command with "Hey Jarvis" to activate the assistant.
Speak clearly and wait for the assistant to process your request.
Internet access is required for online searches, such as retrieving news or Wikipedia information.
Ollama must be installed and properly configured for AI-powered responses.
Docker Desktop is optional and can be used to provide an isolated sandbox for executing and testing model-generated Python code.

### 4. Quit the assistant

Say **"Hey Jarvis"** followed by **"au revoir"**, **"bye"** or **ciao**  to exit the program.

> "Hey Jarvis, au revoir"

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
| `music/` | Functions to play and control music (`music_controls/` has a `rules.py` and a `neural.py` extractor) |
| `neural/` | Shared neural network toolkit (preprocessing, model, trainer, classifier) used by the neural engine and the neural music controls |
| `nlp/` | Understanding: `rules/` (default), `neural/`, `sentiment/` or `laya/` (optional)|
| `parse/` | Scans your music folder to generate the file stored in `data/` |
| `reply/` | Predefined sentences the assistant uses to reply |
| `speech/` | Speech-to-Text (faster-whisper) and Text-to-Speech (Piper) |
| `voice_assistant` | Entry point that assembles the voice assistant |

The shared contract is the `Command` class (`nlp/command.py`): any
understanding engine (rules, a future LLM, a future trained model) must
produce this same object.

## Configuration

Behavior is configured in `config/config.yaml`.

### Music

Before starting the application, make sure to update the `music_dir` value in the configuration file with the path to your music folder.

```
player:
  music_dir: "The path for your music folder"
```

Replace `The path for your music folder` with the actual path to your music directory.

For example:

```
player:
  music_dir: "/home/user/Music"
```

### Music folder structure

Your music files must follow one of the following directory structures:

**Option 1: Artist / Music file**

```
music_dir/
└── Artist/
    └── music.extension
```

**Option 2: Artist / Album / Music file**

```
music_dir/
└── Artist/
    └── Album/
        └── music.extension
```

Replace `music_dir` with your music folder and `extension` with the appropriate audio file extension (e.g., `.mp3`, `.flac`, `.wav`).

Make sure your music files are organized according to one of these structures so the application can correctly scan and parse your music library.

### NLP engine

Default:  
```yaml
nlp:
  engine: "rules"
```

The `zeroshot` engine is optional and recommended for more powerful machines.

```yaml
nlp:
  engine: "zeroshot"
```

The `laya` engine is optional and recommended for more powerful machines.

```yaml
nlp:
  engine: "laya"
```

The `neural` engine detects the intent with a small neural network.
**Train it first** with `python main.py --train intents` (see [Usage](#2-train-the-neural-models-neural-engine--neural-extraction-only)).

```yaml
nlp:
  engine: "neural"
```

### Music controls extraction

Whatever the NLP engine, you choose how the music controls
(pause, stop, continue, volume up / down, set volume) are extracted from a sentence
with `music_controls_intent`:

| Value | Description |
|---|---|
| `rules` (default) | Keyword matching. No training required |
| `neural` | A neural network predicts the action, so it understands more phrasings ("plus fort", "monte un peu le son"...). **Train it first** with `python main.py --train controls` |

```yaml
nlp:
  engine: "rules"
  music_controls_intent: "neural"
```

In both cases, the volume value ("mets le volume à 50 pour cent") is read with a regular expression.

### AI agent

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
```

| Backend | Variable | Required |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` | Yes |
| `openai` | `OPENAI_API_KEY` | Yes |
| `anthropic` | `ANTHROPIC_API_KEY` | Yes |

Where to get a key:
[Google AI Studio](https://aistudio.google.com/apikey) ·
[OpenAI](https://platform.openai.com/api-keys) ·
[Anthropic Console](https://console.anthropic.com/)
