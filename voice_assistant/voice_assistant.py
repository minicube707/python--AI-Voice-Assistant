"""
voice_assistante/voice_assistante.py
------------------
Main class of voice assistant.

Pipeline :
    Microphone
        -> WakeWordDetector
        -> record_command
        -> SpeechToText
        -> NLPEngine
        -> Command
        -> handler_command
        -> TextToSpeech / AI Agent / MusicPlayer
"""

from __future__ import annotations

import logging

from audio.microphone import Microphone
from audio.wakeword import WakeWordDetector
from activation_sound import activation_sound
from speech.stt import SpeechToText, record_command
from speech.tts import TextToSpeech

from music.music_player.music_player import MusicPlayer
from handle.handler_command import handler_command

from nlp.engine import NLPEngine
from nlp.rules.rules import RulesEngine
from nlp.sentiment.sentiment import ZeroShotIntentEngine
from ai_agent.ai_agent import AIAgent


class VoiceAssistant:
    """
    Main voice assistante.

    This class orchestrates all the components:
        - Wake word
        - Microphone
        - STT
        - NLP / LLM
        - TTS
        - Music Player
        - AI Agent
    """


    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Initializing Voice Assistant...")

        # ---------------------------------------------------------
        # Composants
        # ---------------------------------------------------------

        self.mic_wake = None
        self.wake_detector = None

        self.mic_command = None
        self.stt = None
        self.tts = None

        self.nlp_engine = None
        self.player = None
        self.ai_agent = None

        self._build_audio()
        self._build_stt()
        self._build_nlp()
        self._build_music_player()
        self._build_tts()
        self._build_ai_agent()

        self.logger.info("Voice Assistant initialized.")

    # =============================================================
    # BUILDERS
    # =============================================================


    def _build_audio(self):
        """
        Initializes the microphones and the wake word detector.
        """

        audio_cfg = self.cfg["audio"]
        wake_cfg = self.cfg["wakeword"]

        self.logger.info("Loading wake word (%s)...", wake_cfg["model"])

        self.mic_wake = Microphone(
            sample_rate=audio_cfg["sample_rate"],
            channels=audio_cfg["channels"],
            device_index=audio_cfg["device_index"],
            chunk_size=wake_cfg["chunk_size"],
        )

        self.wake_detector = WakeWordDetector(
            mic=self.mic_wake,
            model_name=wake_cfg["model"],
            threshold=wake_cfg["threshold"],
            vad_threshold=wake_cfg["vad_threshold"],
            cooldown_seconds=wake_cfg["cooldown_seconds"],
            inference_framework=wake_cfg["inference_framework"],
        )

        self.mic_command = Microphone(
            sample_rate=audio_cfg["sample_rate"],
            channels=audio_cfg["channels"],
            device_index=audio_cfg["device_index"],
            chunk_size=1024,
        )


    def _build_stt(self):
        """
        Initialize Speech-to-Text.
        """

        stt_cfg = self.cfg["stt"]

        self.logger.info(
            "Loading Speech-to-Text (faster-whisper, %s)...",
            stt_cfg["model_size"]
        )

        self.stt = SpeechToText(
            model_size=stt_cfg["model_size"],
            device=stt_cfg["device"],
            compute_type=stt_cfg["compute_type"],
            language=stt_cfg["language"],
            beam_size=stt_cfg["beam_size"],
            vad_filter=stt_cfg["vad_filter"],
        )


    def _build_nlp(self):
        """
        Initialize NLP Engine.
        """

        engine_name = self.cfg["nlp"]["engine"]
        self.logger.info("Loading NLP Engine: %s", engine_name)

        if engine_name == "rules":
            self.nlp_engine = RulesEngine(
                catalog_path=self.cfg["music_library"]["catalog_path"],
            )

        elif engine_name == "zeroshot":
            self.nlp_engine = ZeroShotIntentEngine(
                catalog_path=self.cfg["music_library"]["catalog_path"],
            )

        else:
            raise ValueError(f"Unknown NLP engine: {engine_name}")

        self.logger.info("Engine selected: %s", self.nlp_engine.__class__.__name__)


    def _build_music_player(self):
        """
        Initialize player music.
        """

        self.player = MusicPlayer(
            music_dir=self.cfg["player"]["music_dir"],
        )


    def _build_tts(self):
        """
        Initialize Text-to-Speech.
        """

        tts_cfg = self.cfg["tts"]

        self.logger.info(
            "Loading TTS English (piper, %s)...",
            tts_cfg["en_model"]
        )

        self.logger.info(
            "Loading TTS French (piper, %s)...",
            tts_cfg["fr_model"]
        )

        self.tts = TextToSpeech(
            path=tts_cfg["path"],
            en_model_name=tts_cfg["en_model"],
            fr_model_name=tts_cfg["fr_model"],
        )


    def _build_ai_agent(self):
        """
        Initialize AI Agent.
        """

        agent_cfg = self.cfg["ai_agent"]

        self.ai_agent = AIAgent(
            backend=agent_cfg["backend"],
            model=agent_cfg["model"],
        )

        self.logger.info(
            "LLM loading : %s (%s)",
            agent_cfg["model"],
            agent_cfg["backend"],
        )

    # =============================================================
    # AUDIO
    # =============================================================


    def listen_for_wakeword(self):
        """
        Waits for the wake word. 

        Returns:
            str | None: The detected wake word.
        """

        return self.wake_detector.listen_for_wakeword()


    def record_command(self):
        """
        Records a user command.
        """

        audio_cfg = self.cfg["audio"]

        return record_command(
            self.mic_command,
            max_seconds=audio_cfg["command_max_seconds"],
            silence_timeout=audio_cfg["command_silence_timeout"],
        )


    def transcribe(self, audio):
        """
        Converts audio to text.
        """

        if audio.size == 0:
            return ""

        return self.stt.transcribe(
            audio,
            sample_rate=self.cfg["audio"]["sample_rate"],
        )

    # =============================================================
    # PIPELINE
    # =============================================================


    def process_audio(self, audio):
        """
        Processes a complete audio recording.

        Pipeline :

            Audio
              ↓
            STT
              ↓
            NLP
              ↓
            Handler
        """

        text = self.transcribe(audio)

        if not text:
            self.logger.info("Empty transcription.")
            return None

        self.logger.info("📝 Transcription : %s", text)

        # NLP Comprehension
        command = self.nlp_engine.understand(text)

        self.logger.info("🧩 Structured command :\n%s", command.to_json())

        # Execution
        result = handler_command(
            command,
            self.player,
            self.tts,
            self.ai_agent,
        )

        self.logger.info(
            "Result : %s",
            result
        )

        return result

    # =============================================================
    # MAIN LOOP
    # =============================================================


    def run(self):
        """
        Start the assistant's main loop.
        """

        self.logger.info("✅ Assistant ready. Waiting for the wake word...")

        try:

            while True:
                
                print("Listen...", end="\r", flush=True)
                triggered = self.listen_for_wakeword()
                
                print(" " * 20, end="\r", flush=True)

                if triggered is None:
                    continue

                self.logger.info("🔔 Wake word detected (%s)", triggered)

                resume_music = False
                if self.player._is_playing:
                    resume_music = True

                self.player.pause()
                activation_sound.play_activation_sound()

                audio = self.record_command()

                activation_sound.play_deactivation_sound()

                if resume_music:
                    self.player.resume()

                if audio.size == 0:
                    self.logger.info("No audio captured.")
                    continue

                self.process_audio(audio)

        except KeyboardInterrupt:
            self.logger.info("Stop requested by the user.")

        finally:
            self.shutdown()

    # =============================================================
    # SHUTDOWN
    # =============================================================


    def shutdown(self):
        """
        Resource cleanup.
        """

        self.logger.info("Stopping Voice Assistant...")

        self.logger.info("Voice Assistant stopped.")
