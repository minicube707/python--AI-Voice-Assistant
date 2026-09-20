"""
nlp/engine.py
--------------
A common contract that ALL comprehension engines must adhere to
(lightweight rules, future LLMs, future trained NLP models, etc.).

This is the cornerstone of the required modularity: the rest of the app
(commands/, player/) only knows about `Command`, never the implementation
that produced it.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Callable

from nlp.command import Command
from nlp.utils import normalize, apply_prefix_corrections
from music.music_player.extraction import extract_play_music
from music.controls.extraction import extract_music_controls
from reply.reply import extract_reply
from ai_agent.ai_agent import extract_ai_agent


class NLPEngine(ABC):
    """
    Base class for NLP engines.

    Implementations must provide `detect_intent()`.
    The engine is responsible for:
    - preprocessing input
    - detecting the intent
    - extracting the command
    """


    def __init__(self, catalog_path: str | None = None):
        self.catalog = self._load_catalog(catalog_path) if catalog_path else []

        self._intent_handlers: dict[str, Callable[[str], Command]] = {
            "play_music": self._handle_play_music,
            "music_control": self._handle_music_control,
            "exit": self._handle_exit,
            "reply": self._handle_reply,
            "ai_agent": self._handle_ai_agent,
        }


    @staticmethod
    def _load_catalog(path: str) -> list[dict]:
        if not path:
            return []

        p = Path(path)

        if not p.is_file():
            return []

        with p.open(encoding="utf-8") as file:
            return json.load(file)


    @abstractmethod
    def detect_intent(self, text: str) -> str:
        """Detect the intent from normalized text."""
        ...


    def understand(self, text: str) -> Command:
        """Convert raw user input into a Command."""

        normalized_text = normalize(text)
        corrected_text = apply_prefix_corrections(normalized_text)

        intent = self.detect_intent(corrected_text)

        command = self.process_intent(intent, corrected_text)
        command.text = corrected_text

        return command


    def process_intent(self, intent: str, text: str) -> Command:
        """Process an intent and return the corresponding command."""

        handler = self._intent_handlers.get(intent)

        if handler is None:
            return Command(
                intent="unknown",
                status="intent not supported",
                text=text,
            )

        command = handler(text)
        command.intent = intent

        return command


    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_play_music(self, text: str) -> Command:
        return extract_play_music(self.catalog, text)


    def _handle_music_control(self, text: str) -> Command:
        return extract_music_controls(text)


    def _handle_exit(self, text: str) -> Command:
        return Command(status="ok")


    def _handle_reply(self, text: str) -> Command:
        return extract_reply(text)


    def _handle_ai_agent(self, text: str) -> Command:
        return extract_ai_agent(text)
