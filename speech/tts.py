"""
speech/tts.py
--------------
Bilingual TTS: French voice for sentences, English voice for titles/artists,
leveraging pre-structured segments (Command.title, Command.artist)
rather than language detection in free-form text.
"""

import numpy as np
import sounddevice as sd
from piper import PiperVoice
from pathlib import Path

START_SENTENCE = [
    "OK",
    "D'accord",
    "Compris",
    "Bien sur",
    "Très bien"
]

VERBE_SENTENCE = [
    "je mets",
    "je lance",
    "je joue"
]

BASE_DIR = Path(__file__).parent

class TextToSpeech:
    def __init__(
        self,
        path: str,
        fr_model_name: str,
        en_model_name: str
    ):

        fr_model_full_path = BASE_DIR / path / fr_model_name
        en_model_full_path = BASE_DIR / path / en_model_name

        self.voice_fr = PiperVoice.load(fr_model_full_path)
        self.voice_en = PiperVoice.load(en_model_full_path, )


    def _synth_segment(self, text: str, voice: PiperVoice) -> np.ndarray:
        chunks = [
            np.frombuffer(c.audio_int16_bytes, dtype=np.int16)
            for c in voice.synthesize(text)
        ]
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)


    def _play(self, audio: np.ndarray, samplerate: int):
        sd.play(audio, samplerate=samplerate)
        sd.wait()


    def speak_french(self, text: str):
        """100% French TTS for sentences without proper nouns (errors, questions)."""
        audio = self._synth_segment(text, self.voice_fr)
        self._play(audio, self.voice_fr.config.sample_rate)


    def speak_english(self, text: str):
        """100% English TTS for sentences without proper nouns (errors, questions)."""
        audio = self._synth_segment(text, self.voice_en)
        self._play(audio, self.voice_en.config.sample_rate)


    def speak_mixed(self, segments: list[tuple[str, str]]):
        """
        segments: a list of (text, language) pairs, where language is "fr" or "en". 
        Concatenates the segments with a brief silence between them to
        avoid an overly abrupt transition between the two voices.
        """
        silence_gap = np.zeros(int(0.08 * self.voice_fr.config.sample_rate), dtype=np.int16)
        pieces = []

        text_speech = ""
        for text, lang in segments:
            if not text.strip():
                continue

            voice = self.voice_en if lang == "en" else self.voice_fr
            pieces.append(self._synth_segment(text, voice))
            pieces.append(silence_gap)
            text_speech += text + ' '

        if not pieces:
            return

        full_audio = np.concatenate(pieces)
        # Both voices likely have the same sample rate (22050 Hz for medium quality),
        # but we use the French voice's rate by default for playback.
        self._play(full_audio, self.voice_fr.config.sample_rate)


    def announce_play_music(self, title: str | None, artist: str | None):
        """Builds the ad based on an already structured Command."""
        segments: list[tuple[str, str]] = []
        
        start_sentence = np.random.choice(START_SENTENCE)
        verbe_sentence = np.random.choice(VERBE_SENTENCE)

        sentence = []
        segments.append((start_sentence, "fr"))
        segments.append((verbe_sentence, "fr"))

        if title and artist:
            segments.extend([(title, "en"), ("de", "fr"), (artist, "en")])
        elif title:
            segments.append((title, "en"))
        elif artist:
            segments.append((artist, "en"))
        else:
            segments.append(("de la musique", "fr"))

        self.speak_mixed(segments)
