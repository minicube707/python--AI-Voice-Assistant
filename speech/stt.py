"""
speech/stt.py
--------------
Local speech-to-text via faster-whisper (CTranslate2, int8 quantization).

Deliberately minimal interface: `record_command()` -> `transcribe()`.
This module is fully swappable (Vosk, whisper.cpp, etc.) provided that
`transcribe(audio: np.ndarray) -> str` is adhered to.
"""

from __future__ import annotations
import time
import numpy as np
from faster_whisper import WhisperModel

from audio.microphone import Microphone


class SpeechToText:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "fr",
        beam_size: int = 1,
        vad_filter: bool = True
    ):

        # The model is loaded only once at startup (not for every command),
        # otherwise the latency after the wake word would be unacceptable.
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter


    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> str:
        # faster-whisper attend du float32 normalisé [-1, 1]
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        segments, info = self.model.transcribe(
            audio_f32,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,  # commandes courtes indépendantes
            task="transcribe", #conserve les mots en anglais 
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()


def record_command(mic: Microphone, max_seconds: float = 6.0,
                    silence_timeout: float = 1.2,
                    silence_rms_threshold: int = 300,
                    ) -> np.ndarray:
    """
    Records the voice command following the wake word. 
    Automatically stops after a prolonged silence (avoids waiting 6 seconds every time). 
    Simple silence detection using RMS (no additional dependencies, negligible CPU usage).
    """

    mic.start()
    frames = [] 
    start_time = time.time()
    last_voice_time = start_time

    try:
        while True:
            chunk = mic.read_chunk(timeout=0.5)
            if chunk is None:
                if time.time() - start_time > max_seconds:
                    break
                continue

            frames.append(chunk)
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            now = time.time()

            if rms > silence_rms_threshold:
                last_voice_time = now

            if now - last_voice_time > silence_timeout and now - start_time > 0.5:
                break
            if now - start_time > max_seconds:
                break
    finally:
        mic.stop()

    if not frames:
        return np.array([], dtype=np.int16)
    return np.concatenate(frames)
