"""
audio/wakeword.py
------------------
Wake word detection using openWakeWord 

This module runs CONTINUOUSLY while the assistant is active, unlike
the STT, which is only activated after detection.
"""

from __future__ import annotations
import time
import numpy as np
from openwakeword.model import Model

from audio.microphone import Microphone


class WakeWordDetector:

    def __init__(
        self,
        mic: Microphone,
        model_name: str = "hey_jarvis_v0.1",
        threshold: float = 0.5,
        vad_threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
        inference_framework: str = "onnx"
    ):

        self.mic = mic
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger = 0.0

        # NOTE: "hey_jarvis" is a pre-trained model included with openWakeWord. 
        # For a 100% custom wake word ("Hey DJ"), see the "Custom wake word"
        # section of the README -> training using the official notebook,
        # or the fallback listed in option B (see wakeword_fallback.py).
        self.model = Model(
            wakeword_models=[model_name],
            inference_framework=inference_framework,
            vad_threshold=vad_threshold,
        )


    def listen_for_wakeword(self) -> str | None:
        """
        Blocks until the wake word is detected. 
        Returns the name of the triggered model, or None if a stop was requested.
        """
        self.mic.start()
        try:
            while True:
                chunk = self.mic.read_chunk(timeout=1.0)
                if chunk is None:
                    continue

                predictions = self.model.predict(chunk)
                now = time.time()

                for wakeword_name, score in predictions.items():
                    if score > self.threshold and (now - self._last_trigger) > self.cooldown_seconds:
                        self._last_trigger = now
                        self.model.reset()  # prevents an immediate second trigger
                        return wakeword_name
        finally:
            self.mic.stop()
