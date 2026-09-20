"""
audio/microphone.py
--------------------
Low-level audio capture via sounddevice.
This module knows NOTHING about the wake word or STT; it simply provides
audio streams/segments. It can therefore be replaced independently.
"""

from __future__ import annotations
import queue
import numpy as np
import sounddevice as sd


class Microphone:
    """Callback-based microphone capture for streaming (low latency, low CPU usage)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
        chunk_size: int = 1280
    ):
    
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.chunk_size = chunk_size
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: sd.InputStream | None = None


    def _callback(self, indata, frames, time_info, status):
        if status:
            # Occasional xruns: log them, but never block the audio thread.
            pass
        # copy() is necessary: ​​the sounddevice buffer is reused
        self._q.put(indata[:, 0].copy())


    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_size,
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()


    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # empty the queue
        with self._q.mutex:
            self._q.queue.clear()


    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Returns the next audio block (int16, mono) or None if a timeout occurs."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


    @staticmethod
    def list_devices():
        return sd.query_devices()
