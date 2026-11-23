"""
music/music_controls/training.py
--------------------------------
Training of the music-control action classifier
(pause, stop, continue, increase_volume, decrease_volume, volume).
"""

from __future__ import annotations

from neural.trainer import NeuralTrainer, load_intents
from music.music_controls.neural import INTENTS_PATH, MODEL_PATH


def train_music_controls() -> dict:
    print("Training the music controls classifier...\n")

    return NeuralTrainer(load_intents(INTENTS_PATH), MODEL_PATH, epochs=300).run()
