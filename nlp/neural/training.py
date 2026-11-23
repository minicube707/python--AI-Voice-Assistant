"""
nlp/neural/training.py
----------------------
Training of the intent classifier (play_music, music_control, exit, ...).
The generic machinery lives in the top-level `neural` package.
"""

from __future__ import annotations

from pathlib import Path

from neural.trainer import NeuralTrainer, load_intents
from reply.reply import REPLY

BASE_DIR = Path(__file__).resolve().parent

INTENTS_PATH = BASE_DIR / "intents.json"
MODEL_PATH = BASE_DIR / "models" / "intent_classifier.pth"


def train_intent_classifier() -> dict:
    intents = load_intents(INTENTS_PATH)

    # The "reply" intent is generated from reply/reply.json
    intents["reply"] = [item["quote"] for item in REPLY()]

    print("Training the intent classifier...\n")

    return NeuralTrainer(intents, MODEL_PATH).run()
