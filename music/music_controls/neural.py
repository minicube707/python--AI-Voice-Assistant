from __future__ import annotations

from pathlib import Path

from neural.classifier import NeuralClassifier
from nlp.command import Command
from music.music_controls.volume import extract_volume

BASE_DIR = Path(__file__).resolve().parent

INTENTS_PATH = BASE_DIR / "intents.json"
MODEL_PATH = BASE_DIR / "models" / "music_controls.pth"

# Lower than for intents: the sentences are short ("stop", "pause")
CONFIDENCE_THRESHOLD = 0.6


class NeuralMusicControls:
    """Extract the music controls using a neural network.

    The network predicts the action; the volume value, which a
    bag-of-words model cannot extract, is read with a regular expression.

    Raises:
        FileNotFoundError: at creation, if the model has not been trained yet
            (``python main.py --train controls``).
    """

    def __init__(self):
        self.classifier = NeuralClassifier(
            MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            name="music_controls",
        )

    def __call__(self, sentence: str) -> Command:
        prediction = self.classifier.predict(sentence)

        if not prediction.accepted:
            return Command(status="error")

        action = prediction.label

        if action == "volume":
            return Command(
                argument={"action": "volume", "volume": extract_volume(sentence)}
            )

        return Command(argument={"action": action})
