"""
nlp/neural/neural.py
--------------------
NLP engine based on a neural network that detects the intent.
(The music controls extractor is chosen separately in the config.)
"""

from __future__ import annotations

from nlp.engine import NLPEngine
from neural.classifier import NeuralClassifier
from nlp.neural.training import MODEL_PATH


class NeuralEngine(NLPEngine):

    def __init__(
        self,
        catalog_path: str | None = None,
        music_controls_intent: str = "rules",
    ):
        super().__init__(
            catalog_path=catalog_path,
            music_controls_intent=music_controls_intent,
        )
        self.classifier = NeuralClassifier(MODEL_PATH, name="intent")

    def detect_intent(self, text: str) -> str:
        return self.classifier.predict(text).label
