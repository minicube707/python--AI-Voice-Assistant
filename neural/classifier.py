"""
neural/classifier.py
--------------------
Generic text classifier: loads a checkpoint and predicts a label.
Used for both intent detection and music-control action detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from neural.checkpoint import load_checkpoint
from neural.preprocessing import TextPreprocessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    accepted: bool


class NeuralClassifier:
    """Load a trained model once and classify texts."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        name: str = "neural",
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = DEVICE
        self.logger = logging.getLogger(f"neural.{name}")

        self.preprocessor = TextPreprocessor()

        checkpoint = load_checkpoint(self.model_path, self.device)
        self.model = checkpoint.model
        self.words = checkpoint.words
        self.classes = checkpoint.classes

        self.logger.info("Model loaded (%d classes).", len(self.classes))

    @torch.inference_mode()
    def predict(self, text: str) -> Prediction:
        if not text or not text.strip():
            return Prediction(UNKNOWN, 0.0, False)

        vector = self.preprocessor.bag_of_words(text, self.words)
        inputs = torch.from_numpy(vector).unsqueeze(0).to(self.device)

        probabilities = torch.softmax(self.model(inputs), dim=1)
        confidence, index = torch.max(probabilities, dim=1)

        confidence = confidence.item()
        label = self.classes[index.item()]
        accepted = confidence >= self.confidence_threshold

        self.logger.info(
            "label=%s confidence=%.2f accepted=%s", label, confidence, accepted
        )

        return Prediction(label if accepted else UNKNOWN, confidence, accepted)
