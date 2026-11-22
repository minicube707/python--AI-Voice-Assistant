
from __future__ import annotations

from pathlib import Path

import logging

import torch
import numpy as np

from nlp.neural.model import Model
from nlp.neural.training import TextPreprocessor


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "intent_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = 0.7

# ============================================================
# Configuration
# ============================================================

logger = logging.getLogger("nlp.neural_classifier")

# ============================================================
# Neural Intent Classifier
# ============================================================

class NeuralClassifier:
    """Load a trained neural model and classify user messages."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        self.device = DEVICE

        self.model = None
        self.words = []
        self.classes = []

        self.preprocessor = TextPreprocessor()

        self.load()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    def load(self):
        """Load model checkpoint and initialize the neural network."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Neural model not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True
        )

        if checkpoint.get("framework") != "pytorch":
            raise ValueError("Invalid model framework.")

        input_size = checkpoint["input_size"]
        output_size = checkpoint["output_size"]

        self.words = checkpoint["words"]
        self.classes = checkpoint["classes"]

        if len(self.words) != input_size:
            raise ValueError("Vocabulary size mismatch.")

        if len(self.classes) != output_size:
            raise ValueError("Intent classes size mismatch.")

        self.model = Model(
            input_size=input_size,
            output_size=output_size
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

        logger.info("Neural model loaded successfully (%d intents).", len(self.classes))

        return self

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    def preprocess(self, text: str) -> torch.Tensor:
        """Convert input text into a neural network tensor."""

        vector = self.preprocessor.bag_of_words(
            text,
            self.words
        )

        return torch.tensor(
            vector,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    # --------------------------------------------------------
    # Predict intent
    # --------------------------------------------------------

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        """Predict the intent and return its confidence."""

        if not text or not text.strip():
            return {
                "intent": "unkown",
                "confidence": 0.0,
                "accepted": False
            }

        inputs = self.preprocess(text)
        logits = self.model(inputs)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)
        confidence = confidence.item()

        intent = self.classes[predicted_index.item()]
        accepted = confidence >= self.confidence_threshold
        
        logger.info("Intent: %s", intent)
        logger.info("Confidence: %.2f", confidence)
        logger.info("Accepted: %s", accepted)

        if accepted:
            return intent
        return "unknown"
