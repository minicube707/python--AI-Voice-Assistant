"""
neural/trainer.py
-----------------
Generic trainer: takes a {tag: patterns} dataset, trains a bag-of-words
classifier and saves a checkpoint.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neural.checkpoint import save_checkpoint
from neural.model import DEFAULT_DROPOUT, DEFAULT_HIDDEN_SIZES, Model
from neural.preprocessing import TextPreprocessor

logger = logging.getLogger("neural.trainer")

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_intents(path: Path) -> dict[str, list[str]]:
    """Read an intents JSON file and return {tag: patterns}."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Intents file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return validate_intents(data["intents"])


def validate_intents(intents: list[dict]) -> dict[str, list[str]]:
    """Validate a list of {"tag", "patterns"} and convert it to a dict."""

    if not isinstance(intents, list) or not intents:
        raise ValueError("Invalid or empty intents configuration.")

    result: dict[str, list[str]] = {}

    for intent in intents:
        tag = intent.get("tag")
        patterns = intent.get("patterns")

        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("Each intent requires a valid tag.")

        if tag in result:
            raise ValueError(f"Duplicate intent tag: {tag}")

        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"Intent '{tag}' requires at least one pattern.")

        if not all(isinstance(p, str) for p in patterns):
            raise ValueError("Intent patterns must be strings.")

        result[tag] = patterns

    return result


class NeuralTrainer:
    """Train a classifier from {tag: patterns} and save it to `model_path`."""

    def __init__(
        self,
        intents: dict[str, list[str]],
        model_path: Path,
        epochs: int = 200,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        dropout: float = DEFAULT_DROPOUT,
    ):
        self.intents = intents
        self.model_path = Path(model_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        self.device = DEVICE
        self.preprocessor = TextPreprocessor()

        self.words: list[str] = []
        self.classes: list[str] = sorted(intents)
        self.model: Model | None = None

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        class_to_index = {tag: i for i, tag in enumerate(self.classes)}
        documents: list[tuple[list[str], int]] = []
        vocabulary: set[str] = set()

        for tag, patterns in self.intents.items():
            for pattern in patterns:
                tokens = self.preprocessor.tokenize(pattern)

                if not tokens:
                    continue

                vocabulary.update(tokens)
                documents.append((tokens, class_to_index[tag]))

        self.words = sorted(vocabulary)

        if not documents or not self.words:
            raise ValueError("Insufficient training data.")

        random.shuffle(documents)

        train_x = np.asarray(
            [[float(w in token_set) for w in self.words]
             for token_set in (set(t) for t, _ in documents)],
            dtype=np.float32,
        )
        train_y = np.asarray([label for _, label in documents], dtype=np.int64)

        logger.info(
            "Vocabulary: %d | Classes: %d | Samples: %d",
            len(self.words), len(self.classes), len(train_x),
        )

        return train_x, train_y

    def train(self, train_x: np.ndarray, train_y: np.ndarray) -> dict:
        self.model = Model(
            input_size=train_x.shape[1],
            output_size=len(self.classes),
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        history = {"loss": [], "accuracy": []}
        self.model.train()

        for epoch in range(self.epochs):
            running_loss, correct, total = 0.0, 0, 0

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_y.size(0)
                correct += (outputs.argmax(dim=1) == batch_y).sum().item()
                total += batch_y.size(0)

            history["loss"].append(running_loss / total)
            history["accuracy"].append(correct / total)

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Loss: {history['loss'][-1]:.4f} | "
                    f"Accuracy: {history['accuracy'][-1] * 100:.2f}%"
                )

        return history

    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("No trained model available.")

        save_checkpoint(
            self.model_path,
            self.model,
            self.words,
            self.classes,
            self.hidden_sizes,
            self.dropout,
        )

        print(f"Model saved: {self.model_path}")

    def run(self) -> dict:
        """Complete pipeline: prepare, train, save."""

        set_seed()
        train_x, train_y = self.prepare_data()
        history = self.train(train_x, train_y)
        self.save()

        return history
