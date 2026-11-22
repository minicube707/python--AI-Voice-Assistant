
from __future__ import annotations

import json
import random

from pathlib import Path

import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from nltk.stem.snowball import SnowballStemmer

from nlp.utils import normalize
from nlp.neural.model import Model
from reply.reply import REPLY

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INTENTS_PATH = BASE_DIR / "intents.json"


MODEL_PATH = BASE_DIR / "models" / "intent_classifier.pth"

RANDOM_SEED = 42

EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.001

IGNORED_CHARACTERS = {"?", "!", ".", ","}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# NLP Preprocessing
# ============================================================

class TextPreprocessor:

    def __init__(self):

        self.stemmer = SnowballStemmer("french")

        self._download_resources()

    @staticmethod
    def _download_resources():

        resources = {
            "corpora/wordnet": "wordnet",
            "tokenizers/punkt": "punkt",
            "tokenizers/punkt_tab": "punkt_tab"
        }

        for resource, package in resources.items():

            try:
                nltk.data.find(resource)

            except LookupError:
                print(f"Downloading NLTK resource: {package}")
                nltk.download(package, quiet=True)

    def tokenize(self, text: str) -> list[str]:

        tokens = nltk.word_tokenize(text, language="french")

        return [
            self.stemmer.stem(normalize(token))
            for token in tokens
            if token not in IGNORED_CHARACTERS
        ]

    def bag_of_words(
        self,
        text: str,
        vocabulary: list[str]
    ) -> np.ndarray:

        tokens = set(self.tokenize(text))

        return np.asarray(
            [float(word in tokens) for word in vocabulary],
            dtype=np.float32
        )


# ============================================================
# Neural NLP Trainer
# ============================================================

class NeuralTrainer:

    def __init__(self):

        set_seed()

        self.preprocessor = TextPreprocessor()

        self.words: list[str] = []
        self.classes: list[str] = []

        self.model: Model | None = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.device = DEVICE

        print(f"Using device: {self.device}")

    # --------------------------------------------------------
    # Load intents
    # --------------------------------------------------------

    def load_intents(self) -> dict:

        if not INTENTS_PATH.exists():
            raise FileNotFoundError(
                f"Intents file not found: {INTENTS_PATH}"
            )

        with INTENTS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        
        list_quote = []
        for item in REPLY():
            list_quote.append(item["quote"])

        data['intents'].append({
            "tag": "reply",
            "patterns": list_quote
        })

        intents = data.get("intents")

        if not isinstance(intents, list) or not intents:
            raise ValueError("Invalid or empty intents configuration.")

        seen_tags = set()

        for intent in intents:

            tag = intent.get("tag")
            patterns = intent.get("patterns")

            if not isinstance(tag, str) or not tag.strip():
                raise ValueError("Each intent requires a valid tag.")

            if tag in seen_tags:
                raise ValueError(f"Duplicate intent tag: {tag}")

            seen_tags.add(tag)

            if not isinstance(patterns, list) or not patterns:
                raise ValueError(
                    f"Intent '{tag}' requires at least one pattern."
                )

        print(f"Loaded {len(intents)} intents.\n")

        return data

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------

    def prepare_training_data(self, intents: dict):

        documents = []
        vocabulary = set()

        self.classes = sorted(
            intent["tag"] for intent in intents["intents"]
        )

        for intent in intents["intents"]:

            tag = intent["tag"]

            for pattern in intent["patterns"]:

                if not isinstance(pattern, str):
                    raise ValueError("Intent patterns must be strings.")

                tokens = self.preprocessor.tokenize(pattern)

                if not tokens:
                    continue

                vocabulary.update(tokens)
                documents.append((tokens, tag))

        self.words = sorted(vocabulary)

        if not documents or not self.words:
            raise ValueError("Insufficient training data.")

        class_to_index = {
            tag: index
            for index, tag in enumerate(self.classes)
        }

        training_data = []

        for tokens, tag in documents:

            token_set = set(tokens)

            bag = [
                float(word in token_set)
                for word in self.words
            ]

            label = class_to_index[tag]

            training_data.append((bag, label))

        random.shuffle(training_data)

        train_x = np.asarray(
            [item[0] for item in training_data],
            dtype=np.float32
        )

        train_y = np.asarray(
            [item[1] for item in training_data],
            dtype=np.int64
        )

        print(f"Vocabulary size: {len(self.words)}")
        print(f"Intent classes: {len(self.classes)}")
        print(f"Training samples: {len(train_x)}\n")

        return train_x, train_y

    # --------------------------------------------------------
    # Build neural network
    # --------------------------------------------------------

    def build_model(
        self,
        input_size: int,
        output_size: int
    ):

        self.model = Model(
            input_size=input_size,
            output_size=output_size
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        print("Neural network initialized.")

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    def train(self, train_x, train_y):

        if self.model is None:
            self.build_model(
                input_size=train_x.shape[1],
                output_size=len(self.classes)
            )

        dataset = TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y)
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        print("Starting neural network training...\n")

        history = {
            "loss": [],
            "accuracy": []
        }

        self.model.train()

        for epoch in range(EPOCHS):

            running_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in loader:

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_y.size(0)

                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

            epoch_loss = running_loss / total
            accuracy = correct / total

            history["loss"].append(epoch_loss)
            history["accuracy"].append(accuracy)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} |"
                    f"Loss: {epoch_loss:.4f} |"
                    f" Accuracy: {(accuracy * 100):.2f}",
                )

        print("Training completed.\n")

        return history

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    def save(self):

        if self.model is None:
            raise RuntimeError("No trained model available.")

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_size": len(self.words),
            "output_size": len(self.classes),
            "words": self.words,
            "classes": self.classes,
            "architecture": "ann",
            "framework": "pytorch"
        }

        torch.save(checkpoint, MODEL_PATH)

        print("Model checkpoint saved: %s", MODEL_PATH)

    # --------------------------------------------------------
    # Complete training pipeline
    # --------------------------------------------------------

    def training(self):

        print("Initializing Neural NLP Engine...\n")

        intents = self.load_intents()

        train_x, train_y = self.prepare_training_data(intents)

        self.build_model(
            input_size=train_x.shape[1],
            output_size=len(self.classes)
        )

        history = self.train(train_x, train_y)

        self.save()

        print("Neural NLP Engine is ready.")

        return history

