"""
neural/checkpoint.py
--------------------
Save / load a trained classifier (weights + vocabulary + classes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from neural.model import Model


@dataclass
class Checkpoint:
    model: Model
    words: list[str]
    classes: list[str]


def save_checkpoint(
    path: Path,
    model: Model,
    words: list[str],
    classes: list[str],
    hidden_sizes: tuple[int, ...],
    dropout: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(words),
            "output_size": len(classes),
            "hidden_sizes": list(hidden_sizes),
            "dropout": dropout,
            "words": words,
            "classes": classes,
            "architecture": "ann",
            "framework": "pytorch",
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> Checkpoint:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Neural model not found: {path}")

    data = torch.load(path, map_location=device, weights_only=True)

    if data.get("framework") != "pytorch":
        raise ValueError("Invalid model framework.")

    words = data["words"]
    classes = data["classes"]

    if len(words) != data["input_size"]:
        raise ValueError("Vocabulary size mismatch.")

    if len(classes) != data["output_size"]:
        raise ValueError("Classes size mismatch.")

    model = Model(
        input_size=data["input_size"],
        output_size=data["output_size"],
        hidden_sizes=tuple(data["hidden_sizes"]),
        dropout=data["dropout"],
    ).to(device)

    model.load_state_dict(data["model_state_dict"])
    model.eval()

    return Checkpoint(model=model, words=words, classes=classes)
