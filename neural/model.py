"""
neural/model.py
---------------
Configurable feed-forward network used by every intent/action classifier.
"""

from __future__ import annotations

import torch.nn as nn

DEFAULT_HIDDEN_SIZES = (128, 64)
DEFAULT_DROPOUT = 0.5


class Model(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        previous = input_size

        for size in hidden_sizes:
            layers += [nn.Linear(previous, size), nn.ReLU(), nn.Dropout(dropout)]
            previous = size

        layers.append(nn.Linear(previous, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
