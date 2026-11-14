"""A small feedforward network that predicts the number of bike rented."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Two hidden layers. Returns one raw number (rentals) per row."""

    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)
