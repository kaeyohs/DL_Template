import torch
from omegaconf import DictConfig
from torch import nn


class ExampleModel(nn.Module):
    """Placeholder model. Replace with your architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: DictConfig) -> nn.Module:
    """Build the model from the Hydra model config."""
    return ExampleModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
    )
