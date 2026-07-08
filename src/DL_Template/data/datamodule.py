import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from DL_Template.data.dataset import ExampleDataset


def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible RNG seed.

    Without this, NumPy/random state in worker processes is not tied to the
    global seed, so multi-worker loading (num_workers > 0) reintroduces
    nondeterminism even after seed_everything.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders from the Hydra data config."""
    train_set = ExampleDataset(cfg.data.root, split="train")
    val_set = ExampleDataset(cfg.data.root, split="val")

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader
