import logging
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    Also forces cuDNN into deterministic mode. This can slow training and
    disables cuDNN autotuning, which is the usual reproducibility trade-off;
    relax these two flags if throughput matters more than exact repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
