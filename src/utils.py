"""Small helpers: seeding, device selection, running averages."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val, k=1):
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
