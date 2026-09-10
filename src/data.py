"""CIFAR10 subset loaders.

Images stay in [0, 1] without mean/std normalization, because eps = 8/255 is defined in pixel space.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class RemapLabels(torch.utils.data.Dataset):
    """Wrap a dataset and remap the kept labels to range(K)."""

    def __init__(self, base, keep_classes: List[int]):
        self.base = base
        self.mapping = {c: i for i, c in enumerate(keep_classes)}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.mapping[int(y)]


def _subset_indices(targets, keep_classes: List[int], per_class: Optional[int], seed: int):
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    idx = []
    for c in keep_classes:
        c_idx = np.where(targets == c)[0]
        if per_class is not None and per_class < len(c_idx):
            c_idx = rng.choice(c_idx, size=per_class, replace=False)
        idx.append(c_idx)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx.tolist()


def get_cifar10_subset(
    data_dir: str = "./data",
    keep_classes: Tuple[int, ...] = (0, 1, 3, 5),
    train_per_class: Optional[int] = 2000,
    test_per_class: Optional[int] = 250,
    batch_size: int = 128,
    test_batch_size: int = 128,
    augment_train: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    download: bool = True,
):
    """Returns (train_loader, test_loader, class_names)."""
    keep_classes = list(keep_classes)

    train_tf = transforms.Compose(
        ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment_train else [])
        + [transforms.ToTensor()]
    )
    test_tf = transforms.ToTensor()

    train_full = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_tf)
    test_full = datasets.CIFAR10(data_dir, train=False, download=download, transform=test_tf)

    tr_idx = _subset_indices(train_full.targets, keep_classes, train_per_class, seed)
    te_idx = _subset_indices(test_full.targets, keep_classes, test_per_class, seed + 1)

    train_set = RemapLabels(Subset(train_full, tr_idx), keep_classes)
    test_set = RemapLabels(Subset(test_full, te_idx), keep_classes)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    names = [CIFAR10_CLASSES[c] for c in keep_classes]
    return train_loader, test_loader, names


def fake_loaders(n_classes=4, n_train=64, n_test=32, batch_size=16):
    """Random noise loaders for smoke tests."""
    def make(n):
        x = torch.rand(n, 3, 32, 32)
        y = torch.randint(0, n_classes, (n,))
        return torch.utils.data.TensorDataset(x, y)

    return (
        DataLoader(make(n_train), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(make(n_test), batch_size=batch_size, shuffle=False),
        [f"class_{i}" for i in range(n_classes)],
    )
