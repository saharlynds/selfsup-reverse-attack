"""PGD, BIM and C&W attacks, plus the defense aware attack."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_delta(delta, x, eps, norm):
    if norm == "l_inf":
        delta = delta.clamp(-eps, eps)
    elif norm == "l_2":
        delta = delta.view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=eps).view_as(delta)
    else:
        raise ValueError(norm)
    return (x + delta).clamp(0.0, 1.0) - x


def _step(delta, grad, alpha, norm):
    if norm == "l_inf":
        return delta + alpha * grad.sign()
    g_norm = grad.view(grad.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
    return delta + alpha * grad / (g_norm + 1e-10)


def _init_delta(x, eps, norm, random_start):
    delta = torch.zeros_like(x)
    if random_start:
        if norm == "l_inf":
            delta.uniform_(-eps, eps)
        else:
            delta.normal_()
            n = delta.view(delta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta = delta * r / (n + 1e-10) * eps
    return _clamp_delta(delta, x, eps, norm).requires_grad_(True)


def _cw_margin_loss(logits, y, kappa: float = 50.0):
    """C&W margin loss: push the true logit below the best wrong one."""
    onehot = F.one_hot(y, logits.size(1)).float()
    correct = (onehot * logits).sum(1)
    wrong = ((1 - onehot) * logits - 1e4 * onehot).max(1).values
    return -F.relu(correct - wrong + kappa).sum()


class _Logits(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)[0]


def _autoattack(backbone, x, y, eps, norm):
    model = _Logits(backbone).eval()
    with torch.no_grad():
        n_classes = model(x[:1]).size(1)
    if n_classes < 4:
        raise ValueError(f"standard AutoAttack needs at least 4 classes, the model has {n_classes}")
    try:
        from autoattack import AutoAttack
    except ImportError as e:
        raise ImportError(
            f"kind='autoattack' needs the autoattack package ({e}). "
            "Install it with:  pip install git+https://github.com/fra31/auto-attack"
        ) from e
    adversary = AutoAttack(model, norm={"l_inf": "Linf", "l_2": "L2"}[norm], eps=eps, version="standard",
                           verbose=False, device=x.device)
    adversary.apgd_targeted.n_target_classes = min(9, n_classes - 1)
    adversary.fab.n_target_classes = min(9, n_classes - 1)
    x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
    return (x_adv - x).detach()


def attack(
    backbone,
    x,
    y,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    iters: int = 50,
    norm: str = "l_inf",
    kind: str = "pgd",
    lambda_s: float = 0.0,
    ssl_head=None,
    aug=None,
    n_views: int = 4,
    temperature: float = 0.2,
    negatives=None,
):
    """Return the perturbation delta; lambda_s > 0 gives the defense aware attack."""
    if kind == "none":
        return torch.zeros_like(x)
    if kind == "autoattack":
        if lambda_s > 0:
            raise ValueError("the defense aware attack is not available with autoattack")
        return _autoattack(backbone, x, y, eps, norm)

    random_start = kind == "pgd"
    delta = _init_delta(x, eps, norm, random_start)

    for _ in range(iters):
        logits, _ = backbone(x + delta)
        if kind == "cw":
            loss = _cw_margin_loss(logits, y)
        else:
            loss = F.cross_entropy(logits, y)

        if lambda_s > 0:
            from .ssl_loss import contrastive_loss_on_images

            l_s, _, _ = contrastive_loss_on_images(
                x + delta, backbone, ssl_head, aug, n_views=n_views, temperature=temperature, negatives=negatives
            )
            loss = loss - lambda_s * l_s

        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            new_delta = _step(delta.detach(), grad, alpha, norm)
            new_delta = _clamp_delta(new_delta, x, eps, norm)
        delta = new_delta.requires_grad_(True)

    return delta.detach()
