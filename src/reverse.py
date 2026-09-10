"""The self supervised reverse attack (Algorithm 1 in the paper)."""
import torch

from .ssl_loss import contrastive_loss_on_images


def reverse_attack(
    backbone,
    ssl_head,
    aug,
    x_in,
    eps_rev: float = 16 / 255,
    alpha: float = 2 / 255,
    iters: int = 40,
    n_views: int = 4,
    temperature: float = 0.2,
    norm: str = "l_inf",
    step: str = "sign",
    random_start: bool = True,
    track_every: int = 0,
    labels=None,
    negatives=None,
):
    """Returns (r, history)."""
    backbone.eval()
    ssl_head.eval()

    r = torch.zeros_like(x_in)
    if random_start:
        if norm == "l_inf":
            r.uniform_(-eps_rev, eps_rev)
        else:
            r.normal_()
            n = r.view(r.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
            r = r / (n + 1e-10) * eps_rev * torch.rand_like(n)
    r = ((x_in + r).clamp(0, 1) - x_in).requires_grad_(True)

    history = []
    for k in range(iters):
        loss, _, _ = contrastive_loss_on_images(
            x_in + r, backbone, ssl_head, aug, n_views=n_views, temperature=temperature, negatives=negatives
        )
        grad = torch.autograd.grad(loss, r)[0]

        with torch.no_grad():
            if step == "sign":
                new_r = r - alpha * grad.sign()
            else:
                new_r = r - alpha * grad

            if norm == "l_inf":
                new_r = new_r.clamp(-eps_rev, eps_rev)
            else:
                new_r = new_r.view(new_r.size(0), -1).renorm(p=2, dim=0, maxnorm=eps_rev).view_as(new_r)
            new_r = (x_in + new_r).clamp(0, 1) - x_in

        r = new_r.requires_grad_(True)

        if track_every and labels is not None and ((k + 1) % track_every == 0):
            with torch.no_grad():
                logits, _ = backbone(x_in + r)
                acc = (logits.argmax(1) == labels).float().mean().item()
            history.append((k + 1, acc))

    return r.detach(), history


@torch.no_grad()
def predict(backbone, x):
    logits, _ = backbone(x)
    return logits.argmax(1)
