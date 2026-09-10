"""End to end check of every component on random data."""
import sys, os
from unittest import mock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.attacks import attack
from src.augment import SimCLRAugment, make_views
from src.external import LogitsOnlyWrapper
from src.models import ProjectionHead, build_backbone
from src.reverse import reverse_attack
from src.ssl_loss import contrastive_loss_on_images, nt_xent
from src.utils import set_seed


def main():
    set_seed(0)
    device = torch.device("cpu")
    x = torch.rand(4, 3, 32, 32, device=device)
    y = torch.randint(0, 4, (4,), device=device)

    backbone = build_backbone("smallcnn", num_classes=4, width=8).to(device).eval()
    head = ProjectionHead(backbone.feat_dim, 32, 16).to(device).eval()
    aug = SimCLRAugment()

    xi = x.clone().requires_grad_(True)
    v = make_views(xi, aug, n_views=2)
    v.sum().backward()
    assert xi.grad is not None and xi.grad.abs().sum() > 0, "augmentations broke the graph"
    print("[ok] differentiable augmentations, views shape:", tuple(v.shape))

    z = torch.randn(4 * 3, 16)
    loss, acc, per_img = nt_xent(z, batch_size=4, n_views=3)
    assert torch.isfinite(loss), "NT-Xent produced non-finite loss"
    print(f"[ok] nt_xent loss={loss.item():.3f} acc={acc:.3f} per_image={tuple(per_img.shape)}")

    zn = F.normalize(z, dim=1)
    s = zn @ zn.t() / 0.2
    ids = torch.arange(4).repeat(3)
    terms = []
    for i in range(12):
        others = [k for k in range(12) if k != i]
        lse = torch.logsumexp(s[i, others], 0)
        terms += [lse - s[i, j] for j in others if ids[j] == ids[i]]
    assert torch.allclose(loss, torch.stack(terms).mean(), atol=1e-5), "nt_xent does not match Eq. 4"
    neg = torch.randn(10, 16)
    _, _, pa = nt_xent(z, batch_size=4, n_views=3, negatives=neg)
    z2 = z.clone()
    z2[1::4] += 1.0
    _, _, pb = nt_xent(z2, batch_size=4, n_views=3, negatives=neg)
    assert torch.allclose(pa[0], pb[0]) and not torch.allclose(pa[1], pb[1]), "negatives leak across images"
    print("[ok] nt_xent matches Eq. 4, and with a negative set each image is scored on its own")

    for kind in ["pgd", "bim", "cw"]:
        d = attack(backbone, x, y, eps=8 / 255, alpha=2 / 255, iters=3, kind=kind)
        assert d.abs().max().item() <= 8 / 255 + 1e-6, f"{kind} exceeded epsilon"
        assert ((x + d).min() >= -1e-6) and ((x + d).max() <= 1 + 1e-6), "image left [0,1]"
    print("[ok] pgd / bim / cw within budget and in [0,1]")

    d = attack(backbone, x, y, eps=8 / 255, iters=2, kind="pgd", lambda_s=1.0,
               ssl_head=head, aug=aug, n_views=2)
    assert d.shape == x.shape
    print("[ok] defense-aware attack runs")

    before, _, _ = contrastive_loss_on_images(x, backbone, head, aug, n_views=2, no_grad=True)
    r, hist = reverse_attack(backbone, head, aug, x, eps_rev=16 / 255, alpha=2 / 255,
                             iters=5, n_views=2, track_every=2, labels=y)
    after, _, _ = contrastive_loss_on_images((x + r).clamp(0, 1), backbone, head, aug, n_views=2, no_grad=True)
    assert r.abs().max().item() <= 16 / 255 + 1e-6, "reverse attack exceeded v"
    print(f"[ok] reverse attack: L_s {before.item():.3f} -> {after.item():.3f}, history={hist}")

    negs = torch.rand(6, 3, 32, 32)
    r2, _ = reverse_attack(backbone, head, aug, x, eps_rev=0.5, alpha=0.1, iters=3, n_views=2,
                           norm="l_2", negatives=negs)
    assert r2.flatten(1).norm(dim=1).max().item() <= 0.5 + 1e-4, "L2 reverse attack exceeded v"
    d = attack(backbone, x, y, eps=0.5, alpha=0.1, iters=2, kind="pgd", norm="l_2", lambda_s=1.0,
               ssl_head=head, aug=aug, n_views=2, negatives=negs)
    assert d.flatten(1).norm(dim=1).max().item() <= 0.5 + 1e-4, "L2 attack exceeded epsilon"
    with mock.patch.dict(sys.modules, {"autoattack": None}):
        try:
            attack(backbone, x, y, kind="autoattack")
        except ImportError as e:
            assert "pip install" in str(e), f"unhelpful ImportError: {e}"
        else:
            raise AssertionError("expected ImportError when autoattack is missing")
    try:
        attack(build_backbone("smallcnn", num_classes=2, width=8).eval(), x, y % 2, kind="autoattack")
    except ValueError as e:
        assert "at least 4 classes" in str(e), f"unexpected error: {e}"
    else:
        raise AssertionError("expected ValueError for a 2 class model with autoattack")
    print("[ok] L2 attack and reverse within budget, negative set works, clear autoattack errors")

    plain = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(), torch.nn.Linear(8, 12), torch.nn.ReLU(),
        torch.nn.Linear(12, 4),
    )
    wrapped = LogitsOnlyWrapper(plain).eval()
    for prm in wrapped.parameters():
        prm.requires_grad_(False)
    xi = x.clone().requires_grad_(True)
    out = wrapped(xi)
    assert isinstance(out, tuple) and len(out) == 2, "wrapper must return (logits, feats)"
    logits, feats = out
    assert feats.shape[1] == wrapped.feat_dim == 12, "wrapper hooked the wrong Linear"
    assert torch.allclose(logits, plain(x)), "wrapper changed the logits"
    assert feats.requires_grad, "features are detached from the input"
    feats.sum().backward()
    assert xi.grad is not None and xi.grad.abs().sum() > 0, "no gradient reaches the input"
    head_w = ProjectionHead(wrapped.feat_dim, 32, 16).eval()
    xi = x.clone().requires_grad_(True)
    l_s, _, _ = contrastive_loss_on_images(xi, wrapped, head_w, aug, n_views=2)
    assert torch.autograd.grad(l_s, xi)[0].abs().sum() > 0, "L_s gradient does not reach the pixels"
    print(f"[ok] LogitsOnlyWrapper: feat_dim={wrapped.feat_dim}, gradients reach the input pixels")

    assert "robustbench" not in sys.modules, "robustbench was imported eagerly"
    with mock.patch.dict(sys.modules, {"robustbench": None, "robustbench.utils": None}):
        try:
            build_backbone("robustbench:Rice2020Overfitting", num_classes=10)
        except ImportError as e:
            assert "pip install" in str(e), f"unhelpful ImportError: {e}"
        else:
            raise AssertionError("expected ImportError when robustbench is missing")
    print("[ok] robustbench is optional (lazy import, clear error when missing)")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
