"""Contrastive loss."""
import torch
import torch.nn.functional as F


def nt_xent(features: torch.Tensor, batch_size: int, n_views: int = 4, temperature: float = 0.2):
    """features: (n_views*batch_size, D), stacked view by view. Returns (loss, acc, per_image_loss)."""
    device = features.device
    z = F.normalize(features, dim=1)

    labels = torch.cat([torch.arange(batch_size, device=device) for _ in range(n_views)], dim=0)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    sim = z @ z.t()

    eye = torch.eye(sim.size(0), dtype=torch.bool, device=device)
    positive_mask = positive_mask[~eye].view(sim.size(0), -1)
    sim = sim[~eye].view(sim.size(0), -1)

    positives = sim[positive_mask.bool()].view(sim.size(0), -1)
    negatives = sim[~positive_mask.bool()].view(sim.size(0), -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)

    per_row = F.cross_entropy(logits, targets, reduction="none")
    loss = per_row.mean()
    acc = (logits.argmax(dim=1) == targets).float().mean().item()
    per_image = per_row.view(n_views, batch_size).mean(dim=0).detach()
    return loss, acc, per_image


def contrastive_loss_on_images(x, backbone, head, aug, n_views=4, temperature=0.2, no_grad=False):
    """Contrastive loss L_s(x) over n_views augmented views of x."""
    from .augment import make_views

    b = x.size(0)
    views = make_views(x, aug, n_views=n_views)
    if no_grad:
        with torch.no_grad():
            _, feats = backbone(views)
            z = head(feats)
            return nt_xent(z, b, n_views, temperature)
    _, feats = backbone(views)
    z = head(feats)
    return nt_xent(z, b, n_views, temperature)
