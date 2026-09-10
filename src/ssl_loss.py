"""Contrastive loss."""
import torch
import torch.nn.functional as F


def nt_xent(features: torch.Tensor, batch_size: int, n_views: int = 4, temperature: float = 0.2, negatives=None):
    """features: (n_views*batch_size, D), stacked view by view. Returns (loss, acc, per_image_loss)."""
    z = F.normalize(features, dim=1)
    n = z.size(0)
    image_id = torch.arange(batch_size, device=z.device).repeat(n_views)
    same_image = image_id[:, None] == image_id[None, :]
    is_self = torch.eye(n, dtype=torch.bool, device=z.device)
    positive = same_image & ~is_self

    scores = (z @ z.t() / temperature).masked_fill(is_self if negatives is None else ~positive, float("-inf"))
    if negatives is not None:
        scores = torch.cat([scores, z @ F.normalize(negatives, dim=1).t() / temperature], dim=1)
        positive = torch.cat([positive, positive.new_zeros(n, negatives.size(0))], dim=1)

    log_prob = scores - scores.logsumexp(dim=1, keepdim=True)
    per_row = -log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1)
    loss = per_row.mean()
    acc = positive.gather(1, scores.argmax(dim=1, keepdim=True)).float().mean().item()
    per_image = per_row.view(n_views, batch_size).mean(dim=0).detach()
    return loss, acc, per_image


def contrastive_loss_on_images(x, backbone, head, aug, n_views=4, temperature=0.2, no_grad=False, negatives=None):
    """Contrastive loss L_s(x) over n_views augmented views of x."""
    from .augment import make_views

    b = x.size(0)
    views = make_views(x, aug, n_views=n_views)
    neg_z = None
    if negatives is not None:
        with torch.no_grad():
            neg_z = head(backbone(aug(negatives))[1])
    if no_grad:
        with torch.no_grad():
            _, feats = backbone(views)
            z = head(feats)
            return nt_xent(z, b, n_views, temperature, neg_z)
    _, feats = backbone(views)
    z = head(feats)
    return nt_xent(z, b, n_views, temperature, neg_z)
