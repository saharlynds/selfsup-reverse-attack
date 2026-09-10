"""Stage 2: train the contrastive head on clean images, with the backbone frozen."""
import argparse
import os
import time

import torch

from src.augment import SimCLRAugment
from src.data import fake_loaders, get_cifar10_subset
from src.models import ProjectionHead, build_backbone
from src.ssl_loss import contrastive_loss_on_images
from src.utils import AvgMeter, ensure_dir, get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="backbone checkpoint from train_classifier.py or prepare_robustbench.py")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--train-per-class", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--proj-hidden", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--no-hue", action="store_true", help="skip hue jitter in the augmentations")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake-data", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = build_backbone(ck["arch"], num_classes=len(ck["class_names"]), width=ck.get("width"))
    backbone.load_state_dict(ck["state_dict"])
    backbone = backbone.to(device).eval()
    for prm in backbone.parameters():
        prm.requires_grad_(False)

    if args.fake_data:
        train_loader, _, _ = fake_loaders(n_classes=len(ck["class_names"]), batch_size=args.batch_size)
    else:
        train_loader, _, _ = get_cifar10_subset(
            data_dir=args.data_dir,
            keep_classes=tuple(ck["classes"]),
            train_per_class=args.train_per_class,
            batch_size=args.batch_size,
            augment_train=False,
            seed=args.seed,
        )

    head = ProjectionHead(backbone.feat_dim, args.proj_hidden, args.proj_dim).to(device)
    aug = SimCLRAugment(use_hue=not args.no_hue)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        head.train()
        loss_m, acc_m = AvgMeter(), AvgMeter()
        t0 = time.time()
        for x, _ in train_loader:
            x = x.to(device)
            loss, acc, _ = contrastive_loss_on_images(
                x, backbone, head, aug, n_views=args.n_views, temperature=args.temperature
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_m.update(loss.item(), x.size(0))
            acc_m.update(acc, x.size(0))
        print(f"epoch {epoch+1:3d}/{args.epochs}  L_s={loss_m.avg:.4f}  contrastive_acc={acc_m.avg:.4f}  ({time.time()-t0:.1f}s)")

    out = os.path.join(args.out, "ssl_head.pt")
    torch.save(
        {
            "state_dict": head.state_dict(),
            "feat_dim": backbone.feat_dim,
            "proj_hidden": args.proj_hidden,
            "proj_dim": args.proj_dim,
            "backbone_ckpt": os.path.basename(args.ckpt),
            "n_views": args.n_views,
            "temperature": args.temperature,
            "use_hue": not args.no_hue,
        },
        out,
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
