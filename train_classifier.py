"""Stage 1: train the backbone classifier, with or without adversarial training."""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from src.attacks import attack
from src.data import fake_loaders, get_cifar10_subset
from src.models import build_backbone
from src.utils import AvgMeter, count_params, ensure_dir, get_device, set_seed


def evaluate(model, loader, device, eps, alpha, iters):
    model.eval()
    clean, robust, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits, _ = model(x)
        clean += (logits.argmax(1) == y).sum().item()
        delta = attack(model, x, y, eps=eps, alpha=alpha, iters=iters, kind="pgd")
        with torch.no_grad():
            logits_a, _ = model(x + delta)
        robust += (logits_a.argmax(1) == y).sum().item()
        n += y.size(0)
    return clean / n, robust / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="preactresnet18", choices=["preactresnet18", "smallcnn"])
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--mode", default="fgsm_rs", choices=["std", "fgsm_rs", "pgd"])
    p.add_argument("--classes", type=int, nargs="+", default=[0, 1, 3, 5])
    p.add_argument("--train-per-class", type=int, default=2000)
    p.add_argument("--test-per-class", type=int, default=250)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epsilon", type=float, default=8.0, help="attack budget, in units of 1/255")
    p.add_argument("--alpha", type=float, default=2.0, help="attack step size, in units of 1/255")
    p.add_argument("--at-iters", type=int, default=7)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake-data", action="store_true", help="use random data instead of CIFAR10")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out)

    if args.fake_data:
        train_loader, test_loader, names = fake_loaders(n_classes=len(args.classes), batch_size=args.batch_size)
    else:
        train_loader, test_loader, names = get_cifar10_subset(
            data_dir=args.data_dir,
            keep_classes=tuple(args.classes),
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    model = build_backbone(args.arch, num_classes=len(names), width=args.width).to(device)
    print(f"device={device}  arch={args.arch}  params={count_params(model)/1e6:.2f}M  classes={names}")

    eps, alpha = args.epsilon / 255, args.alpha / 255
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader)
    )

    for epoch in range(args.epochs):
        model.train()
        loss_m, acc_m = AvgMeter(), AvgMeter()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if args.mode == "std":
                delta = torch.zeros_like(x)
            elif args.mode == "fgsm_rs":
                model.eval()
                delta = attack(model, x, y, eps=eps, alpha=1.25 * eps, iters=1, kind="pgd")
                model.train()
            else:
                model.eval()
                delta = attack(model, x, y, eps=eps, alpha=alpha, iters=args.at_iters, kind="pgd")
                model.train()

            logits, _ = model(x + delta)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            loss_m.update(loss.item(), y.size(0))
            acc_m.update((logits.argmax(1) == y).float().mean().item(), y.size(0))

        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={loss_m.avg:.4f}  train_acc={acc_m.avg:.4f}  ({time.time()-t0:.1f}s)")

    clean_acc, robust_acc = evaluate(model, test_loader, device, eps, alpha, args.eval_iters)
    print(f"\nFINAL  clean={clean_acc*100:.2f}%   PGD-{args.eval_iters} robust={robust_acc*100:.2f}%")

    ckpt = os.path.join(args.out, f"backbone_{args.arch}_{args.mode}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": args.arch,
            "width": args.width,
            "classes": args.classes,
            "class_names": names,
            "mode": args.mode,
            "clean_acc": clean_acc,
            "robust_acc": robust_acc,
        },
        ckpt,
    )
    with open(os.path.join(args.out, "train_classifier_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"saved -> {ckpt}")


if __name__ == "__main__":
    main()
