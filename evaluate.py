"""Stage 3: evaluate the classifier with and without the reverse attack."""
import argparse
import json
import os
import time

import torch

from src.attacks import attack
from src.augment import SimCLRAugment
from src.data import fake_loaders, get_cifar10_subset
from src.models import ProjectionHead, build_backbone
from src.reverse import reverse_attack
from src.ssl_loss import contrastive_loss_on_images
from src.utils import ensure_dir, get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ssl-ckpt", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--test-per-class", type=int, default=250)
    p.add_argument("--n-test", type=int, default=500, help="maximum number of test images")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--attack", default="pgd", choices=["pgd", "bim", "cw", "none"])
    p.add_argument("--attack-iters", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=8.0, help="attack budget, in units of 1/255")
    p.add_argument("--alpha", type=float, default=2.0, help="attack step size, in units of 1/255")
    p.add_argument("--lambda-s", type=float, default=0.0, help="weight of L_s in the defense aware attack; 0 turns it off")
    p.add_argument("--rev-mult", type=float, default=2.0, help="reverse attack budget, as a multiple of epsilon")
    p.add_argument("--rev-iters", type=int, default=40, help="number of reverse attack steps")
    p.add_argument("--rev-step", default="sign", choices=["sign", "grad"])
    p.add_argument("--rev-alpha", type=float, default=2.0, help="reverse attack step size, in units of 1/255")
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--random-reverse", action="store_true", help="baseline: add random noise instead of running the reverse attack")
    p.add_argument("--track-every", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="run")
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

    sk = torch.load(args.ssl_ckpt, map_location="cpu", weights_only=False)
    head = ProjectionHead(sk["feat_dim"], sk["proj_hidden"], sk["proj_dim"])
    head.load_state_dict(sk["state_dict"])
    head = head.to(device).eval()
    for prm in head.parameters():
        prm.requires_grad_(False)

    aug = SimCLRAugment(use_hue=sk.get("use_hue", True))

    if args.fake_data:
        _, test_loader, names = fake_loaders(n_classes=len(ck["class_names"]), batch_size=args.batch_size)
    else:
        _, test_loader, names = get_cifar10_subset(
            data_dir=args.data_dir,
            keep_classes=tuple(ck["classes"]),
            test_per_class=args.test_per_class,
            test_batch_size=args.batch_size,
            seed=args.seed,
        )

    eps = args.epsilon / 255
    alpha = args.alpha / 255
    eps_rev = args.rev_mult * eps
    rev_alpha = args.rev_alpha / 255

    counts = dict(clean_std=0, clean_ours=0, rob_std=0, rob_ours=0, n=0)
    ls_clean, ls_att, ls_rev = [], [], []
    curve = {}
    t0 = time.time()

    for x, y in test_loader:
        if counts["n"] >= args.n_test:
            break
        x, y = x.to(device), y.to(device)

        delta = attack(
            backbone, x, y, eps=eps, alpha=alpha, iters=args.attack_iters, kind=args.attack,
            lambda_s=args.lambda_s, ssl_head=head, aug=aug, n_views=args.n_views,
        )
        x_a = (x + delta).clamp(0, 1)

        with torch.no_grad():
            counts["clean_std"] += (backbone(x)[0].argmax(1) == y).sum().item()
            counts["rob_std"] += (backbone(x_a)[0].argmax(1) == y).sum().item()

        if args.random_reverse:
            r_a = torch.empty_like(x_a).uniform_(-eps_rev, eps_rev)
            r_c = torch.empty_like(x).uniform_(-eps_rev, eps_rev)
            hist = []
        else:
            r_a, hist = reverse_attack(
                backbone, head, aug, x_a, eps_rev=eps_rev, alpha=rev_alpha, iters=args.rev_iters,
                n_views=args.n_views, step=args.rev_step, track_every=args.track_every, labels=y,
            )
            r_c, _ = reverse_attack(
                backbone, head, aug, x, eps_rev=eps_rev, alpha=rev_alpha, iters=args.rev_iters,
                n_views=args.n_views, step=args.rev_step,
            )

        with torch.no_grad():
            counts["rob_ours"] += (backbone((x_a + r_a).clamp(0, 1))[0].argmax(1) == y).sum().item()
            counts["clean_ours"] += (backbone((x + r_c).clamp(0, 1))[0].argmax(1) == y).sum().item()

        for tag, img, store in [
            ("clean", x, ls_clean), ("attacked", x_a, ls_att), ("reversed", (x_a + r_a).clamp(0, 1), ls_rev)
        ]:
            _, _, per_img = contrastive_loss_on_images(
                img, backbone, head, aug, n_views=args.n_views, no_grad=True
            )
            store.extend(per_img.cpu().tolist())

        for k, acc in hist:
            curve.setdefault(k, []).append(acc)

        counts["n"] += y.size(0)
        print(f"  [{counts['n']:5d} imgs]  rob_std={counts['rob_std']/counts['n']*100:.2f}%  "
              f"rob_ours={counts['rob_ours']/counts['n']*100:.2f}%  ({time.time()-t0:.0f}s)")

    n = max(counts["n"], 1)
    res = {
        "tag": args.tag,
        "n_images": counts["n"],
        "attack": args.attack,
        "attack_iters": args.attack_iters,
        "epsilon_255": args.epsilon,
        "lambda_s": args.lambda_s,
        "reverse": {"v_mult": args.rev_mult, "K": args.rev_iters, "step": args.rev_step,
                    "random_baseline": args.random_reverse},
        "clean_standard": counts["clean_std"] / n * 100,
        "clean_ours": counts["clean_ours"] / n * 100,
        "robust_standard": counts["rob_std"] / n * 100,
        "robust_ours": counts["rob_ours"] / n * 100,
        "gain": (counts["rob_ours"] - counts["rob_std"]) / n * 100,
        "contrastive_loss": {
            "clean": sum(ls_clean) / max(len(ls_clean), 1),
            "attacked": sum(ls_att) / max(len(ls_att), 1),
            "reversed": sum(ls_rev) / max(len(ls_rev), 1),
        },
        "acc_vs_iterations": {str(k): sum(v) / len(v) for k, v in sorted(curve.items())},
        "class_names": names,
    }

    print("\n=== RESULTS " + "=" * 46)
    print(f"clean   : standard {res['clean_standard']:.2f}%   ours {res['clean_ours']:.2f}%")
    print(f"robust  : standard {res['robust_standard']:.2f}%   ours {res['robust_ours']:.2f}%   "
          f"(gain {res['gain']:+.2f} pts)")
    print(f"L_s     : clean {res['contrastive_loss']['clean']:.3f} | "
          f"attacked {res['contrastive_loss']['attacked']:.3f} | "
          f"reversed {res['contrastive_loss']['reversed']:.3f}")

    import numpy as np
    np.savez(
        os.path.join(args.out, f"{args.tag}_{args.attack}_contrastive.npz"),
        clean=np.array(ls_clean), attacked=np.array(ls_att), reversed=np.array(ls_rev),
    )
    path = os.path.join(args.out, f"{args.tag}_{args.attack}_eps{int(args.epsilon)}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
