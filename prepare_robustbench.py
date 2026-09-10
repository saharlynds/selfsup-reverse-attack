"""Use a pretrained RobustBench model instead of training one in stage 1."""
import argparse
import os

import torch

from src.data import CIFAR10_CLASSES, get_cifar10_subset
from src.external import load_robustbench, robustbench_arch
from src.utils import count_params, ensure_dir, get_device

N_EVAL = 512


@torch.no_grad()
def clean_accuracy(model, loader, device, n_images):
    model.eval()
    correct = seen = 0
    for x, y in loader:
        x, y = x[: n_images - seen].to(device), y[: n_images - seen].to(device)
        assert x.min() >= 0 and x.max() <= 1, "expected [0, 1] images without mean/std normalisation"
        logits, _ = model(x)
        assert logits.size(1) == len(CIFAR10_CLASSES), f"model has {logits.size(1)} outputs, expected 10"
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
        if seen >= n_images:
            break
    return correct / seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Rice2020Overfitting", help="RobustBench model name, case sensitive")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10"],
                   help="only cifar10 is supported")
    p.add_argument("--threat-model", default="Linf", choices=["Linf", "L2", "corruptions"])
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ensure_dir(args.out)

    model = load_robustbench(args.model, args.dataset, args.threat_model)
    print(f"{args.model} ({args.dataset}, {args.threat_model}): {type(model.model).__name__}  "
          f"params={count_params(model)/1e6:.1f}M  feat_dim={model.feat_dim}")

    suffix = "" if args.threat_model == "Linf" else f"_{args.threat_model}"
    path = os.path.join(args.out, f"backbone_robustbench_{args.model}{suffix}.pt")
    ckpt = {
        "state_dict": model.state_dict(),
        "arch": robustbench_arch(args.model, args.threat_model),
        "width": None,
        "classes": list(range(10)),
        "class_names": list(CIFAR10_CLASSES),
        "mode": "pretrained",
        "clean_acc": None,
        "robust_acc": None,
    }
    torch.save(ckpt, path)
    print(f"saved -> {path}")

    _, test_loader, _ = get_cifar10_subset(
        data_dir=args.data_dir, keep_classes=tuple(range(10)), test_per_class=None, test_batch_size=128
    )
    acc = clean_accuracy(model.to(device), test_loader, device, N_EVAL)
    ckpt["clean_acc"] = acc
    torch.save(ckpt, path)
    print(f"clean accuracy on {N_EVAL} CIFAR-10 test images: {acc*100:.2f}%  "
          f"(compare with RobustBench's number for {args.model}; a {N_EVAL}-image sample is within ~3 pts)")


if __name__ == "__main__":
    main()
