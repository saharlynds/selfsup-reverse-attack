"""Plot the figures and print a summary table from evaluate.py results."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.results, f"{args.tag}_*_contrastive.npz")))
    if npz_files:
        d = np.load(npz_files[0])
        plt.figure(figsize=(6, 4))
        bins = np.linspace(
            min(d["clean"].min(), d["attacked"].min(), d["reversed"].min()),
            max(d["clean"].max(), d["attacked"].max(), d["reversed"].max()),
            40,
        )
        for key, label in [("clean", "Clean"), ("attacked", "Attacked"), ("reversed", "Reverse Attacked")]:
            plt.hist(d[key], bins=bins, histtype="step", linewidth=2, label=label)
        plt.xlabel("Contrastive loss $L_s$")
        plt.ylabel("Count")
        plt.title("Contrastive score distribution (cf. Figure 3)")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.out, "fig3_contrastive_hist.png")
        plt.savefig(out, dpi=150)
        print("saved ->", out)

    json_files = sorted(glob.glob(os.path.join(args.results, f"{args.tag}_*.json")))
    for jf in json_files:
        res = json.load(open(jf))
        curve = res.get("acc_vs_iterations", {})
        if not curve:
            continue
        ks = sorted(int(k) for k in curve)
        ys = [curve[str(k)] * 100 for k in ks]
        plt.figure(figsize=(6, 4))
        plt.plot(ks, ys, marker="o", label="Ours (reverse attack)")
        plt.axhline(res["robust_standard"], linestyle="--", color="gray", label="Baseline (no defense)")
        plt.xlabel("Reverse-attack iterations")
        plt.ylabel("Robust accuracy (%)")
        plt.title(f"Speed vs robustness — {res['attack'].upper()} (cf. Figure 5)")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.out, os.path.basename(jf).replace(".json", "_fig5.png"))
        plt.savefig(out, dpi=150)
        print("saved ->", out)

    rows = []
    for jf in json_files:
        r = json.load(open(jf))
        rows.append((r["attack"], r["robust_standard"], r["robust_ours"], r["gain"], r["clean_standard"], r["clean_ours"]))
    if rows:
        print("\n| attack | robust standard | robust ours | gain | clean standard | clean ours |")
        print("|---|---|---|---|---|---|")
        for a, rs, ro, g, cs, co in rows:
            print(f"| {a} | {rs:.2f}% | {ro:.2f}% | {g:+.2f} | {cs:.2f}% | {co:.2f}% |")


if __name__ == "__main__":
    main()
