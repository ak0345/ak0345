"""How many integration steps does each coupling actually need?

Straighter trajectories can be integrated more coarsely. This sweeps the number
of Euler steps (NFE) and measures energy distance to the true target - a proper
metric for distributions, zero only when they match.

    python evaluate.py --checkpoint checkpoints/caffeine_ind.pt:independent \
                       --checkpoint checkpoints/caffeine_ot.pt:"minibatch OT"
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import build_target
from model import load_checkpoint

NFES = [2, 4, 6, 8, 12, 16, 24, 32, 64, 128]

THEMES = {
    "dark": dict(bg="#0b0f14", fg="#e6edf3", muted="#8b949e", grid="#1f2a37",
                 series=["#f97583", "#5eead4"]),
    "light": dict(bg="#ffffff", fg="#0b0f14", muted="#57606a", grid="#d8dee6",
                  series=["#c9432f", "#0f9d78"]),
}


def parse_args():
    p = argparse.ArgumentParser(description="Energy distance vs NFE for each coupling.")
    p.add_argument("--checkpoint", action="append", required=True, help="path[:label]")
    p.add_argument("--target", choices=["gmm", "image"], default="image")
    p.add_argument("--target-path", type=str, default="targets/caffeine.png")
    p.add_argument("--extent", type=float, default=3.5)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-chart", type=str, default="results/nfe")
    p.add_argument("--out-json", type=str, default="results/nfe.json")
    return p.parse_args()


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """2 E|X-Y| - E|X-X'| - E|Y-Y'|; zero iff the distributions agree."""
    xy = torch.cdist(x, y).mean()
    xx = torch.cdist(x, x).mean()
    yy = torch.cdist(y, y).mean()
    return float(2 * xy - xx - yy)


@torch.no_grad()
def sample(model, n: int, steps: int, generator) -> torch.Tensor:
    x = torch.randn(n, 2, generator=generator)
    dt = 1.0 / steps
    for i in range(steps):
        x = x + model(x, torch.full((n,), i * dt)) * dt
    return x


def crossover(curve_a, curve_b, nfes):
    """Cheapest NFE at which b matches a's best-effort (128-step) quality."""
    reference = curve_a[-1]
    for nfe, value in zip(nfes, curve_b):
        if value <= reference:
            return nfe, reference
    return None, reference


def make_chart(results, nfes, stem: str):
    paths = []
    for theme, c in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110)
        fig.patch.set_facecolor(c["bg"])
        ax.set_facecolor(c["bg"])

        for i, (label, curve) in enumerate(results.items()):
            ax.plot(nfes, curve, marker="o", markersize=4, linewidth=1.8,
                    color=c["series"][i % len(c["series"])], label=label)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("function evaluations (Euler steps)", color=c["muted"], fontsize=9)
        ax.set_ylabel("energy distance to target", color=c["muted"], fontsize=9)
        ax.set_xticks(nfes)
        ax.set_xticklabels(nfes)
        ax.tick_params(colors=c["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(c["grid"])
        ax.grid(True, which="both", color=c["grid"], linewidth=0.6, alpha=0.5)
        leg = ax.legend(frameon=False, fontsize=9)
        for text in leg.get_texts():
            text.set_color(c["fg"])
        fig.tight_layout()

        path = Path(f"{stem}-{theme}.svg")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=c["bg"], transparent=False)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    target = build_target(args.target, args.target_path, extent=args.extent)

    results = {}
    for spec in args.checkpoint:
        path, _, label = spec.partition(":")
        label = label or Path(path).stem
        model = load_checkpoint(path)

        curve = []
        for nfe in NFES:
            scores = []
            for r in range(args.repeats):
                gen = torch.Generator().manual_seed(1234 + r)
                gen_t = torch.Generator().manual_seed(9876 + r)
                fake = sample(model, args.samples, nfe, gen)
                real = target.sample(args.samples, generator=gen_t)
                scores.append(energy_distance(fake, real))
            curve.append(float(np.mean(scores)))
            print(f"{label:>22s}  NFE {nfe:>4d}  energy distance {curve[-1]:.4f}")
        results[label] = curve

    labels = list(results)
    if len(labels) == 2:
        nfe, ref = crossover(results[labels[0]], results[labels[1]], NFES)
        if nfe is not None:
            print(f"\n{labels[1]} reaches {labels[0]}'s 128-step quality "
                  f"(energy distance {ref:.4f}) at {nfe} steps.")

    charts = make_chart(results, NFES, args.out_chart)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"nfes": NFES, "results": results}, indent=2))
    print("wrote " + ", ".join(str(p) for p in charts) + f", {out_json}")


if __name__ == "__main__":
    main()