"""Solve the learned ODE and render the transport as an animated GIF.

One panel per checkpoint, so two couplings can be compared under an identical
seed - same starting cloud, same colours, only the pairing differs.

    python render.py --checkpoint checkpoints/caffeine_ind.pt:independent \
                     --checkpoint checkpoints/caffeine_ot.pt:"minibatch OT" \
                     --out ../assets/flow_dark.gif
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless runners have no display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

BG = "#0b0f14"
FG = "#8b949e"
TITLE = "#e6edf3"


def parse_args():
    p = argparse.ArgumentParser(description="Render flow-matching transport as a GIF.")
    p.add_argument("--checkpoint", action="append", required=True,
                   help="path[:label], repeatable - one panel per checkpoint")
    p.add_argument("--out", type=str, default="../assets/flow_dark.gif")
    p.add_argument("--particles", type=int, default=900)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--hold-ms", type=int, default=1100)
    p.add_argument("--trail", type=int, default=9, help="states of motion blur (0 disables)")
    p.add_argument("--panel-size", type=float, default=3.9)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--colors", type=int, default=64)
    p.add_argument("--limit", type=float, default=4.0, help="half-width of the view")
    p.add_argument("--limit-y", type=float, default=None, help="defaults to --limit")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def integrate(model, x0, steps: int) -> np.ndarray:
    """Euler-solve dx/dt = v(x, t) from t=0 to t=1. Returns (steps+1, N, 2)."""
    import torch

    x = x0.clone()
    dt = 1.0 / steps
    traj = [x.numpy().copy()]
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt)
            x = x + model(x, t) * dt
            traj.append(x.numpy().copy())
    return np.stack(traj)


def angle_colors(x0: np.ndarray) -> np.ndarray:
    """Colour by starting angle.

    This is the point of the comparison: under OT pairing neighbouring starts
    land together and the colour wheel survives the transport, while under
    independent pairing it is shredded.
    """
    ang = (np.arctan2(x0[:, 1], x0[:, 0]) + np.pi) / (2 * np.pi)
    return plt.get_cmap("twilight_shifted")(ang)


def render_panels(panels, out: str, fps=18, stride=3, hold_ms=1100, trail=9,
                  panel_size=3.9, dpi=100, colors=64, limit=4.0, limit_y=None) -> Path:
    """panels: list of dicts with keys 'label', 'traj', 'rgba'."""
    n = len(panels)
    limit_y = limit if limit_y is None else limit_y
    # Keep pixels where the density is: a wide target should not be
    # letterboxed into a square panel.
    panel_h = panel_size * (limit_y / limit)
    fig, axes = plt.subplots(1, n, figsize=(panel_size * n, panel_h + 0.46), dpi=dpi)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(BG)

    artists = []
    for ax, panel in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit_y, limit_y)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(panel["label"], color=TITLE, fontsize=11, pad=8,
                     fontfamily="monospace")

        lc = LineCollection([], linewidths=0.5)
        ax.add_collection(lc)
        dots = ax.scatter(panel["traj"][0][:, 0], panel["traj"][0][:, 1],
                          s=4.5, c=panel["rgba"], alpha=0.9, linewidths=0)
        artists.append((lc, dots, panel))

    clock = fig.text(0.5, 0.028, "", color=FG, fontsize=9, ha="center", family="monospace")
    top = 1 - 0.30 / (panel_h + 0.46)
    bottom = 0.16 / (panel_h + 0.46)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom, wspace=0.02)

    n_states = len(panels[0]["traj"])
    keep = list(range(0, n_states, stride))
    if keep[-1] != n_states - 1:
        keep.append(n_states - 1)

    frames = []
    for f in keep:
        for lc, dots, panel in artists:
            traj, rgba = panel["traj"], panel["rgba"]
            dots.set_offsets(traj[f])

            if trail > 0 and f > 0:
                lo = max(0, f - trail)
                window = traj[lo:f + 1]                       # (k+1, N, 2)
                seg = np.stack([window[:-1], window[1:]], axis=2)  # (k, N, 2, 2)
                k = seg.shape[0]
                seg = seg.transpose(1, 0, 2, 3).reshape(-1, 2, 2)

                age = np.linspace(0.12, 0.55, k)              # older = fainter
                rgba_seg = np.repeat(rgba, k, axis=0).copy()
                rgba_seg[:, 3] = np.tile(age, len(rgba))
                lc.set_segments(seg)
                lc.set_color(rgba_seg)
            else:
                lc.set_segments([])

        clock.set_text(f"t = {f / (n_states - 1):.2f}   ·   Euler, {n_states - 1} steps")
        fig.canvas.draw()
        img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        frames.append(img.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT))

    plt.close(fig)

    frame_ms = int(round(1000 / fps))
    durations = [frame_ms] * (len(frames) - 1) + [max(hold_ms, frame_ms)]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    return out_path


def main():
    args = parse_args()
    import torch

    from model import load_checkpoint

    seed = args.seed if args.seed is not None else torch.seed() % (2**31)
    print(f"seed: {seed}")

    gen = torch.Generator().manual_seed(int(seed))
    x0 = torch.randn(args.particles, 2, generator=gen)
    rgba = angle_colors(x0.numpy())

    panels = []
    for spec in args.checkpoint:
        path, _, label = spec.partition(":")
        model = load_checkpoint(path)
        panels.append({
            "label": label or Path(path).stem,
            "traj": integrate(model, x0, args.steps),   # identical x0 across panels
            "rgba": rgba,
        })

    out = render_panels(panels, args.out, fps=args.fps, stride=args.stride,
                        hold_ms=args.hold_ms, trail=args.trail,
                        panel_size=args.panel_size, dpi=args.dpi,
                        colors=args.colors, limit=args.limit, limit_y=args.limit_y)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()