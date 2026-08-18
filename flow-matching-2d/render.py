"""Solve the learned ODE and render the transport as an animated GIF.

Inference only - this is what CI runs each day with a fresh seed:

    python render.py --checkpoint checkpoints/vector_field.pt --out assets/flow_dark.gif
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless runners have no display

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BG = "#0d1117"  # GitHub dark canvas
FG = "#8b949e"  # GitHub dark muted text


def parse_args():
    p = argparse.ArgumentParser(description="Render the flow-matching transport as a GIF.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/vector_field.pt")
    p.add_argument("--out", type=str, default="assets/flow_dark.gif")
    p.add_argument("--particles", type=int, default=1000)
    p.add_argument("--steps", type=int, default=100, help="Euler integration steps")
    p.add_argument("--stride", type=int, default=3, help="keep every Nth state as a frame")
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--hold-ms", type=int, default=900, help="pause on the final frame")
    p.add_argument("--size", type=float, default=5.5, help="figure size in inches")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--colors", type=int, default=64, help="GIF palette size")
    p.add_argument("--seed", type=int, default=None, help="default: fresh randomness each run")
    return p.parse_args()


def integrate(model, n_particles: int, steps: int, seed=None) -> np.ndarray:
    """Push particles from N(0, I) to p_1 with Euler integration.

    Returns an array of shape (steps + 1, n_particles, 2).
    """
    import torch  # imported here so the renderer stays testable without torch

    gen = torch.Generator()
    if seed is None:
        seed = torch.seed() % (2**31)
    gen.manual_seed(int(seed))
    print(f"seed: {seed}")

    x = torch.randn(n_particles, 2, generator=gen)
    dt = 1.0 / steps
    traj = [x.numpy().copy()]

    with torch.no_grad():
        for i in range(steps):
            t = torch.full((n_particles,), i * dt)
            x = x + model(x, t) * dt
            traj.append(x.numpy().copy())

    return np.stack(traj)


def particle_colors(final: np.ndarray, n_modes: int = 4, radius: float = 3.5) -> np.ndarray:
    """Colour each particle by the mode it lands in, so the split is legible."""
    ang = np.arange(n_modes) * (2 * np.pi / n_modes)
    centers = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=-1)
    nearest = np.argmin(np.linalg.norm(final[:, None, :] - centers[None], axis=-1), axis=1)
    return plt.get_cmap("plasma")(nearest / max(n_modes - 1, 1))


def render_gif(traj: np.ndarray, out: str, fps=18, stride=3, hold_ms=900,
               size=5.5, dpi=100, colors=64) -> Path:
    """Rasterise the trajectory into an optimised GIF."""
    rgba = particle_colors(traj[-1])
    limit = float(np.abs(traj).max()) * 1.08

    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    glow = ax.scatter(traj[0][:, 0], traj[0][:, 1], s=42, c=rgba, alpha=0.10, linewidths=0)
    core = ax.scatter(traj[0][:, 0], traj[0][:, 1], s=5, c=rgba, alpha=0.85, linewidths=0)
    label = ax.text(0.04, 0.94, "", transform=ax.transAxes, color=FG, fontsize=9, family="monospace")

    n_states = len(traj)
    keep = list(range(0, n_states, stride))
    if keep[-1] != n_states - 1:
        keep.append(n_states - 1)

    frames = []
    for f in keep:
        glow.set_offsets(traj[f])
        core.set_offsets(traj[f])
        label.set_text(f"t = {f / (n_states - 1):.2f}")
        fig.canvas.draw()
        img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        frames.append(img.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT))
    plt.close(fig)

    frame_ms = int(round(1000 / fps))
    durations = [frame_ms] * (len(frames) - 1) + [max(hold_ms, frame_ms)]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def main():
    args = parse_args()
    from model import load_checkpoint

    model = load_checkpoint(args.checkpoint)
    traj = integrate(model, args.particles, args.steps, seed=args.seed)
    out = render_gif(
        traj,
        args.out,
        fps=args.fps,
        stride=args.stride,
        hold_ms=args.hold_ms,
        size=args.size,
        dpi=args.dpi,
        colors=args.colors,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
