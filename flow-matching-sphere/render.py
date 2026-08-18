"""Integrate the learned field on S^2 and render a rotating globe.

The globe completes exactly one revolution over the GIF, so the animation loops
seamlessly. The flow finishes partway through, leaving the rest of the
revolution to show the resulting constellation from every side.

    python render.py --checkpoint checkpoints/sphere.pt --out ../assets/sphere_dark.gif
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from PIL import Image

BG = "#0b0f14"
GLOBE = "#111a24"
RIM = "#22303f"
GRID = "#1b2836"
FG = "#8b949e"


def parse_args():
    p = argparse.ArgumentParser(description="Render Riemannian flow matching on the sphere.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/sphere.pt")
    p.add_argument("--out", type=str, default="../assets/sphere_dark.gif")
    p.add_argument("--particles", type=int, default=1400)
    p.add_argument("--steps", type=int, default=80, help="exponential-map Euler steps")
    p.add_argument("--frames", type=int, default=44, help="one full revolution")
    p.add_argument("--flow-fraction", type=float, default=0.62,
                   help="fraction of the revolution during which the flow runs")
    p.add_argument("--tilt", type=float, default=20.0, help="camera tilt, degrees")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--size", type=float, default=4.6)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--colors", type=int, default=64)
    p.add_argument("--target", choices=["ring", "icosahedron"], default="ring")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# --- projection --------------------------------------------------------------

def rotation(spin: float, tilt: float) -> np.ndarray:
    """Spin about the polar axis, then tilt the camera."""
    cs, ss = math.cos(spin), math.sin(spin)
    rz = np.array([[cs, -ss, 0.0], [ss, cs, 0.0], [0.0, 0.0, 1.0]])
    ct, st = math.cos(tilt), math.sin(tilt)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    return rx @ rz


def project(points: np.ndarray, rot: np.ndarray):
    """Orthographic projection. Returns screen xy and depth (>0 faces camera)."""
    p = points @ rot.T
    return np.stack([p[:, 0], p[:, 2]], axis=-1), -p[:, 1]


def graticule(n_lat: int = 5, n_lon: int = 8, res: int = 90) -> list[np.ndarray]:
    """Latitude rings and longitude great circles as 3D polylines."""
    curves = []
    for lat in np.linspace(-60, 60, n_lat):
        phi = math.radians(lat)
        u = np.linspace(0, 2 * math.pi, res)
        curves.append(np.stack([
            math.cos(phi) * np.cos(u), math.cos(phi) * np.sin(u),
            np.full_like(u, math.sin(phi)),
        ], axis=-1))
    for lon in np.linspace(0, math.pi, n_lon, endpoint=False):
        u = np.linspace(0, 2 * math.pi, res)
        curves.append(np.stack([
            np.cos(u) * math.cos(lon), np.cos(u) * math.sin(lon), np.sin(u),
        ], axis=-1))
    return curves


def visible_segments(curves, rot, cutoff: float = 0.0):
    """Project polylines, dropping any segment that passes behind the globe."""
    segs = []
    for curve in curves:
        xy, depth = project(curve, rot)
        ok = depth > cutoff
        keep = ok[:-1] & ok[1:]
        if keep.any():
            segs.append(np.stack([xy[:-1][keep], xy[1:][keep]], axis=1))
    return np.concatenate(segs, axis=0) if segs else np.empty((0, 2, 2))


# --- rendering ---------------------------------------------------------------

def render_globe(traj: np.ndarray, rgba: np.ndarray, out: str, frames=44,
                 flow_fraction=0.62, tilt=20.0, fps=16, size=4.6, dpi=100,
                 colors=64) -> Path:
    """traj: (S+1, N, 3) on the unit sphere. rgba: (N, 4)."""
    tilt_rad = math.radians(tilt)
    curves = graticule()

    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.add_patch(Circle((0, 0), 1.0, facecolor=GLOBE, edgecolor=RIM, linewidth=1.0, zorder=1))
    grid_lc = LineCollection([], colors=GRID, linewidths=0.55, zorder=2)
    ax.add_collection(grid_lc)
    dots = ax.scatter(np.zeros(0), np.zeros(0), s=6, zorder=3, linewidths=0)
    # Passing c= as an array registers a scalar mappable, which would overwrite
    # per-point face colours on every draw. Detach it.
    dots.set_array(None)
    clock = ax.text(0.5, 0.045, "", transform=ax.transAxes, color=FG, fontsize=9,
                    ha="center", family="monospace", zorder=4)

    n_states = len(traj)
    flow_frames = max(1, int(round(frames * flow_fraction)))

    images = []
    for f in range(frames):
        rot = rotation(2 * math.pi * f / frames, tilt_rad)

        progress = min(1.0, f / flow_frames)
        state = traj[int(round(progress * (n_states - 1)))]

        grid_lc.set_segments(visible_segments(curves, rot))

        xy, depth = project(state, rot)
        front = depth > -0.02          # points on the far side are occluded
        xy, depth = xy[front], depth[front]
        face = rgba[front].copy()

        shade = np.clip(0.35 + 0.65 * depth, 0.0, 1.0)   # dim towards the limb
        face[:, 3] = shade
        order = np.argsort(depth)                        # painter's algorithm

        dots.set_offsets(xy[order])
        dots.set_facecolor(face[order])
        dots.set_sizes(3.0 + 7.0 * shade[order])
        clock.set_text(f"t = {progress:.2f}   ·   exp-map Euler, {n_states - 1} steps")

        fig.canvas.draw()
        img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        images.append(img.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT))

    plt.close(fig)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=int(round(1000 / fps)), loop=0, optimize=True, disposal=2)
    return out_path


def phase_colors(phase: np.ndarray) -> np.ndarray:
    """Cyclic colour around the ring, so the band reads as a continuous sweep."""
    return plt.get_cmap("twilight_shifted")(phase)


def mode_colors(final: np.ndarray, centers: np.ndarray) -> np.ndarray:
    idx = np.argmin(((final[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    return plt.get_cmap("turbo")(idx / max(len(centers) - 1, 1))


def main():
    args = parse_args()
    import torch

    from geometry import build_target, exp_map, icosahedron_vertices, sample_uniform
    from model import load_checkpoint

    seed = args.seed if args.seed is not None else torch.seed() % (2**31)
    print(f"seed: {seed}")
    gen = torch.Generator().manual_seed(int(seed))

    model = load_checkpoint(args.checkpoint)
    x = sample_uniform(args.particles, generator=gen)

    dt = 1.0 / args.steps
    traj = [x.numpy().copy()]
    with torch.no_grad():
        for i in range(args.steps):
            t = torch.full((x.shape[0],), i * dt)
            x = exp_map(x, model(x, t) * dt)   # integrate along the manifold
            traj.append(x.numpy().copy())
    traj = np.stack(traj)

    target = build_target(args.target)
    if hasattr(target, "phase"):
        rgba = phase_colors(target.phase(torch.from_numpy(traj[-1])).numpy())
    else:
        rgba = mode_colors(traj[-1], icosahedron_vertices().numpy())
    out = render_globe(traj, rgba, args.out, frames=args.frames,
                       flow_fraction=args.flow_fraction, tilt=args.tilt,
                       fps=args.fps, size=args.size, dpi=args.dpi, colors=args.colors)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()