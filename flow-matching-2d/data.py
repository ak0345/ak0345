"""Base and target distributions for the 2D flow-matching demo."""

# Defer annotation evaluation so `str | None` parses on Python 3.9.
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

N_MODES = 4
RADIUS = 3.5
STD = 0.35


def sample_base(n: int, device="cpu", generator=None) -> torch.Tensor:
    """p_0 = N(0, I)."""
    return torch.randn(n, 2, device=device, generator=generator)


# --- mixture target ----------------------------------------------------------

def mode_centers(n_modes: int = N_MODES, radius: float = RADIUS, device="cpu") -> torch.Tensor:
    ang = torch.arange(n_modes, device=device, dtype=torch.float32) * (2 * math.pi / n_modes)
    return torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)


class GaussianMixture:
    """Isotropic mixture with uniform weights, arranged on a circle."""

    name = "gmm"

    def __init__(self, n_modes: int = N_MODES, radius: float = RADIUS, std: float = STD):
        self.n_modes, self.radius, self.std = n_modes, radius, std

    def sample(self, n: int, device="cpu", generator=None) -> torch.Tensor:
        centers = mode_centers(self.n_modes, self.radius, device=device)
        idx = torch.randint(0, self.n_modes, (n,), device=device, generator=generator)
        return centers[idx] + self.std * torch.randn(n, 2, device=device, generator=generator)


# --- image-density target ----------------------------------------------------

class ImageDensity:
    """Treat greyscale pixel intensity as an unnormalised 2D density.

    Sampling is exact: pick a pixel with probability proportional to its
    intensity, then place the point uniformly inside that pixel. With enough
    particles the cloud reproduces the drawing, atom labels included.
    """

    name = "image"

    def __init__(self, path: str, extent: float = 3.5):
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
        if arr.sum() <= 0:
            raise ValueError(f"{path} contains no density")

        self.h, self.w = arr.shape
        self.extent = extent
        self.weights = torch.from_numpy((arr / arr.sum()).reshape(-1)).float()

    def sample(self, n: int, device="cpu", generator=None) -> torch.Tensor:
        w = self.weights.to(device)
        flat = torch.multinomial(w, n, replacement=True, generator=generator)
        row = torch.div(flat, self.w, rounding_mode="floor").float()
        col = (flat % self.w).float()

        jitter = torch.rand(n, 2, device=device, generator=generator)
        u = (col + jitter[:, 0]) / self.w      # 0..1, left to right
        v = (row + jitter[:, 1]) / self.h      # 0..1, top to bottom

        x = (u * 2 - 1) * self.extent
        y = (1 - v * 2) * self.extent          # image rows run downward
        return torch.stack([x, y], dim=-1)


def build_target(kind: str, path: str | None = None, extent: float = 3.5):
    if kind == "gmm":
        return GaussianMixture()
    if kind == "image":
        if path is None:
            raise ValueError("--target image requires --target-path")
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found - run build_target.py first")
        return ImageDensity(path, extent=extent)
    raise ValueError(f"unknown target: {kind}")


# Kept so the original scripts still import cleanly.
def sample_target(n: int, n_modes: int = N_MODES, std: float = STD, device="cpu", generator=None):
    return GaussianMixture(n_modes=n_modes, std=std).sample(n, device=device, generator=generator)