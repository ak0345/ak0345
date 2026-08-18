"""Base and target distributions for the 2D flow-matching demo."""

import math

import torch

N_MODES = 4
RADIUS = 3.5
STD = 0.35


def mode_centers(n_modes: int = N_MODES, radius: float = RADIUS, device="cpu") -> torch.Tensor:
    """(n_modes, 2) mode locations evenly spaced on a circle."""
    ang = torch.arange(n_modes, device=device, dtype=torch.float32) * (2 * math.pi / n_modes)
    return torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)


def sample_base(n: int, device="cpu", generator=None) -> torch.Tensor:
    """p_0 = N(0, I)."""
    return torch.randn(n, 2, device=device, generator=generator)


def sample_target(n: int, n_modes: int = N_MODES, std: float = STD, device="cpu", generator=None) -> torch.Tensor:
    """p_1 = isotropic Gaussian mixture with uniform mixing weights."""
    centers = mode_centers(n_modes, device=device)
    idx = torch.randint(0, n_modes, (n,), device=device, generator=generator)
    return centers[idx] + std * torch.randn(n, 2, device=device, generator=generator)
