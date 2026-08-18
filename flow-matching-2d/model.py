"""Vector field network v_theta(x, t) for continuous flow matching."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal features for continuous time.

    Args:
        t: shape (B,), values in [0, 1].
        dim: output width.
    Returns:
        Tensor of shape (B, dim).
    """
    t = t.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    # t lives in [0, 1] rather than [0, 1000], so rescale before hitting the bank.
    args = t[:, None] * 1000.0 * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class VectorField(nn.Module):
    """4-layer MLP with SiLU activations and sinusoidal time conditioning."""

    def __init__(self, data_dim: int = 2, hidden: int = 256, time_dim: int = 128, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.config = dict(data_dim=data_dim, hidden=hidden, time_dim=time_dim, depth=depth)
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        layers: list[nn.Module] = [nn.Linear(data_dim + hidden, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, data_dim); t: (B,) or scalar. Returns (B, data_dim)."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        h = self.time_mlp(timestep_embedding(t, self.time_dim))
        return self.net(torch.cat([x, h], dim=-1))


def save_checkpoint(model: VectorField, path: str, **extra) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str, device: str = "cpu") -> VectorField:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = VectorField(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
