"""Vector field on the sphere: v_theta(x, t) in T_x S^2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry import project_tangent


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    t = t.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * 1000.0 * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class SphereField(nn.Module):
    """MLP on ambient R^3, projected onto the tangent space at the output.

    Projecting rather than parameterising a chart keeps the network simple and
    guarantees the prediction is a legal tangent vector everywhere, with no
    coordinate singularity at the poles.
    """

    def __init__(self, hidden: int = 256, time_dim: int = 128, depth: int = 5):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.config = dict(hidden=hidden, time_dim=time_dim, depth=depth)
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        layers: list[nn.Module] = [nn.Linear(3 + hidden, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        h = self.time_mlp(timestep_embedding(t, self.time_dim))
        raw = self.net(torch.cat([x, h], dim=-1))
        return project_tangent(x, raw)


def save_checkpoint(model: SphereField, path: str, **extra) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str, device: str = "cpu") -> SphereField:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SphereField(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()