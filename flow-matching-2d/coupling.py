"""Couplings between the base and target samples.

Flow matching only specifies the *marginals*, so the pairing of x_0 with x_1 is
a free choice. Independent pairing gives curved, crossing trajectories that the
network has to average over. Pairing each minibatch by optimal transport
straightens them, which is what lets the ODE be solved in fewer steps.

Exact assignment on the full batch is O(n^3), so the batch is cut into blocks
and solved blockwise - the standard OT-CFM approximation.
"""

import torch
from scipy.optimize import linear_sum_assignment


def independent(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair samples as drawn."""
    return x0, x1


def minibatch_ot(x0: torch.Tensor, x1: torch.Tensor, block: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder x1 to minimise total squared distance to x0, block by block."""
    if x0.shape != x1.shape:
        raise ValueError("coupling needs equally sized batches")

    n = x0.shape[0]
    out = torch.empty_like(x1)

    for s in range(0, n, block):
        e = min(s + block, n)
        a, b = x0[s:e], x1[s:e]
        cost = torch.cdist(a, b).pow(2)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        # rows comes back sorted for a square problem, so this is a permutation of b.
        out[s:e] = b[torch.as_tensor(cols, device=b.device)]

    return x0, out


COUPLINGS = {"independent": independent, "ot": minibatch_ot}


def get_coupling(name: str):
    if name not in COUPLINGS:
        raise ValueError(f"unknown coupling {name!r}, expected one of {list(COUPLINGS)}")
    return COUPLINGS[name]