"""Geometry of the 2-sphere.

Everything flow matching needs on a manifold: geodesics as probability paths,
their velocities as regression targets, and the exponential map as the
integrator. All formulas verified against the analytic geodesic - exp-map Euler
retraces a great circle to machine precision.
"""

import math

import torch

EPS = 1e-7


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Remove the radial component, leaving a vector in T_x S^2."""
    return v - (v * x).sum(-1, keepdim=True) * x


def exp_map(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Move from x along the geodesic in direction u for distance |u|."""
    n = u.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.cos(n) * x + torch.sin(n) * (u / n)


def log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tangent vector at x pointing at y, with length equal to their distance."""
    cos = (x * y).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos)
    direction = project_tangent(x, y - x)
    return theta * normalize(direction)


def geodesic(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """Point and velocity on the great circle from x0 to x1 at time t.

    Returns (x_t, dx_t/dt). The velocity is tangent at x_t and has constant
    norm equal to the geodesic distance, which is the conditional vector field
    that Riemannian flow matching regresses onto.
    """
    if t.dim() == 1:
        t = t[:, None]

    cos = (x0 * x1).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos)
    sin = torch.sin(theta).clamp_min(EPS)

    xt = (torch.sin((1 - t) * theta) * x0 + torch.sin(t * theta) * x1) / sin
    vt = theta * (-torch.cos((1 - t) * theta) * x0 + torch.cos(t * theta) * x1) / sin
    return normalize(xt), vt


# --- distributions -----------------------------------------------------------

def sample_uniform(n: int, device="cpu", generator=None) -> torch.Tensor:
    """p_0: uniform on the sphere, via a normalised isotropic Gaussian."""
    return normalize(torch.randn(n, 3, device=device, generator=generator))


def icosahedron_vertices(device="cpu") -> torch.Tensor:
    """Twelve maximally symmetric mode locations."""
    phi = (1 + math.sqrt(5)) / 2
    raw = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            raw += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    return normalize(torch.tensor(raw, dtype=torch.float32, device=device))


def sample_vmf(mu: torch.Tensor, kappa: float, n: int, generator=None) -> torch.Tensor:
    """Exact von Mises-Fisher sampling, specialised to S^2.

    In three dimensions the polar weight has a closed-form inverse CDF, so no
    rejection loop is needed.
    """
    device = mu.device

    # The polar weight is computed in float64 on purpose. For kappa > ~44,
    # exp(-2*kappa) underflows to zero in float32 - and that term is exactly
    # what bounds the logarithm. Lose it and a uniform draw of 0.0 (which
    # torch.rand does return, with probability 2^-24) gives w = -inf, which
    # normalises to NaN and silently poisons training a few thousand steps in.
    u = torch.rand(n, 1, device=device, generator=generator, dtype=torch.float64)
    tail = math.exp(-2 * kappa)                     # exact in float64
    w = 1 + torch.log(u + (1 - u) * tail) / kappa
    w = w.clamp(-1.0, 1.0).to(mu.dtype)             # belt and braces

    angle = torch.rand(n, 1, device=device, generator=generator) * 2 * math.pi
    r = torch.sqrt((1 - w ** 2).clamp_min(0))

    # Orthonormal frame around mu, avoiding a degenerate cross product.
    axis = torch.zeros(n, 3, device=device)
    axis[:, 2] = 1.0
    degenerate = mu[:, 2].abs() > 0.9
    axis[degenerate] = torch.tensor([1.0, 0.0, 0.0], device=device)

    e1 = normalize(torch.cross(mu, axis, dim=-1))
    e2 = torch.cross(mu, e1, dim=-1)
    return normalize(w * mu + r * torch.cos(angle) * e1 + r * torch.sin(angle) * e2)


class SphereMixture:
    """von Mises-Fisher modes at icosahedron vertices.

    Kept for reference, but NOT the default - see the README. Twelve
    symmetrically arranged modes make the marginal field nearly unlearnable:
    the destination of a point near a Voronoi boundary depends on the minibatch,
    so even geodesic OT only exposes ~0.15-0.38 of the velocity signal.
    """

    name = "icosahedron"

    def __init__(self, kappa: float = 60.0, device="cpu"):
        self.centers = icosahedron_vertices(device=device)
        self.kappa = kappa

    def sample(self, n: int, device="cpu", generator=None) -> torch.Tensor:
        idx = torch.randint(0, len(self.centers), (n,), device=device, generator=generator)
        return sample_vmf(self.centers.to(device)[idx], self.kappa, n, generator=generator)


# Tilted so the ring cuts across the graticule rather than sitting on a
# parallel - it reads as a band around the globe rather than a latitude line.
RING_AXIS = (0.42, 0.18, 0.89)


class SphereRing:
    """A band at fixed angular radius from a tilted axis.

    Breaking the symmetry is the point. A uniform base pushed towards a
    rotationally symmetric target leaves a strong, single-valued marginal field:
    ~0.86 of the velocity energy is explainable under geodesic OT, against 0.15
    for the icosahedral mixture.
    """

    name = "ring"

    def __init__(self, radius_deg: float = 55.0, width_deg: float = 5.0,
                 axis=RING_AXIS, device="cpu"):
        self.radius = math.radians(radius_deg)
        self.width = math.radians(width_deg)
        self.axis = normalize(torch.tensor(axis, dtype=torch.float32, device=device))

        # Orthonormal frame spanning the plane the ring is drawn in.
        helper = torch.tensor([0.0, 0.0, 1.0], device=device)
        if abs(float(self.axis[2])) > 0.9:
            helper = torch.tensor([1.0, 0.0, 0.0], device=device)
        self.e1 = normalize(torch.cross(self.axis, helper, dim=-1))
        self.e2 = torch.cross(self.axis, self.e1, dim=-1)

    def sample(self, n: int, device="cpu", generator=None) -> torch.Tensor:
        axis, e1, e2 = self.axis.to(device), self.e1.to(device), self.e2.to(device)

        phi = torch.rand(n, 1, device=device, generator=generator) * 2 * math.pi
        alpha = self.radius + self.width * torch.randn(n, 1, device=device, generator=generator)

        return normalize(
            torch.cos(alpha) * axis
            + torch.sin(alpha) * (torch.cos(phi) * e1 + torch.sin(phi) * e2)
        )

    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """Angle around the ring axis, in [0, 1) - used to colour the render."""
        e1, e2 = self.e1.to(x.device), self.e2.to(x.device)
        ang = torch.atan2((x * e2).sum(-1), (x * e1).sum(-1))
        return (ang / (2 * math.pi)) % 1.0


TARGETS = {"ring": SphereRing, "icosahedron": SphereMixture}


def build_target(kind: str, device="cpu", **kwargs):
    if kind not in TARGETS:
        raise ValueError(f"unknown target {kind!r}, expected one of {list(TARGETS)}")
    return TARGETS[kind](device=device, **kwargs)