# Riemannian Flow Matching on $S^2$

Flow matching where the data lives on a manifold rather than in $\mathbb{R}^n$.
A uniform distribution on the sphere is transported onto a tilted band cutting
across the graticule.

<p align="center">
  <img src="https://raw.githubusercontent.com/ak0345/ak0345/renders/sphere_dark.gif" width="460" alt="Particles flowing across a rotating globe into a tilted band">
</p>

The globe completes exactly one revolution per loop, so the animation is
seamless and the final constellation is visible from every side.

## What changes off the flat case

The Euclidean interpolation $x_t = (1-t)x_0 + t x_1$ is useless here — the
straight line between two points on a sphere leaves the sphere. Riemannian flow
matching replaces it with the **geodesic** between the endpoints, and the
regression target becomes that geodesic's velocity:

$$x_t = \exp_{x_0}\!\big(t \log_{x_0}(x_1)\big), \qquad
\mathcal{L}(\theta) = \mathbb{E}\left\lVert v_\theta(x_t, t) - \dot{x}_t \right\rVert^2$$

On $S^2$ the geodesic has a closed form, so nothing needs to be simulated:

$$x_t = \frac{\sin((1-t)\theta)\,x_0 + \sin(t\theta)\,x_1}{\sin\theta}, \qquad
\dot{x}_t = \theta\,\frac{-\cos((1-t)\theta)\,x_0 + \cos(t\theta)\,x_1}{\sin\theta}$$

where $\theta = \arccos\langle x_0, x_1 \rangle$. The velocity is tangent at
$x_t$ and has constant norm $\theta$ — the particle travels the great circle at
uniform speed.

Three consequences for the implementation:

- **The network output must be tangent.** `SphereField` predicts in ambient
  $\mathbb{R}^3$ and projects out the radial component. Projecting beats
  parameterising a chart: no coordinate singularity at the poles.
- **The integrator must stay on the manifold.** Euler steps are taken with the
  exponential map, $x \leftarrow \exp_x(\Delta t \cdot v)$, not by adding a
  vector and renormalising.
- **Sampling the target needs care.** von Mises–Fisher draws use the exact
  inverse-CDF method, which closes in three dimensions and needs no rejection
  loop.

The geometry is verified rather than assumed: tangency $\langle x_t, \dot{x}_t
\rangle = 0$, the norm identity $\lVert \dot{x}_t \rVert = \theta$, and both
endpoints hold to machine precision, and exponential-map Euler retraces an exact
great circle to $\sim10^{-15}$.

## Two things had to be true for this to train

Riemannian flow matching on a compact manifold is easy to set up and easy to
make unlearnable. Both failures below were measured before training anything, by
binning $x_t$ and asking what fraction of the target velocity's energy is
explainable by position alone — an upper bound on any model's $R^2$.

| target | coupling | mean $R^2$ over $t$ |
| --- | --- | --- |
| 12 icosahedral modes | independent | 0.04 |
| 12 icosahedral modes | geodesic OT | 0.29 |
| tilted ring | independent | 0.29 |
| **tilted ring** | **geodesic OT** | **0.86** |

**The target must break symmetry.** Twelve modes in an icosahedral arrangement
are close to isotropic, so the expected direction from any point very nearly
cancels. Worse, a point near a Voronoi boundary between modes has a destination
that depends on which minibatch it landed in, so the conditional field is
genuinely multi-valued and no model can fit it. The loss parks at 2.913 — which
is exactly $\mathbb{E}[\theta^2]$, the loss of predicting zero.

**The coupling must be geodesic.** Pairing by squared angle rather than
independently roughly triples the explainable signal, for the same reason as in
the flat case: fewer crossing paths, less conditional averaging.

Both failure modes are reproducible — `--target icosahedron` and
`--coupling independent` are still there.

## Model

| | |
| --- | --- |
| Architecture | 5-layer MLP, width 256, SiLU, tangent projection at the output |
| Time conditioning | Sinusoidal embedding (128-dim) → 2-layer MLP |
| Base | Uniform on $S^2$ |
| Target | Band at 55° from a tilted axis, 5° width |
| Coupling | Minibatch geodesic OT, blocks of 256 |
| Sampling | Exponential-map Euler, 80 steps |

## Usage

```bash
pip install -r requirements.txt        # macOS: install torch directly, see below

python train.py --steps 8000 --out checkpoints/sphere.pt

# Reproduce either failure: the loss stalls near 2.9 and never recovers.
python train.py --steps 2000 --target icosahedron --out /tmp/degenerate.pt
python render.py --checkpoint checkpoints/sphere.pt --out ../out/sphere_dark.gif
```

`requirements.txt` pins `+cpu` wheels, which exist only for Linux and Windows.
On macOS install `torch numpy matplotlib pillow` directly.

## Files

| File | Purpose |
| --- | --- |
| `geometry.py` | Exp/log maps, geodesics, vMF sampling, ring and mixture targets |
| `coupling.py` | Independent and geodesic-OT pairings |
| `model.py` | `SphereField` with tangent projection |
| `train.py` | Riemannian flow matching training loop |
| `render.py` | Manifold integration, orthographic globe, seamless rotation |