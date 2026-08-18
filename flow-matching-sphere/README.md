# Riemannian Flow Matching on $S^2$

Flow matching where the data lives on a manifold rather than in $\mathbb{R}^n$.
A uniform distribution on the sphere is transported onto twelve von Mises–Fisher
modes at the vertices of an icosahedron.

<p align="center">
  <img src="https://raw.githubusercontent.com/ak0345/ak0345/renders/sphere_dark.gif" width="460" alt="Particles flowing across a rotating globe into an icosahedral constellation">
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

## The coupling is not optional here

In $\mathbb{R}^n$ the pairing of $x_0$ with $x_1$ is a performance choice — it
changes how many integration steps you need. On a compact manifold with a
symmetric target it decides whether the problem is learnable at all.

Averaged over twelve icosahedrally arranged modes, the expected direction from
any point on the sphere cancels almost exactly. Independent pairing therefore
hands the network a marginal field that is close to zero nearly everywhere, with
all the structure compressed into $t \to 1$. The model converges within a few
hundred steps to predicting nothing, and the loss stalls. Measured at $t = 0.5$
on a batch of 8192:

| | independent | geodesic OT |
| --- | --- | --- |
| mean pairing angle | 89.7° | **16.4°** |
| loss of a zero predictor | 2.913 | **0.098** |
| within-region velocity spread | 0.967 | **0.171** |

Pairing each minibatch by squared *geodesic* distance — the angle between
points, not the chord — leaves a sharp local field a small MLP fits easily.
`--coupling independent` is kept so the degenerate case can be reproduced; it is
a cleaner demonstration of why coupling matters than anything in the flat case,
because it fails outright rather than merely costing steps.

## Model

| | |
| --- | --- |
| Architecture | 5-layer MLP, width 256, SiLU, tangent projection at the output |
| Time conditioning | Sinusoidal embedding (128-dim) → 2-layer MLP |
| Base | Uniform on $S^2$ |
| Target | 12 vMF modes, $\kappa = 60$, at icosahedron vertices |
| Coupling | Minibatch geodesic OT, blocks of 256 |
| Sampling | Exponential-map Euler, 80 steps |

## Usage

```bash
pip install -r requirements.txt        # macOS: install torch directly, see below

python train.py --steps 8000 --out checkpoints/sphere.pt

# Reproduce the degenerate case: loss stalls near 2.9 and never recovers.
python train.py --steps 2000 --coupling independent --out /tmp/degenerate.pt
python render.py --checkpoint checkpoints/sphere.pt --out ../out/sphere_dark.gif
```

`requirements.txt` pins `+cpu` wheels, which exist only for Linux and Windows.
On macOS install `torch numpy matplotlib pillow` directly.

## Files

| File | Purpose |
| --- | --- |
| `geometry.py` | Exp/log maps, geodesics, vMF sampling, icosahedral modes |
| `coupling.py` | Independent and geodesic-OT pairings |
| `model.py` | `SphereField` with tangent projection |
| `train.py` | Riemannian flow matching training loop |
| `render.py` | Manifold integration, orthographic globe, seamless rotation |