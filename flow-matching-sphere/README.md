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

## Model

| | |
| --- | --- |
| Architecture | 5-layer MLP, width 256, SiLU, tangent projection at the output |
| Time conditioning | Sinusoidal embedding (128-dim) → 2-layer MLP |
| Base | Uniform on $S^2$ |
| Target | 12 vMF modes, $\kappa = 60$, at icosahedron vertices |
| Sampling | Exponential-map Euler, 80 steps |

## Usage

```bash
pip install -r requirements.txt        # macOS: install torch directly, see below

python train.py --steps 8000 --out checkpoints/sphere.pt
python render.py --checkpoint checkpoints/sphere.pt --out ../out/sphere_dark.gif
```

`requirements.txt` pins `+cpu` wheels, which exist only for Linux and Windows.
On macOS install `torch numpy matplotlib pillow` directly.

## Files

| File | Purpose |
| --- | --- |
| `geometry.py` | Exp/log maps, geodesics, vMF sampling, icosahedral modes |
| `model.py` | `SphereField` with tangent projection |
| `train.py` | Riemannian flow matching training loop |
| `render.py` | Manifold integration, orthographic globe, seamless rotation |