# Flow Matching in 2D

Simulation-free flow matching that transports a standard Gaussian onto a
molecular structure and a controlled comparison of what the *coupling* choice
costs you at sampling time.

<p align="center">
  <img src="https://raw.githubusercontent.com/ak0345/ak0345/renders/flow_dark.gif" width="100%" alt="Two panels comparing independent and optimal-transport couplings transporting a Gaussian into caffeine">
</p>

Both panels start from the same particles with the same colours. Colour encodes
*starting angle*, so it doubles as a tracer: under OT pairing the colour wheel
survives the transport, under independent pairing it is shredded. That scrambling
is the visual signature of the thing that actually matters, how much conditional
averaging the network is being asked to do.

## The objective

Flow matching regresses the velocity of a linear probability path between $p_0$
and $p_1$. With $t \sim U[0,1]$:

$$x_t = (1 - t)\,x_0 + t\,x_1, \qquad
\mathcal{L}(\theta) = \mathbb{E}\left\lVert v_\theta(x_t, t) - (x_1 - x_0) \right\rVert^2$$

The conditional target $x_1 - x_0$ is constant along each path, so no simulation
is needed during training — every gradient step is one forward pass on a freshly
sampled triple.

## Why the coupling matters

Flow matching constrains only the **marginals**. How $x_0$ is paired with $x_1$
is free, and the choice does not change what the model converges to — it changes
how hard the regression is. Where many paths cross, the network can only predict
their *average* velocity, so the learned field curves and needs fine integration
to follow.

Pairing each minibatch by optimal transport minimises total squared displacement,
which reduces crossings. Measured on a batch of 1024 against the caffeine target:

| | independent | minibatch OT |
| --- | --- | --- |
| mean pairing distance | 2.12 | **0.72** |
| conditional velocity spread at $t{=}0.5$ | 1.34 | **0.23** |

A ~6× reduction in the spread the network must average over. Exact assignment is
$O(n^3)$, so the batch is solved in blocks of 256 — the standard OT-CFM
approximation.

## The payoff: fewer function evaluations

Straighter paths tolerate coarser integration. `evaluate.py` sweeps Euler steps
and measures energy distance to the true target:

<p align="center">
  <img src="results/nfe-dark.svg#gh-dark-mode-only" width="90%" alt="Energy distance against number of function evaluations">
  <img src="results/nfe-light.svg#gh-light-mode-only" width="90%" alt="Energy distance against number of function evaluations">
</p>

> **Headline result:** OT coupling matches independent coupling's 128-step
> quality at **N** steps. *(Run `evaluate.py` and fill this in from
> `results/nfe.json` — the number depends on your training run.)*

This is the argument for rectified flow and few-step distillation in miniature:
the model is unchanged, the data pairing is doing the work.

## Model

| | |
| --- | --- |
| Architecture | 4-layer MLP, width 256, SiLU |
| Time conditioning | Sinusoidal embedding (128-dim) → 2-layer MLP, concatenated with $x$ |
| Parameters | ~300k |
| Sampling | Euler integration of $\dot{x} = v_\theta(x, t)$ |

## The target

`build_target.py` rasterises a SMILES string into a density mask with rdkit.
Pixel intensity is treated as an unnormalised density and sampled exactly:
pick a pixel proportional to intensity, then place the point uniformly inside it.
The atom labels survive, which is why the cloud spells out `N` and `O`.

rdkit is needed **only** to regenerate the mask. The committed 4 KB PNG is all
that training and CI depend on.

## Usage

```bash
pip install -r requirements.txt        # see note on macOS below

# Optional: swap the molecule.
pip install rdkit
python build_target.py --smiles "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"

# Train both couplings (~2 min each on CPU).
python train.py --target image --target-path targets/caffeine.png \
                --coupling independent --out checkpoints/caffeine_ind.pt
python train.py --target image --target-path targets/caffeine.png \
                --coupling ot --out checkpoints/caffeine_ot.pt

# The quantitative comparison.
python evaluate.py --checkpoint checkpoints/caffeine_ind.pt:"independent coupling" \
                   --checkpoint checkpoints/caffeine_ot.pt:"minibatch OT coupling"

# The animation. CI runs exactly this, with a fresh seed.
python render.py --checkpoint checkpoints/caffeine_ind.pt:"independent coupling" \
                 --checkpoint checkpoints/caffeine_ot.pt:"minibatch OT coupling" \
                 --limit 4.0 --limit-y 2.9 --out ../out/flow_dark.gif
```

`requirements.txt` pins `+cpu` wheels, which exist only for Linux and Windows —
it is tuned for the CI runner. On macOS install `torch numpy scipy matplotlib
pillow` directly; pip already gives you a CPU build.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | `VectorField` network and checkpoint helpers |
| `data.py` | Gaussian base, mixture and image-density targets |
| `coupling.py` | Independent and minibatch-OT pairings |
| `train.py` | Flow matching training loop |
| `evaluate.py` | Energy distance vs NFE, chart and JSON |
| `render.py` | ODE solve, multi-panel GIF with motion trails |
| `build_target.py` | SMILES → density mask (rdkit, offline only) |