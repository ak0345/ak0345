<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/hero-light.svg">
  <img alt="Muhammad Ali Khan — MSc Data Science and Machine Learning, UCL" src="assets/hero-dark.svg" width="100%">
</picture>

<p align="center">
  <a href="https://linkedin.com/in/muhammad-ali-khan"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-muhammad--ali--khan-7c9cff?style=flat-square&logo=linkedin&logoColor=white&labelColor=0b0f14"></a>
  &nbsp;
  <a href="mailto:alikhan232002@gmail.com"><img alt="Email" src="https://img.shields.io/badge/Email-alikhan232002%40gmail.com-5eead4?style=flat-square&logo=gmail&logoColor=white&labelColor=0b0f14"></a>
  &nbsp;
  <img alt="Location" src="https://img.shields.io/badge/London-UK-8b949e?style=flat-square&logo=googlemaps&logoColor=white&labelColor=0b0f14">
</p>

<p align="center">
  <b>I work on generative models (flow matching, discrete diffusion, GFlowNets) and on the reinforcement learning that steers them.</b><br>
  <sub>MSc at UCL (predicted Distinction) · First-class BSc from KCL · two NeurIPS workshop submissions in preparation</sub>
</p>

<br>

## Sample work

### Flow matching: what the coupling costs you

<p align="center">
  <img src="https://raw.githubusercontent.com/ak0345/ak0345/renders/flow_dark.gif" width="100%" alt="Two panels comparing independent and optimal-transport couplings transporting a Gaussian into caffeine">
</p>

Flow matching pins down the marginals but leaves the pairing of noise to data
free. Both panels transport the same Gaussian onto the same molecule with the
same seed; only the coupling differs. Colour tracks each particle's starting
angle, so the shredded colour wheel on the left *is* the conditional averaging
the network is forced to do and it is why that model needs far more
integration steps to sample cleanly. Re-rendered weekly by CI, inference only.

<sub>**PyTorch · minibatch OT · rdkit · GitHub Actions**</sub> &nbsp;
<a href="https://github.com/ak0345/ak0345/tree/main/flow-matching-2d">Code, derivation and the NFE ablation →</a>

<table>
<tr>
<td width="50%" valign="top">

### Flow matching on a manifold

<p align="center">
  <img src="https://raw.githubusercontent.com/ak0345/ak0345/renders/sphere_dark.gif" width="94%" alt="Particles flowing across a rotating globe into an icosahedral constellation">
</p>

Riemannian flow matching on $S^2$: geodesic probability paths, a tangent-space
network, and integration by exponential map. Uniform on the sphere to twelve von
Mises–Fisher modes.

<sub>**Riemannian geometry · PyTorch**</sub>

<a href="https://github.com/ak0345/ak0345/tree/main/flow-matching-sphere">Read the code →</a>

</td>
<td width="50%" valign="top">

### Remasking Discrete Diffusion, live

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/remdm-preview-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/remdm-preview-light.svg">
  <img src="assets/remdm-preview-dark.svg" width="100%" alt="Schematic of any-order masked diffusion decoding with a remasking step">
</picture>

An interactive Space that writes text **out of order** — committing
high-confidence tokens first, then remasking and resampling the ones it got
wrong, side by side with left-to-right decoding.

<sub>**Discrete diffusion · Gradio · Hugging Face Spaces**</sub>

<img alt="Status" src="https://img.shields.io/badge/status-in%20progress-8b949e?style=flat-square&labelColor=0b0f14">

</td>
</tr>
</table>

<br>

## Research

<table>
<tr><td width="24%" valign="top"><b>MSc Dissertation</b><br><sub>UCL AI Centre<br>Prof. Brooks Paige</sub></td>
<td valign="top">

**Where does reward steering stop working on a frozen molecular generative model?**
Trained logit and hidden-state GFlowNet guides under RTB/DB objectives against
full-weight and LoRA fine-tuning on Quetzal / GEOM-Drugs, evaluated on GuacaMol
MPO. Found a hard ceiling on achievable reward set by the pretraining
distribution and identified the mechanism that enforces it.

</td></tr>

<tr><td valign="top"><b>Discrete Diffusion Planners</b><br><sub>Supervised by Jack Parker-Holder<br>Google DeepMind / UCL</sub></td>
<td valign="top">

ReMDM as a **non-myopic trajectory planner** across MiniHack and Craftax: an
order-of-magnitude improvement over PPO, DQN and Decision Transformer baselines,
with the strongest zero-shot OOD transfer of the set. A 25-condition ablation
suite isolated a *double intractability* blocking RL fine-tuning of masked
discrete diffusion which is an analytically intractable log-likelihood, compounded by
non-discriminative reward surrogates under sparse rewards.

</td></tr>

<tr><td valign="top"><b>BSc Dissertation</b><br><sub>KCL · First-class</sub></td>
<td valign="top">

Built a full Settlers of Catan environment from scratch in PyTorch and trained
DQN agents by self-play, scaling from a Mini-Catan simulator to the complete
game. 88.9% win rate against random opponents and more interestingly, a
characterisation of *how* it collapses against stronger play, which is a
readable argument for learned world models.

</td></tr>
</table>

<br>

## Elsewhere

**Pakistan National Disaster Management Authority** — hybrid LSTM+CNN models
predicting urban smog from three years of satellite air quality data (≈8% MSE),
on `xarray`/`rioxarray` pipelines handling billions of points. Deployed in
NDMA's real-time alert system; now expanding from city-level models to a
nationwide spatio-temporal framework.

<br>

## Toolbelt

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-7c9cff?style=flat-square&logo=python&logoColor=white&labelColor=0b0f14">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-7c9cff?style=flat-square&logo=pytorch&logoColor=white&labelColor=0b0f14">
  <img alt="JAX" src="https://img.shields.io/badge/JAX-7c9cff?style=flat-square&logo=google&logoColor=white&labelColor=0b0f14">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-7c9cff?style=flat-square&logo=tensorflow&logoColor=white&labelColor=0b0f14">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-5eead4?style=flat-square&logo=numpy&logoColor=white&labelColor=0b0f14">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-5eead4?style=flat-square&logo=scikitlearn&logoColor=white&labelColor=0b0f14">
  <img alt="xarray" src="https://img.shields.io/badge/xarray-5eead4?style=flat-square&labelColor=0b0f14&color=5eead4">
  <img alt="Weights and Biases" src="https://img.shields.io/badge/W%26B-8b949e?style=flat-square&logo=weightsandbiases&logoColor=white&labelColor=0b0f14">
  <img alt="Linux" src="https://img.shields.io/badge/Linux-8b949e?style=flat-square&logo=linux&logoColor=white&labelColor=0b0f14">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-8b949e?style=flat-square&logo=postgresql&logoColor=white&labelColor=0b0f14">
</p>

<sub>Also: mixed-precision and distributed training, Bayesian deep learning, R, C++, Java.</sub>

<br>

---
