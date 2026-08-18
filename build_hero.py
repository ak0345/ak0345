"""Build the profile header SVGs.

GitHub's markdown sanitiser strips <style>, class, id and inline style
attributes, so CSS cannot live in README.md. It *can* live inside an SVG that
the README references as an image - the browser renders it as a normal
document, gradients, filters and CSS animations included.

Two files are emitted (dark / light) and selected in the README with a
<picture> element keyed on prefers-color-scheme.

    python assets/build_hero.py
"""

from pathlib import Path

import numpy as np

W, H = 1000, 230

NAME = "Muhammad Ali Khan"
ROLE = "MSc Data Science &amp; Machine Learning, UCL"
TAGS = ["flow matching", "discrete diffusion", "GFlowNets", "reinforcement learning"]

THEMES = {
    "dark": dict(
        bg0="#0b0f14", bg1="#131c27", edge="#1f2a37",
        text="#e6edf3", muted="#8b949e",
        accent="#7c9cff", accent2="#5eead4",
        cloud="#3b4a5f", arc="#7c9cff", arc_op=0.17, dot_op=0.88,
    ),
    "light": dict(
        bg0="#ffffff", bg1="#eef2f7", edge="#d8dee6",
        text="#0b0f14", muted="#57606a",
        accent="#4457c9", accent2="#0f9d78",
        cloud="#b9c3d0", arc="#4457c9", arc_op=0.16, dot_op=0.92,
    ),
}

# --- flow motif geometry -----------------------------------------------------
MOTIF_CX, MOTIF_CY, MOTIF_R = 815.0, 115.0, 62.0
N_PARTICLES = 110


def motif_paths(seed: int = 7):
    """Particles drifting from a Gaussian blob into four modes, as bezier arcs.

    Same story as Pipeline 1: base distribution on the left, mixture on the
    right, one curve per particle.
    """
    rng = np.random.default_rng(seed)
    ang = np.arange(4) * (np.pi / 2) + np.pi / 4
    centers = np.stack([MOTIF_CX + MOTIF_R * np.cos(ang), MOTIF_CY + MOTIF_R * np.sin(ang)], axis=-1)

    start = np.stack([
        rng.normal(MOTIF_CX - 150, 26, N_PARTICLES),
        rng.normal(MOTIF_CY, 32, N_PARTICLES),
    ], axis=-1)

    idx = rng.integers(0, 4, N_PARTICLES)
    end = centers[idx] + rng.normal(0, 13, (N_PARTICLES, 2))

    # Control point offset perpendicular to travel, so paths fan out.
    mid = (start + end) / 2
    delta = end - start
    perp = np.stack([-delta[:, 1], delta[:, 0]], axis=-1)
    perp /= np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9
    ctrl = mid + perp * rng.normal(0, 18, (N_PARTICLES, 1))

    return start, ctrl, end, idx


def build(theme: str) -> str:
    c = THEMES[theme]
    start, ctrl, end, idx = motif_paths()
    mode_color = [c["accent"], c["accent2"], c["accent"], c["accent2"]]

    arcs, seeds, dots, sparks = [], [], [], []
    for i in range(N_PARTICLES):
        (x0, y0), (cx, cy), (x1, y1) = start[i], ctrl[i], end[i]
        d = f"M{x0:.1f},{y0:.1f} Q{cx:.1f},{cy:.1f} {x1:.1f},{y1:.1f}"

        # Base layer is fully drawn, so the banner still reads correctly in any
        # renderer that ignores CSS animation (and under reduced-motion).
        arcs.append(f'<path class="arc" d="{d}"/>')
        seeds.append(f'<circle class="seed" cx="{x0:.1f}" cy="{y0:.1f}" r="1.5"/>')
        dots.append(f'<circle class="dot" cx="{x1:.1f}" cy="{y1:.1f}" r="2.1" fill="{mode_color[idx[i]]}"/>')

        # Motion comes from a light pulse travelling along every third path.
        if i % 3 == 0:
            delay = round((i % 27) * 0.34, 2)
            sparks.append(
                f'<path class="spark" style="animation-delay:{delay}s" d="{d}" '
                f'stroke="{mode_color[idx[i]]}"/>'
            )

    # Let the text flow naturally - hand-computed tspan offsets space unevenly.
    sep = '<tspan class="sep">  /  </tspan>'
    tags_markup = sep.join(f'<tspan class="tag">{t}</tspan>' for t in TAGS)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" role="img"
     aria-label="{NAME} - {ROLE.replace('&amp;', 'and')}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{c['bg0']}"/>
      <stop offset="100%" stop-color="{c['bg1']}"/>
    </linearGradient>
    <linearGradient id="rule" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{c['accent']}"/>
      <stop offset="55%" stop-color="{c['accent2']}"/>
      <stop offset="100%" stop-color="{c['accent2']}" stop-opacity="0"/>
    </linearGradient>
    <radialGradient id="vignette" cx="0.78" cy="0.5" r="0.5">
      <stop offset="0%" stop-color="{c['accent']}" stop-opacity="0.13"/>
      <stop offset="100%" stop-color="{c['accent']}" stop-opacity="0"/>
    </radialGradient>
  </defs>

  <style>
    .name {{
      font: 600 38px ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      fill: {c['text']}; letter-spacing: -0.6px;
    }}
    .role {{
      font: 400 15.5px ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      fill: {c['muted']};
    }}
    .tags {{ font: 500 12.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .tag  {{ fill: {c['accent']}; }}
    .sep  {{ fill: {c['muted']}; opacity: 0.55; }}

    .arc   {{ fill: none; stroke: {c['arc']}; stroke-width: 0.7; opacity: {c['arc_op']}; }}
    .seed  {{ fill: {c['cloud']}; opacity: 0.75; }}
    .dot   {{ opacity: {c['dot_op']}; }}
    .spark {{ fill: none; stroke-width: 1.5; stroke-linecap: round;
             stroke-dasharray: 22 300; stroke-dashoffset: 300; opacity: 0;
             animation: travel 7s cubic-bezier(.45,.05,.3,1) infinite; }}

    @keyframes travel {{
      0%       {{ stroke-dashoffset: 300; opacity: 0; }}
      8%       {{ opacity: 0.85; }}
      55%      {{ stroke-dashoffset: 40; opacity: 0.85; }}
      70%, 100% {{ stroke-dashoffset: 20; opacity: 0; }}
    }}

    /* Respect the reader's motion preference. */
    @media (prefers-reduced-motion: reduce) {{
      .spark {{ animation: none; opacity: 0; }}
    }}
  </style>

  <rect width="{W}" height="{H}" rx="14" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" rx="14" fill="url(#vignette)"/>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14"
        fill="none" stroke="{c['edge']}"/>

  <g>{''.join(seeds)}</g>
  <g>{''.join(arcs)}</g>
  <g>{''.join(sparks)}</g>
  <g>{''.join(dots)}</g>

  <rect x="42" y="52" width="112" height="3" rx="1.5" fill="url(#rule)"/>
  <text x="42" y="104" class="name">{NAME}</text>
  <text x="42" y="132" class="role">{ROLE}</text>
  <text x="42" y="168" class="tags">{tags_markup}</text>
</svg>
"""


def main():
    out_dir = Path(__file__).resolve().parent
    for theme in THEMES:
        path = out_dir / f"hero-{theme}.svg"
        path.write_text(build(theme), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
