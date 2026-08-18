"""Build the discrete-diffusion schematic for the profile card.

A labelled diagram of any-order masked decoding: tokens are committed in
confidence order, one low-confidence token is remasked and resampled, and the
sequence resolves. Replace with a screen capture once the Space is live.

    python assets/build_remdm_preview.py
"""

from pathlib import Path

W, H = 520, 400
LEFT, TOP = 62, 74
ROW_H = 46
CHIP_H = 28
GAP = 5

SENTENCE = ["the", "model", "writes", "tokens", "in", "any", "order", "it", "likes"]

# Which token indices are committed at each step, top (t=T) to bottom (t=0).
STEPS = [
    ([], None),                                # fully masked canvas
    ([0, 6], None),                            # highest-confidence tokens first
    ([0, 6, 1, 4, 5], None),                   # fill outward from anchors
    ([0, 6, 1, 4, 5, 2, 3, 8], 8),             # token 8 committed but low-confidence
    ([0, 6, 1, 4, 5, 2, 3], 8),                # remasked and resampled
    ([0, 6, 1, 4, 5, 2, 3, 8, 7], 8),          # resolved
]
LABELS = ["t = T", "", "", "", "remask", "t = 0"]

THEMES = {
    "dark": dict(
        bg0="#0b0f14", bg1="#131c27", edge="#1f2a37",
        text="#e6edf3", muted="#8b949e",
        accent="#7c9cff", accent2="#5eead4",
        chip="#1b2532", mask="#161d27", mask_stroke="#2b3949",
    ),
    "light": dict(
        bg0="#ffffff", bg1="#eef2f7", edge="#d8dee6",
        text="#0b0f14", muted="#57606a",
        accent="#4457c9", accent2="#0f9d78",
        chip="#e4e9f2", mask="#f2f5f9", mask_stroke="#c9d2de",
    ),
}

MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
SANS = 'ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'


def chip_width(tok: str) -> float:
    return max(len(tok) * 6.6 + 14, 26)


def build(theme: str) -> str:
    c = THEMES[theme]
    body = []

    for r, (committed, focus) in enumerate(STEPS):
        y = TOP + r * ROW_H
        x = LEFT
        remasked_here = LABELS[r] == "remask"

        for i, tok in enumerate(SENTENCE):
            w = chip_width(tok)
            filled = i in committed
            highlight = focus == i and r >= 3

            if filled:
                fill, stroke = c["chip"], c["accent2"] if highlight else "none"
                stroke_attr = f' stroke="{stroke}" stroke-width="1.2"' if stroke != "none" else ""
                body.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{CHIP_H}" rx="6" '
                    f'fill="{fill}"{stroke_attr}/>'
                )
                body.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + 19}" class="tok" '
                    f'text-anchor="middle">{tok}</text>'
                )
            else:
                dash = ' stroke-dasharray="3 3"' if not highlight else ""
                stroke = c["accent2"] if highlight else c["mask_stroke"]
                body.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{CHIP_H}" rx="6" '
                    f'fill="{c["mask"]}" stroke="{stroke}" stroke-width="1"{dash}/>'
                )
                body.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + 19}" class="mask" '
                    f'text-anchor="middle">·</text>'
                )
            x += w + GAP

        if LABELS[r]:
            cls = "flag" if remasked_here else "step"
            body.append(f'<text x="{LEFT - 12}" y="{y + 19}" class="{cls}" text-anchor="end">{LABELS[r]}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" role="img"
     aria-label="Schematic of any-order masked diffusion decoding with a remasking step">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{c['bg0']}"/>
      <stop offset="100%" stop-color="{c['bg1']}"/>
    </linearGradient>
  </defs>
  <style>
    .tok  {{ font: 500 11.5px {MONO}; fill: {c['text']}; }}
    .mask {{ font: 500 11.5px {MONO}; fill: {c['muted']}; opacity: 0.6; }}
    .step {{ font: 500 10px {MONO}; fill: {c['muted']}; }}
    .flag {{ font: 600 10px {MONO}; fill: {c['accent2']}; }}
    .title {{ font: 600 13px {SANS}; fill: {c['text']}; }}
    .sub   {{ font: 400 11px {SANS}; fill: {c['muted']}; }}
    .arrow {{ stroke: {c['edge']}; stroke-width: 1; }}
  </style>

  <rect width="{W}" height="{H}" rx="12" fill="url(#bg)"/>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="none" stroke="{c['edge']}"/>

  <text x="{LEFT - 20}" y="34" class="title">Any-order decoding</text>
  <text x="{LEFT - 20}" y="52" class="sub">tokens commit by confidence, not left to right</text>

  <line class="arrow" x1="{LEFT - 40}" y1="{TOP + 4}" x2="{LEFT - 40}" y2="{TOP + (len(STEPS) - 1) * ROW_H + CHIP_H - 4}"/>

  {''.join(body)}

  <text x="{LEFT - 20}" y="{H - 22}" class="sub">
    <tspan fill="{c['accent2']}">—</tspan> remasked, then resampled
  </text>
</svg>
"""


def main():
    out_dir = Path(__file__).resolve().parent
    for theme in THEMES:
        path = out_dir / f"remdm-preview-{theme}.svg"
        path.write_text(build(theme), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
