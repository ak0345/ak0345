"""Render a molecule to a binary density mask.

The mask is committed to the repo, so training and CI never depend on rdkit -
it is only needed here, if you want to swap the molecule.

    pip install rdkit
    python build_target.py --smiles "CN1C=NC2=C1C(=O)N(C)C(=O)N2C" --out targets/caffeine.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


def parse_args():
    p = argparse.ArgumentParser(description="Rasterise a SMILES string into a density mask.")
    p.add_argument("--smiles", type=str, default=CAFFEINE)
    p.add_argument("--out", type=str, default="targets/caffeine.png")
    p.add_argument("--render-size", type=int, default=900, help="rdkit canvas size")
    p.add_argument("--mask-size", type=int, default=320, help="stored mask resolution")
    p.add_argument("--bond-width", type=float, default=3.4)
    return p.parse_args()


def main():
    args = parse_args()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise SystemExit(f"could not parse SMILES: {args.smiles}")
    AllChem.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(args.render_size, args.render_size)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()      # a density has no colour channel
    opts.bondLineWidth = args.bond_width
    opts.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    png = drawer.GetDrawingText()
    tmp = Path(args.out).with_suffix(".raw.png")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(png)

    img = Image.open(tmp).convert("L")
    img = img.resize((args.mask_size, args.mask_size), Image.LANCZOS)

    # Ink is dark on white; invert so intensity reads as density.
    arr = 255 - np.asarray(img, dtype=np.float32)
    arr[arr < 40] = 0.0  # kill antialiasing haze so the cloud has clean edges
    if arr.max() <= 0:
        raise SystemExit("mask came out empty")
    arr = (arr / arr.max() * 255).astype(np.uint8)

    out = Path(args.out)
    Image.fromarray(arr).save(out, optimize=True)
    tmp.unlink()

    ink = float((arr > 0).mean())
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, {ink:.1%} ink coverage)")


if __name__ == "__main__":
    main()