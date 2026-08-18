"""Riemannian flow matching on S^2.

The Euclidean path x_t = (1-t)x_0 + t*x_1 leaves the manifold, so it is
replaced by the geodesic between the endpoints, and the regression target
becomes that geodesic's velocity:

    x_t    = geodesic(x_0, x_1, t)
    L      = E || v_theta(x_t, t) - d/dt geodesic(x_0, x_1, t) ||^2

with v_theta constrained to the tangent space at x_t.

    python train.py --steps 8000 --out checkpoints/sphere.pt
"""

import argparse
import math
import time
from pathlib import Path

import torch

from geometry import SphereMixture, geodesic, sample_uniform
from model import SphereField, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train a Riemannian flow-matching field on S^2.")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--kappa", type=float, default=50.0, help="mode concentration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/sphere.pt")
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target = SphereMixture(kappa=args.kappa, device=device)
    print(f"device: {device} | target: {target.name} ({len(target.centers)} modes)")

    model = SphereField(hidden=args.hidden, depth=args.depth, time_dim=args.time_dim).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    model.train()
    running, start = 0.0, time.time()
    for step in range(1, args.steps + 1):
        x0 = sample_uniform(args.batch_size, device=device)
        x1 = target.sample(args.batch_size, device=device)
        t = torch.rand(args.batch_size, device=device)

        xt, vt = geodesic(x0, x1, t)
        loss = (model(xt, t) - vt).pow(2).sum(-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        running += loss.item()
        if step % args.log_every == 0:
            print(f"step {step:>6}/{args.steps}  loss {running / args.log_every:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}")
            running = 0.0

    if not math.isfinite(loss.item()):
        raise RuntimeError("training diverged - lower the learning rate")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, str(out), steps=args.steps, final_loss=loss.item(), kappa=args.kappa)
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()