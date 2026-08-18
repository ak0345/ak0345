"""Simulation-free flow matching training.

Objective (linear probability path):

    x_t    = (1 - t) * x_0 + t * x_1
    target = x_1 - x_0
    L      = E_{t, x_0, x_1} || v_theta(x_t, t) - (x_1 - x_0) ||^2

The pairing of x_0 with x_1 is set by --coupling. Both couplings induce the
same marginals, so both are valid flow matching - but OT pairing produces much
straighter trajectories and so needs fewer integration steps at sampling time.

    python train.py --target image --target-path targets/caffeine.png \
                    --coupling ot --out checkpoints/caffeine_ot.pt
"""

import argparse
import math
import time
from pathlib import Path

import torch

from coupling import get_coupling
from data import build_target, sample_base
from model import VectorField, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train a 2D flow-matching vector field.")
    p.add_argument("--target", choices=["gmm", "image"], default="gmm")
    p.add_argument("--target-path", type=str, default=None)
    p.add_argument("--extent", type=float, default=3.5)
    p.add_argument("--coupling", choices=["independent", "ot"], default="independent")
    p.add_argument("--ot-block", type=int, default=256, help="block size for the assignment solver")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/vector_field.pt")
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target = build_target(args.target, args.target_path, extent=args.extent)
    couple = get_coupling(args.coupling)
    print(f"device: {device} | target: {target.name} | coupling: {args.coupling}")

    model = VectorField(hidden=args.hidden, depth=args.depth, time_dim=args.time_dim).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    model.train()
    running, start = 0.0, time.time()
    for step in range(1, args.steps + 1):
        x0 = sample_base(args.batch_size, device=device)
        x1 = target.sample(args.batch_size, device=device)

        if args.coupling == "ot":
            x0, x1 = couple(x0, x1, block=args.ot_block)
        else:
            x0, x1 = couple(x0, x1)

        t = torch.rand(args.batch_size, device=device)
        xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        velocity = x1 - x0

        loss = (model(xt, t) - velocity).pow(2).mean()

        if not torch.isfinite(loss):
            # Fail at the step it happens, not after the full run. A NaN here
            # almost always means a bad sample reached the loss, not a bad
            # learning rate - check the target sampler before touching --lr.
            raise RuntimeError(f"loss became {loss.item()} at step {step}")

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
    save_checkpoint(
        model, str(out),
        steps=args.steps, final_loss=loss.item(),
        coupling=args.coupling, target=args.target,
        target_path=args.target_path, extent=args.extent,
    )
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()