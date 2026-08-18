"""Simulation-free flow matching training.

Objective (independent coupling, linear probability path):

    x_t    = (1 - t) * x_0 + t * x_1
    target = x_1 - x_0
    L      = E_{t, x_0, x_1} || v_theta(x_t, t) - (x_1 - x_0) ||^2

Run once locally, commit the checkpoint, and let CI do inference only.

    python train.py --steps 6000 --out checkpoints/vector_field.pt
"""

import argparse
import math
import time
from pathlib import Path

import torch

from data import sample_base, sample_target
from model import VectorField, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train a 2D flow-matching vector field.")
    p.add_argument("--steps", type=int, default=6000)
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
    print(f"device: {device}")

    model = VectorField(hidden=args.hidden, depth=args.depth, time_dim=args.time_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    model.train()
    running, start = 0.0, time.time()
    for step in range(1, args.steps + 1):
        x0 = sample_base(args.batch_size, device=device)
        x1 = sample_target(args.batch_size, device=device)
        t = torch.rand(args.batch_size, device=device)

        xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        target = x1 - x0

        loss = (model(xt, t) - target).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        running += loss.item()
        if step % args.log_every == 0:
            avg = running / args.log_every
            print(f"step {step:>6}/{args.steps}  loss {avg:.4f}  lr {sched.get_last_lr()[0]:.2e}")
            running = 0.0

    if not math.isfinite(loss.item()):
        raise RuntimeError("training diverged - lower the learning rate")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, str(out), steps=args.steps, final_loss=loss.item())
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
