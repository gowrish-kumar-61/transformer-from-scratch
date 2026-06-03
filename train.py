"""
transformer-from-scratch/train.py

Training loop with:
  - AdamW optimiser + cosine LR schedule
  - Gradient clipping
  - Train/val loss logging
  - Checkpoint save/load
  - Sample generation every N steps

OPTIMISER — AdamW:
  Adam = Adaptive Moment Estimation
    m_t = β₁·m_{t-1} + (1-β₁)·g_t         (1st moment: gradient EMA)
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²        (2nd moment: squared grad EMA)
    θ   = θ - lr · m̂_t / (√v̂_t + ε)      (bias-corrected update)

  W vs AdamW:
    Adam applies L2 reg by adding λθ to gradient → interacts with
    adaptive scaling (different effective LR per param → wrong decay).
    AdamW decouples weight decay: θ ← θ(1-λ) - lr·adam_step.
    Correct weight decay. GPT-2 uses AdamW; we do too.

GRADIENT CLIPPING:
  clip_grad_norm_(params, max_norm=1.0)
  If ||∇|| > 1.0, scale all gradients by 1.0/||∇||.
  Prevents exploding gradients in deep transformers.

COSINE LR SCHEDULE:
  lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T_max))
  Starts high → decays smoothly → floors at lr_min.
  Better than step decay: no sharp discontinuities.
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from model import GPT, count_params
from data import get_dataloaders

# CONFIG

def get_config():
    p = argparse.ArgumentParser(description="Train a character-level GPT")
    p.add_argument("--n_embd",     type=int,   default=64)
    p.add_argument("--n_heads",    type=int,   default=4)
    p.add_argument("--n_layers",   type=int,   default=4)
    p.add_argument("--block_size", type=int,   default=128)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--max_iters",  type=int,   default=3000)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--lr_min",     type=float, default=3e-5)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--weight_decay",type=float,default=0.1)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--eval_every", type=int,   default=200)
    p.add_argument("--eval_iters", type=int,   default=50)
    p.add_argument("--gen_every",  type=int,   default=500)
    p.add_argument("--gen_tokens", type=int,   default=200)
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    p.add_argument("--data_path",  type=str,   default="data/shakespeare.txt")
    return p.parse_args()

# LR SCHEDULE

def cosine_lr(step: int, max_iters: int, lr_max: float, lr_min: float) -> float:
    """
    Cosine annealing without warmup.
    lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))
    """
    if step >= max_iters:
        return lr_min
    ratio = step / max_iters
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * ratio))

# EVAL

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# TRAIN

def train():
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Data
    train_loader, val_loader, dataset = get_dataloaders(
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        data_path=cfg.data_path,
    )

    # Model
    model = GPT(
        vocab_size  = dataset.vocab_size,
        n_embd      = cfg.n_embd,
        n_heads     = cfg.n_heads,
        n_layers    = cfg.n_layers,
        block_size  = cfg.block_size,
        dropout     = cfg.dropout,
    ).to(device)

    print(f"Model params: {count_params(model)}")

    # Optimiser — separate weight decay groups
    # Don't decay biases and 1D params (LayerNorm γ,β, embedding)
    decay_params   = [p for n, p in model.named_parameters()
                      if p.dim() >= 2 and p.requires_grad]
    no_decay_params= [p for n, p in model.named_parameters()
                      if p.dim() <  2 and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Infinite iterator over train_loader
    train_iter = iter(train_loader)
    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    print(f"\n{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10} | {'ms/step':>8}")
    print("─" * 60)

    t0 = time.time()
    for step in range(1, cfg.max_iters + 1):
        # LR schedule
        lr = cosine_lr(step, cfg.max_iters, cfg.lr, cfg.lr_min)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward + backward
        model.train()
        x, y = next_batch()
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        # Logging
        if step % cfg.eval_every == 0 or step == 1:
            val_loss = evaluate(model, val_loader, device, cfg.eval_iters)
            ms = (time.time() - t0) * 1000 / cfg.eval_every
            print(f"{step:>6} | {loss.item():>10.4f} | {val_loss:>10.4f} | {lr:>10.2e} | {ms:>8.1f}")
            t0 = time.time()

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                }
                torch.save(ckpt, os.path.join(cfg.ckpt_dir, "best.pt"))

        # Generation sample
        if step % cfg.gen_every == 0:
            print(f"\n── Sample at step {step} ──")
            prompt = "\n"
            ctx = dataset.encode(prompt).unsqueeze(0).to(device)
            out = model.generate(ctx, max_new_tokens=cfg.gen_tokens,
                                 temperature=0.8, top_k=40)
            print(dataset.decode(out[0]))
            print("─" * 40)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {cfg.ckpt_dir}/best.pt")


if __name__ == "__main__":
    train()