#!/usr/bin/env python3
"""
Training script for the Hyperspectral Neural Process.

Usage:
    python -m hyperspectral_fm.train [--epochs 200] [--device cuda] [--out runs/exp1]

Key training details:
  • KL annealing: β ramps 0→1 over the first N steps so the decoder learns to
    reconstruct before the latent path is engaged.
  • Random context size: each batch sees a different number of context
    measurements, teaching the model to calibrate uncertainty vs. information.
  • Masked loss: variable-length targets within a batch use a mask so padding
    doesn't affect gradients.
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .config import HyperspectralConfig
from .model import HyperspectralNeuralProcess
from .data import HyperspectralDataset, collate_fn
from .loss import elbo_loss


def get_kl_weight(step: int, anneal_steps: int) -> float:
    """Linear annealing from 0 → 1."""
    if anneal_steps <= 0:
        return 1.0
    return min(1.0, step / anneal_steps)


def masked_elbo_loss(output, target_radiance, tgt_mask, kl_weight):
    """ELBO with masking for padded targets."""
    nll_elem = 0.5 * (
        output["logvar"]
        + (target_radiance - output["mean"]) ** 2 / output["logvar"].exp()
        + math.log(2 * math.pi)
    )
    # Zero out padding
    nll_elem = nll_elem * tgt_mask
    nll = nll_elem.sum() / tgt_mask.sum().clamp(min=1)

    from .loss import kl_divergence

    kl = kl_divergence(
        output["post_mu"], output["post_logvar"],
        output["prior_mu"], output["prior_logvar"],
    ).mean()

    total = nll + kl_weight * kl
    return total, {"nll": nll.item(), "kl": kl.item(), "kl_weight": kl_weight}


def train(cfg: HyperspectralConfig, device: str = "cpu", out_dir: str = "runs/hsf"):
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Data
    train_ds = HyperspectralDataset(cfg, cfg.n_train_scenes, seed=42)
    val_ds = HyperspectralDataset(cfg, cfg.n_val_scenes, seed=12345)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=(device != "cpu"),
    )

    # Model
    model = HyperspectralNeuralProcess(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimiser
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    global_step = 0
    best_val_loss = float("inf")
    log = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss, epoch_nll, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            kl_w = get_kl_weight(global_step, cfg.kl_anneal_steps)

            output = model(
                ctx_radiance=batch["ctx_radiance"],
                ctx_wavelength=batch["ctx_wavelength"],
                ctx_fwhm=batch["ctx_fwhm"],
                ctx_position=batch["ctx_position"],
                query_wavelengths=batch["query_wavelengths"],
                ctx_mask=batch["ctx_mask"],
                tgt_radiance=batch["target_radiance"],
            )

            loss, metrics = masked_elbo_loss(
                output, batch["target_radiance"], batch["tgt_mask"], kl_w
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_nll += metrics["nll"]
            epoch_kl += metrics["kl"]
            n_batches += 1
            global_step += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_nll = epoch_nll / n_batches
        avg_kl = epoch_kl / n_batches
        elapsed = time.time() - t0

        # --- Validation ---
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(
                    ctx_radiance=batch["ctx_radiance"],
                    ctx_wavelength=batch["ctx_wavelength"],
                    ctx_fwhm=batch["ctx_fwhm"],
                    ctx_position=batch["ctx_position"],
                    query_wavelengths=batch["query_wavelengths"],
                    ctx_mask=batch["ctx_mask"],
                    tgt_radiance=batch["target_radiance"],
                )
                vloss, _ = masked_elbo_loss(output, batch["target_radiance"], batch["tgt_mask"], 1.0)
                val_loss_sum += vloss.item()
                val_n += 1

        val_loss = val_loss_sum / max(val_n, 1)

        entry = {
            "epoch": epoch, "train_loss": avg_loss, "nll": avg_nll,
            "kl": avg_kl, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0],
            "kl_weight": kl_w, "time": elapsed,
        }
        log.append(entry)
        print(
            f"[{epoch:3d}/{cfg.max_epochs}]  loss={avg_loss:.4f}  nll={avg_nll:.4f}  "
            f"kl={avg_kl:.4f}  val={val_loss:.4f}  β={kl_w:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"epoch_{epoch}.pt"))

    # Save final
    torch.save(model.state_dict(), os.path.join(out_dir, "final.pt"))
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Hyperspectral Neural Process")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-latent", type=int, default=None)
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="runs/hsf")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    args = parser.parse_args()

    cfg = HyperspectralConfig()
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.d_latent is not None:
        cfg.d_latent = args.d_latent
    if args.n_encoder_layers is not None:
        cfg.n_encoder_layers = args.n_encoder_layers
    if args.n_train is not None:
        cfg.n_train_scenes = args.n_train
    if args.n_val is not None:
        cfg.n_val_scenes = args.n_val

    print(f"Device: {args.device}")
    print(f"Config: {cfg}")
    train(cfg, device=args.device, out_dir=args.out)


if __name__ == "__main__":
    main()
