#!/usr/bin/env python3
"""
Demonstration script: reconstruct a spectrum with varying context and
visualise how uncertainty changes.

Usage:
    python -m hyperspectral_fm.demo --checkpoint runs/hsf/best.pt
    python -m hyperspectral_fm.demo --untrained   # random weights, shows architecture works

Produces:
  demo_reconstruction.png — reconstruction with 5, 20, 100 context samples
  demo_latent.png         — 2-D t-SNE of latent space for different spectra
"""

import argparse
import os

import torch
import numpy as np

from .config import HyperspectralConfig
from .model import HyperspectralNeuralProcess
from .data import _wavelength_grid, _random_spectrum, _simulate_measurement


def generate_test_spectrum(cfg, device="cpu"):
    """Generate a single ground truth spectrum for demonstration."""
    wl = _wavelength_grid(cfg, device=device)
    spec = _random_spectrum(wl, 1, device).squeeze(0)
    return wl, spec


def make_context(spectrum, wl_grid, n_samples, cfg, device="cpu"):
    """Sample random sensor measurements from a spectrum."""
    indices = torch.randperm(len(wl_grid))[:n_samples].sort().values
    ctx_wls, ctx_fwhms, ctx_rads = [], [], []

    for idx in indices:
        cw = wl_grid[idx].item()
        fw = torch.empty(1).uniform_(5.0, 30.0).item()
        rad = _simulate_measurement(
            spectrum, wl_grid,
            torch.tensor(cw, device=device),
            torch.tensor(fw, device=device),
        )
        ctx_wls.append(cw)
        ctx_fwhms.append(fw)
        ctx_rads.append(rad.item())

    return (
        torch.tensor(ctx_rads, device=device).unsqueeze(0),
        torch.tensor(ctx_wls, device=device).unsqueeze(0),
        torch.tensor(ctx_fwhms, device=device).unsqueeze(0),
        torch.zeros(1, n_samples, 2, device=device),  # target pixel at (0,0)
    )


def demo_reconstruction(model, cfg, device="cpu", out_dir="."):
    """Show reconstruction with varying context sizes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation")
        return

    wl, truth = generate_test_spectrum(cfg, device)
    query_wl = wl.unsqueeze(0).to(device)  # (1, N_wl)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    context_sizes = [5, 20, 100]

    for ax, n_ctx in zip(axes, context_sizes):
        ctx_rad, ctx_wl, ctx_fwhm, ctx_pos = make_context(truth, wl, n_ctx, cfg, device)
        result = model.predict(ctx_rad, ctx_wl, ctx_fwhm, ctx_pos, query_wl)

        wl_np = wl.cpu().numpy()
        truth_np = truth.cpu().numpy()
        mean_np = result["mean"][0].cpu().numpy()
        total_np = result["total_std"][0].cpu().numpy()
        aleat_np = result["aleatoric_std"][0].cpu().numpy()
        epist_np = result["epistemic_std"][0].cpu().numpy()

        ax.plot(wl_np, truth_np, "k-", lw=1.5, label="Ground truth")
        ax.plot(wl_np, mean_np, "b-", lw=1.0, label="Predicted mean")
        ax.fill_between(
            wl_np, mean_np - 2 * total_np, mean_np + 2 * total_np,
            alpha=0.2, color="blue", label="±2σ total",
        )
        ax.fill_between(
            wl_np, mean_np - 2 * epist_np, mean_np + 2 * epist_np,
            alpha=0.15, color="red", label="±2σ epistemic",
        )

        # Mark context measurements
        ax.scatter(
            ctx_wl[0].cpu().numpy(), ctx_rad[0].cpu().numpy(),
            c="red", s=20, zorder=5, label=f"Context ({n_ctx} bands)",
        )

        ax.set_ylabel("Radiance")
        ax.set_title(f"{n_ctx} context measurements")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle("Hyperspectral Neural Process — Radiance Reconstruction", fontsize=14)
    plt.tight_layout()

    path = os.path.join(out_dir, "demo_reconstruction.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def demo_latent_space(model, cfg, device="cpu", out_dir=".", n_spectra=200):
    """Visualise latent space with t-SNE."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib/sklearn not available — skipping latent visualisation")
        return

    wl = _wavelength_grid(cfg, device=device)
    latents = []
    labels = []

    for i in range(n_spectra):
        spec = _random_spectrum(wl, 1, device).squeeze(0)
        ctx_rad, ctx_wl, ctx_fwhm, ctx_pos = make_context(spec, wl, 50, cfg, device)
        lat = model.extract_latent(ctx_rad, ctx_wl, ctx_fwhm, ctx_pos)
        latents.append(lat["mu"][0].cpu().numpy())
        labels.append(i % 10)  # colour by index mod 10

    latents = np.stack(latents)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded = tsne.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap="tab10", s=15, alpha=0.7)
    ax.set_title("Latent Space (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax, label="Spectrum group")
    plt.tight_layout()

    path = os.path.join(out_dir, "demo_latent.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def demo_uncertainty_vs_context(model, cfg, device="cpu", out_dir="."):
    """Plot how total uncertainty decreases as context size increases."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping uncertainty plot")
        return

    wl, truth = generate_test_spectrum(cfg, device)
    query_wl = wl.unsqueeze(0).to(device)

    context_sizes = [2, 5, 10, 20, 50, 100, 150, 200]
    mean_uncertainties = []

    for n_ctx in context_sizes:
        ctx_rad, ctx_wl, ctx_fwhm, ctx_pos = make_context(truth, wl, n_ctx, cfg, device)
        result = model.predict(ctx_rad, ctx_wl, ctx_fwhm, ctx_pos, query_wl)
        mean_uncertainties.append(result["total_std"][0].mean().item())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(context_sizes, mean_uncertainties, "bo-", lw=2)
    ax.set_xlabel("Number of context measurements")
    ax.set_ylabel("Mean total uncertainty (σ)")
    ax.set_title("Uncertainty decreases with more context")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "demo_uncertainty_vs_context.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hyperspectral NP Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--untrained", action="store_true", help="Run with random weights")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="demo_output")
    parser.add_argument("--mc-samples", type=int, default=16)
    args = parser.parse_args()

    cfg = HyperspectralConfig()
    cfg.n_mc_samples = args.mc_samples

    device = args.device
    model = HyperspectralNeuralProcess(cfg).to(device)

    if args.checkpoint and not args.untrained:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Running with untrained (random) weights")

    os.makedirs(args.out, exist_ok=True)

    demo_reconstruction(model, cfg, device, args.out)
    demo_uncertainty_vs_context(model, cfg, device, args.out)
    demo_latent_space(model, cfg, device, args.out)

    print("\nDone!")


if __name__ == "__main__":
    main()
