"""
Synthetic hyperspectral data generation.

Generates physically-motivated spectra with:
  • Smooth blackbody-like continua
  • Narrow absorption features (mineral, vegetation, atmospheric)
  • Scene-level spatial correlation (nearby pixels share materials/atmosphere)
  • Sensor simulation via Gaussian spectral response functions

Each training example is a *scene*: a small grid of spatially-correlated
spectra.  One pixel is designated the target; others provide spatial context.
The model sees sensor-convolved measurements for a random subset of bands and
must reconstruct the true high-res radiance.
"""

import math
import torch
from torch.utils.data import Dataset

from .config import HyperspectralConfig


def _wavelength_grid(cfg: HyperspectralConfig, device="cpu"):
    """1-nm resolution wavelength axis."""
    return torch.linspace(cfg.wl_min, cfg.wl_max, cfg.n_wavelengths, device=device)


def _gaussian(x, center, width, amplitude):
    return amplitude * torch.exp(-0.5 * ((x - center) / width) ** 2)


def _random_spectrum(wl, batch_size, device):
    """
    Generate a batch of random plausible radiance spectra.

    Recipe:
      1. Smooth continuum: sum of 2-4 broad Gaussians
      2. Absorption features: 2-8 narrow Gaussian dips
      3. Positive-definite: clamp to small positive floor
    """
    B = batch_size
    spec = torch.zeros(B, wl.shape[0], device=device)

    # --- Continuum ---
    n_cont = torch.randint(2, 5, (1,)).item()
    for _ in range(n_cont):
        center = torch.empty(B, 1, device=device).uniform_(
            wl[0].item() - 200, wl[-1].item() + 200
        )
        width = torch.empty(B, 1, device=device).uniform_(300, 1200)
        amp = torch.empty(B, 1, device=device).uniform_(0.3, 2.0)
        spec = spec + _gaussian(wl.unsqueeze(0), center, width, amp)

    # --- Absorption features ---
    n_abs = torch.randint(2, 9, (1,)).item()
    for _ in range(n_abs):
        center = torch.empty(B, 1, device=device).uniform_(
            wl[0].item(), wl[-1].item()
        )
        width = torch.empty(B, 1, device=device).uniform_(5, 80)
        depth = torch.empty(B, 1, device=device).uniform_(0.05, 0.6)
        spec = spec * (1.0 - _gaussian(
            wl.unsqueeze(0), center, width, depth
        ))

    # --- Ensure positive ---
    spec = spec.clamp(min=0.01)
    return spec


def _generate_scene(wl, n_pixels, device):
    """
    Generate spatially-correlated spectra for a small scene.

    Strategy: generate a "base" spectrum, then perturb it per-pixel with
    decreasing perturbation magnitude for nearby pixels.
    """
    base = _random_spectrum(wl, 1, device).expand(n_pixels, -1).clone()

    for i in range(n_pixels):
        # Per-pixel perturbation: mild spectral variation
        n_perturb = torch.randint(1, 4, (1,)).item()
        for _ in range(n_perturb):
            center = torch.empty(1, 1, device=device).uniform_(
                wl[0].item(), wl[-1].item()
            )
            width = torch.empty(1, 1, device=device).uniform_(20, 200)
            amp = torch.empty(1, 1, device=device).uniform_(-0.3, 0.3)
            base[i] = base[i] + _gaussian(wl.unsqueeze(0), center, width, amp).squeeze(0)

    return base.clamp(min=0.01)


def _simulate_measurement(spectrum, wl_grid, center_wl, fwhm):
    """
    Simulate sensor measurement by convolving spectrum with Gaussian SRF.

    Args:
        spectrum: (N_wl,) true radiance
        wl_grid: (N_wl,) wavelength axis
        center_wl: scalar — centre wavelength
        fwhm: scalar — FWHM of sensor response
    Returns:
        scalar measured radiance
    """
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    weights = torch.exp(-0.5 * ((wl_grid - center_wl) / sigma.clamp(min=0.5)) ** 2)
    weights = weights / weights.sum().clamp(min=1e-8)
    return (spectrum * weights).sum()


class HyperspectralDataset(Dataset):
    """
    On-the-fly synthetic hyperspectral dataset.

    Each sample contains:
      - A target pixel with random sensor measurements (context from target)
      - Optional context pixels with their own measurements
      - Ground truth high-res radiance at target for the target pixel

    The number of context bands and context pixels varies per sample, which
    teaches the model to handle varying information and produce appropriate
    uncertainty.
    """

    def __init__(self, cfg: HyperspectralConfig, n_samples: int, seed: int = 0):
        self.cfg = cfg
        self.n_samples = n_samples
        self.seed = seed
        self.wl = _wavelength_grid(cfg)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Deterministic-ish RNG per sample for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + idx)

        cfg = self.cfg
        device = "cpu"
        wl = self.wl

        # --- Generate scene ---
        grid_size = cfg.scene_grid_size
        n_pixels = grid_size * grid_size
        scene = _generate_scene(wl, n_pixels, device)  # (n_pix, N_wl)

        # Pick target pixel (centre of grid)
        tgt_idx = n_pixels // 2
        tgt_spectrum = scene[tgt_idx]

        # --- Sample context measurements from target pixel ---
        n_ctx_target = torch.randint(
            cfg.min_context_samples, cfg.max_context_samples + 1, (1,), generator=g
        ).item()

        ctx_wls, ctx_fwhms, ctx_rads, ctx_pos = [], [], [], []

        for _ in range(n_ctx_target):
            cw = torch.empty(1).uniform_(cfg.wl_min, cfg.wl_max).item()
            fw = torch.empty(1).uniform_(2.0, 50.0).item()
            rad = _simulate_measurement(
                tgt_spectrum, wl,
                torch.tensor(cw, device=device),
                torch.tensor(fw, device=device),
            )
            ctx_wls.append(cw)
            ctx_fwhms.append(fw)
            ctx_rads.append(rad.item())
            ctx_pos.append([0.0, 0.0])  # target pixel

        # --- Sample context measurements from neighbouring pixels ---
        n_ctx_pixels = torch.randint(0, cfg.max_context_pixels + 1, (1,), generator=g).item()
        if n_ctx_pixels > 0:
            perm = torch.randperm(n_pixels, generator=g)
            ctx_pixel_indices = [i.item() for i in perm if i.item() != tgt_idx][:n_ctx_pixels]

            for pi in ctx_pixel_indices:
                px = (pi % grid_size) - (grid_size // 2)
                py = (pi // grid_size) - (grid_size // 2)
                pix_spectrum = scene[pi]

                n_bands = torch.randint(2, 30, (1,), generator=g).item()
                for _ in range(n_bands):
                    cw = torch.empty(1).uniform_(cfg.wl_min, cfg.wl_max).item()
                    fw = torch.empty(1).uniform_(2.0, 50.0).item()
                    rad = _simulate_measurement(
                        pix_spectrum, wl,
                        torch.tensor(cw, device=device),
                        torch.tensor(fw, device=device),
                    )
                    ctx_wls.append(cw)
                    ctx_fwhms.append(fw)
                    ctx_rads.append(rad.item())
                    ctx_pos.append([float(px), float(py)])

        # --- Target: full-res radiance at the target pixel ---
        n_tgt = torch.randint(
            cfg.min_target_samples, cfg.max_target_samples + 1, (1,), generator=g
        ).item()
        tgt_indices = torch.randperm(len(wl), generator=g)[:n_tgt].sort().values
        tgt_wavelengths = wl[tgt_indices]
        tgt_radiance = tgt_spectrum[tgt_indices]

        return {
            "ctx_radiance": torch.tensor(ctx_rads, dtype=torch.float32),
            "ctx_wavelength": torch.tensor(ctx_wls, dtype=torch.float32),
            "ctx_fwhm": torch.tensor(ctx_fwhms, dtype=torch.float32),
            "ctx_position": torch.tensor(ctx_pos, dtype=torch.float32),
            "query_wavelengths": tgt_wavelengths.float(),
            "target_radiance": tgt_radiance.float(),
        }


def collate_fn(batch):
    """Pad variable-length context and target sets within a batch."""
    max_ctx = max(b["ctx_radiance"].shape[0] for b in batch)
    max_tgt = max(b["query_wavelengths"].shape[0] for b in batch)
    B = len(batch)

    out = {
        "ctx_radiance": torch.zeros(B, max_ctx),
        "ctx_wavelength": torch.zeros(B, max_ctx),
        "ctx_fwhm": torch.zeros(B, max_ctx),
        "ctx_position": torch.zeros(B, max_ctx, 2),
        "ctx_mask": torch.zeros(B, max_ctx, dtype=torch.bool),
        "query_wavelengths": torch.zeros(B, max_tgt),
        "target_radiance": torch.zeros(B, max_tgt),
        "tgt_mask": torch.zeros(B, max_tgt, dtype=torch.bool),
    }

    for i, b in enumerate(batch):
        nc = b["ctx_radiance"].shape[0]
        nt = b["query_wavelengths"].shape[0]
        out["ctx_radiance"][i, :nc] = b["ctx_radiance"]
        out["ctx_wavelength"][i, :nc] = b["ctx_wavelength"]
        out["ctx_fwhm"][i, :nc] = b["ctx_fwhm"]
        out["ctx_position"][i, :nc] = b["ctx_position"]
        out["ctx_mask"][i, :nc] = True
        out["query_wavelengths"][i, :nt] = b["query_wavelengths"]
        out["target_radiance"][i, :nt] = b["target_radiance"]
        out["tgt_mask"][i, :nt] = True

    return out
