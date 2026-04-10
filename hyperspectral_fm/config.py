"""Model and training configuration."""

from dataclasses import dataclass, field


@dataclass
class HyperspectralConfig:
    # --- Model architecture ---
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 2
    d_ff: int = 512
    d_latent: int = 128
    dropout: float = 0.1

    # Fourier feature frequencies
    n_spectral_freqs: int = 64
    n_spatial_freqs: int = 32

    # --- Spectral domain ---
    wl_min: float = 400.0    # nm
    wl_max: float = 2500.0   # nm
    wl_resolution: float = 1.0  # nm — resolution for ground truth spectra

    # --- Spatial domain ---
    max_spatial_extent: float = 50.0  # max pixel offset for context pixels

    # --- Training ---
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 200
    warmup_steps: int = 1000
    kl_anneal_steps: int = 5000  # steps to ramp KL weight from 0 → 1

    # Context / target sampling during training
    min_context_samples: int = 3
    max_context_samples: int = 200
    min_target_samples: int = 50
    max_target_samples: int = 200
    max_context_pixels: int = 8

    # Synthetic data
    n_train_scenes: int = 50_000
    n_val_scenes: int = 2_000
    scene_grid_size: int = 5  # pixels per side in synthetic scene

    # Inference
    n_mc_samples: int = 32  # Monte Carlo samples for uncertainty at test time

    @property
    def n_wavelengths(self) -> int:
        return int((self.wl_max - self.wl_min) / self.wl_resolution) + 1
