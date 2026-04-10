"""
Hyperspectral Attentive Neural Process.

Architecture overview:

    Context tokens                          Query wavelengths
    (radiance, λ, FWHM, dx, dy)            (λ₁, λ₂, ..., λ_Q)
           │                                       │
    ┌──────▼──────┐                         ┌──────▼──────┐
    │ Token Embed  │                         │ Query Embed  │
    │ (spectral +  │                         │ (Fourier     │
    │  spatial +   │                         │  features)   │
    │  type)       │                         └──────┬──────┘
    └──────┬──────┘                                 │
           │                                        │
    ┌──────▼──────┐                                 │
    │ Transformer  │ ◄── self-attention over         │
    │ Encoder      │     all context tokens          │
    └──────┬──────┘                                 │
           │                                        │
      ┌────┴────┐                                   │
      │         │                                   │
  ┌───▼───┐    │                              ┌────▼────┐
  │Latent  │    │                              │Decoder   │
  │Encoder │    └─────────────────────────────►│(cross-   │
  │(μ, σ)  │                                   │ attn +   │
  └───┬───┘                                   │ z proj)  │
      │ z ~ N(μ, σ)                            └────┬────┘
      └────────────────────────────────────────────►│
                                                    │
                                              ┌─────▼─────┐
                                              │ Output Head │
                                              │ (μ_r, σ_r) │
                                              └───────────┘
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HyperspectralConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Learnable Fourier feature encoding for continuous scalar inputs."""

    def __init__(self, n_freqs: int):
        super().__init__()
        # Initialise on a log-linear scale; made learnable so the model can
        # adapt the frequency distribution to the data.
        init = torch.linspace(0.0, math.log(1000.0), n_freqs).exp()
        self.freqs = nn.Parameter(init)

    @property
    def out_dim(self) -> int:
        return self.freqs.shape[0] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...,) or (..., 1) → (..., 2·n_freqs)."""
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        proj = x * self.freqs  # (..., n_freqs)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, N, D = query.shape
        M = key.shape[1]
        H, d = self.n_heads, self.d_head

        q = self.q(query).view(B, N, H, d).transpose(1, 2)  # (B,H,N,d)
        k = self.k(key).view(B, M, H, d).transpose(1, 2)
        v = self.v(value).view(B, M, H, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask: (B, M) — True = valid, False = padding
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, *, cross_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.has_cross = cross_attn
        if cross_attn:
            self.norm_c = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, ctx=None, self_mask=None, cross_mask=None):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, mask=self_mask)
        if self.has_cross and ctx is not None:
            h = self.norm_c(x)
            x = x + self.cross_attn(h, ctx, ctx, mask=cross_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class SpectralTokenEncoder(nn.Module):
    """Encode (radiance, wavelength, FWHM) into a d_model-dim token."""

    def __init__(self, cfg: HyperspectralConfig):
        super().__init__()
        self.cfg = cfg
        self.wl_ff = FourierFeatures(cfg.n_spectral_freqs)
        self.fwhm_ff = FourierFeatures(cfg.n_spectral_freqs // 2)

        in_dim = 1 + self.wl_ff.out_dim + self.fwhm_ff.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, radiance, wavelength, fwhm):
        """Each input: (B, N). Returns (B, N, d_model)."""
        wl_norm = (wavelength - self.cfg.wl_min) / (self.cfg.wl_max - self.cfg.wl_min)
        fwhm_norm = fwhm / (self.cfg.wl_max - self.cfg.wl_min)

        feat = torch.cat([
            radiance.unsqueeze(-1),
            self.wl_ff(wl_norm),
            self.fwhm_ff(fwhm_norm),
        ], dim=-1)
        return self.mlp(feat)


class SpatialEncoder(nn.Module):
    """Fourier positional encoding for 2-D (dx, dy) pixel offsets."""

    def __init__(self, cfg: HyperspectralConfig):
        super().__init__()
        self.cfg = cfg
        self.x_ff = FourierFeatures(cfg.n_spatial_freqs)
        self.y_ff = FourierFeatures(cfg.n_spatial_freqs)
        self.proj = nn.Linear(self.x_ff.out_dim + self.y_ff.out_dim, cfg.d_model)

    def forward(self, position):
        """position: (B, N, 2) — relative (dx, dy)."""
        p = position / self.cfg.max_spatial_extent
        feat = torch.cat([self.x_ff(p[..., 0]), self.y_ff(p[..., 1])], dim=-1)
        return self.proj(feat)


class LatentEncoder(nn.Module):
    """Attention-pool a set of tokens into a Gaussian latent distribution."""

    def __init__(self, d_model: int, d_latent: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        self.to_mu = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_latent))
        self.to_logvar = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_latent))

    def forward(self, tokens, mask=None):
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)
        pooled = self.attn(q, tokens, tokens, mask=mask).squeeze(1)  # (B, d_model)
        return self.to_mu(pooled), self.to_logvar(pooled)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HyperspectralNeuralProcess(nn.Module):
    """
    Foundation model for hyperspectral radiance reconstruction.

    Given an arbitrary-sized context set of spectral measurements — each
    described by (radiance, centre wavelength, FWHM, spatial position) — the
    model reconstructs the full high-resolution radiance at any set of query
    wavelengths together with per-wavelength uncertainty.

    The uncertainty naturally reflects the amount of context:
      • Few context samples  →  wide prior  →  high uncertainty
      • Dense context        →  tight prior →  near-perfect reconstruction

    A variational latent vector *z* captures scene-level information and can be
    extracted for downstream tasks (material ID, atmospheric retrieval, etc.)
    with uncertainty propagated through.
    """

    def __init__(self, cfg: HyperspectralConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = HyperspectralConfig()
        self.cfg = cfg

        # --- Token construction ---
        self.spectral_enc = SpectralTokenEncoder(cfg)
        self.spatial_enc = SpatialEncoder(cfg)
        # 0 = target pixel, 1 = context pixel
        self.pixel_type_emb = nn.Embedding(2, cfg.d_model)

        # --- Transformer encoder (self-attention over context) ---
        self.encoder = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.d_model)

        # --- Latent path ---
        self.prior_enc = LatentEncoder(cfg.d_model, cfg.d_latent)
        self.posterior_enc = LatentEncoder(cfg.d_model, cfg.d_latent)

        # --- Query embedding (target wavelengths) ---
        self.query_wl_ff = FourierFeatures(cfg.n_spectral_freqs)
        self.query_proj = nn.Sequential(
            nn.Linear(self.query_wl_ff.out_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # --- Decoder (cross-attention from queries to encoded context) ---
        self.latent_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.decoder = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cross_attn=True)
            for _ in range(cfg.n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.d_model)

        # --- Output heads ---
        self.head_mean = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, 1),
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, 1),
        )

        self._init_weights()

    # ---- helpers ----------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _reparameterise(mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    # ---- encode context ---------------------------------------------------

    def _encode_tokens(self, radiance, wavelength, fwhm, position, mask=None):
        """Tokenise and run through the transformer encoder."""
        tok = self.spectral_enc(radiance, wavelength, fwhm)
        tok = tok + self.spatial_enc(position)

        # Pixel-type embedding: target pixel has position (0,0)
        is_ctx = (position.abs().sum(-1) > 1e-6).long()
        tok = tok + self.pixel_type_emb(is_ctx)

        for layer in self.encoder:
            tok = layer(tok, self_mask=mask)
        return self.encoder_norm(tok)

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        ctx_radiance,       # (B, N_ctx)
        ctx_wavelength,     # (B, N_ctx)
        ctx_fwhm,           # (B, N_ctx)
        ctx_position,       # (B, N_ctx, 2)
        query_wavelengths,  # (B, N_q)
        ctx_mask=None,      # (B, N_ctx) bool — True = valid
        tgt_radiance=None,  # (B, N_q) — ground truth, training only
    ):
        """
        Returns dict with keys:
            mean, logvar           — predicted radiance distribution at query λ
            prior_mu, prior_logvar — prior latent distribution (from context)
            post_mu, post_logvar   — posterior latent distribution (training)
            z                      — sampled latent vector
        """
        # Encode context
        enc = self._encode_tokens(ctx_radiance, ctx_wavelength, ctx_fwhm,
                                  ctx_position, ctx_mask)

        # --- Latent path ---
        prior_mu, prior_lv = self.prior_enc(enc, ctx_mask)

        if tgt_radiance is not None and self.training:
            # Build "posterior" tokens: context + target observations
            B, N_q = tgt_radiance.shape
            tgt_wl = query_wavelengths
            tgt_fwhm = torch.ones_like(tgt_radiance)  # point measurement
            tgt_pos = torch.zeros(B, N_q, 2, device=tgt_radiance.device)

            all_rad = torch.cat([ctx_radiance, tgt_radiance], 1)
            all_wl = torch.cat([ctx_wavelength, tgt_wl], 1)
            all_fwhm = torch.cat([ctx_fwhm, tgt_fwhm], 1)
            all_pos = torch.cat([ctx_position, tgt_pos], 1)
            all_mask = None
            if ctx_mask is not None:
                tgt_mask = torch.ones(B, N_q, dtype=torch.bool, device=enc.device)
                all_mask = torch.cat([ctx_mask, tgt_mask], 1)

            all_enc = self._encode_tokens(all_rad, all_wl, all_fwhm, all_pos, all_mask)
            post_mu, post_lv = self.posterior_enc(all_enc, all_mask)
            z = self._reparameterise(post_mu, post_lv)
        else:
            post_mu, post_lv = prior_mu, prior_lv
            z = self._reparameterise(prior_mu, prior_lv)

        # --- Decode at query wavelengths ---
        wl_norm = (query_wavelengths - self.cfg.wl_min) / (self.cfg.wl_max - self.cfg.wl_min)
        q = self.query_proj(self.query_wl_ff(wl_norm))     # (B, N_q, d_model)
        q = q + self.latent_proj(z).unsqueeze(1)            # broadcast z

        for layer in self.decoder:
            q = layer(q, ctx=enc, cross_mask=ctx_mask)
        q = self.decoder_norm(q)

        pred_mean = self.head_mean(q).squeeze(-1)           # (B, N_q)
        pred_logvar = self.head_logvar(q).squeeze(-1)       # (B, N_q)

        return {
            "mean": pred_mean,
            "logvar": pred_logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_lv,
            "post_mu": post_mu,
            "post_logvar": post_lv,
            "z": z,
        }

    # ---- inference helpers ------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        ctx_radiance,
        ctx_wavelength,
        ctx_fwhm,
        ctx_position,
        query_wavelengths,
        ctx_mask=None,
        n_samples=None,
    ):
        """
        Monte-Carlo predictive distribution.

        Returns dict with:
            mean          — (B, N_q) expected radiance
            total_std     — (B, N_q) total uncertainty  (aleatoric + epistemic)
            aleatoric_std — (B, N_q) aleatoric component
            epistemic_std — (B, N_q) epistemic component
            latent_mu     — (B, d_latent)
            latent_std    — (B, d_latent)
        """
        was_training = self.training
        self.eval()
        S = n_samples or self.cfg.n_mc_samples

        enc = self._encode_tokens(ctx_radiance, ctx_wavelength, ctx_fwhm,
                                  ctx_position, ctx_mask)
        prior_mu, prior_lv = self.prior_enc(enc, ctx_mask)

        wl_norm = (query_wavelengths - self.cfg.wl_min) / (self.cfg.wl_max - self.cfg.wl_min)
        q_base = self.query_proj(self.query_wl_ff(wl_norm))  # (B, N_q, d)

        means, logvars = [], []
        for _ in range(S):
            z = self._reparameterise(prior_mu, prior_lv)
            q = q_base + self.latent_proj(z).unsqueeze(1)
            for layer in self.decoder:
                q = layer(q, ctx=enc, cross_mask=ctx_mask)
            q = self.decoder_norm(q)
            means.append(self.head_mean(q).squeeze(-1))
            logvars.append(self.head_logvar(q).squeeze(-1))

        means = torch.stack(means)      # (S, B, N_q)
        logvars = torch.stack(logvars)

        pred_mean = means.mean(0)
        aleatoric = logvars.exp().mean(0)
        epistemic = means.var(0)

        if was_training:
            self.train()

        return {
            "mean": pred_mean,
            "total_std": (aleatoric + epistemic).sqrt(),
            "aleatoric_std": aleatoric.sqrt(),
            "epistemic_std": epistemic.sqrt(),
            "latent_mu": prior_mu,
            "latent_std": (0.5 * prior_lv).exp(),
        }

    @torch.no_grad()
    def extract_latent(self, radiance, wavelength, fwhm, position, mask=None):
        """Extract the latent representation for downstream tasks."""
        was_training = self.training
        self.eval()
        enc = self._encode_tokens(radiance, wavelength, fwhm, position, mask)
        mu, lv = self.prior_enc(enc, mask)
        if was_training:
            self.train()
        return {"mu": mu, "logvar": lv, "std": (0.5 * lv).exp()}
