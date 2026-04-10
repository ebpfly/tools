"""
ELBO loss for the Hyperspectral Neural Process.

Loss = E_q[ -log p(y|x,z) ] + β · KL[ q(z|context,target) ‖ p(z|context) ]

The KL term is what makes uncertainty work:
  • When context is rich, prior ≈ posterior → KL small → model is confident.
  • When context is sparse, prior is wide → KL large → model is uncertain.

β is annealed from 0→1 over the first `kl_anneal_steps` training steps so the
decoder learns meaningful reconstructions before the latent path kicks in.
"""

import math
import torch


def gaussian_nll(pred_mean, pred_logvar, target):
    """Per-element negative log-likelihood under a Gaussian."""
    var = pred_logvar.exp()
    return 0.5 * (pred_logvar + (target - pred_mean) ** 2 / var + math.log(2 * math.pi))


def kl_divergence(mu_q, lv_q, mu_p, lv_p):
    """KL( N(mu_q, sigma_q²) ‖ N(mu_p, sigma_p²) ), summed over latent dims."""
    var_q = lv_q.exp()
    var_p = lv_p.exp()
    kl = 0.5 * (lv_p - lv_q + var_q / var_p + (mu_p - mu_q) ** 2 / var_p - 1.0)
    return kl.sum(-1)  # sum over latent dims → (B,)


def elbo_loss(output, target_radiance, kl_weight=1.0):
    """
    Compute the ELBO loss.

    Args:
        output: dict from model.forward() containing
            mean, logvar, prior_mu, prior_logvar, post_mu, post_logvar
        target_radiance: (B, N_q) ground truth
        kl_weight: β for KL annealing (0 → 1 over training)

    Returns:
        total_loss, dict of component losses for logging
    """
    nll = gaussian_nll(output["mean"], output["logvar"], target_radiance)
    nll = nll.mean()  # average over batch and query wavelengths

    kl = kl_divergence(
        output["post_mu"], output["post_logvar"],
        output["prior_mu"], output["prior_logvar"],
    ).mean()  # average over batch

    total = nll + kl_weight * kl

    return total, {"nll": nll.item(), "kl": kl.item(), "kl_weight": kl_weight}
