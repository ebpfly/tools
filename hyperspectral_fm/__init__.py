"""
Hyperspectral Foundation Model

An Attentive Neural Process for hyperspectral radiance reconstruction with
calibrated uncertainty. Supports arbitrary numbers of spectral measurements,
FWHM-aware sensor modeling, and spatial context from neighboring pixels.
"""

from .config import HyperspectralConfig
from .model import HyperspectralNeuralProcess

__all__ = ["HyperspectralConfig", "HyperspectralNeuralProcess"]
