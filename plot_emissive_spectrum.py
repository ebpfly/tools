"""Generate a fake mid-IR emissive spectrum from 7 to 16 micrometers.

The spectrum is built from a Planck blackbody continuum at a realistic
terrestrial brightness temperature, modulated by a handful of absorption
and emission features that are qualitatively located where real atmospheric
bands occur (water vapor, ozone ~9.6 um, CO2 ~15 um).
"""

import numpy as np
import matplotlib.pyplot as plt


def planck_radiance(wavelength_um, temperature_k):
    """Spectral radiance (W / m^2 / sr / um) from Planck's law."""
    h = 6.62607015e-34
    c = 2.99792458e8
    kb = 1.380649e-23
    wl = wavelength_um * 1e-6
    exponent = h * c / (wl * kb * temperature_k)
    radiance = (2.0 * h * c ** 2) / (wl ** 5) / (np.exp(exponent) - 1.0)
    return radiance * 1e-6  # convert from per-meter to per-micrometer


def gaussian(x, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def main():
    rng = np.random.default_rng(seed=42)

    wl = np.linspace(7.0, 16.0, 2000)
    continuum = planck_radiance(wl, temperature_k=288.0)

    # Emissivity profile: start from 1.0 and carve / add features.
    emissivity = np.ones_like(wl)
    # Broad water-vapor absorption on the short-wavelength edge.
    emissivity -= gaussian(wl, center=7.3, width=0.45, amplitude=0.35)
    # Methane-ish dip.
    emissivity -= gaussian(wl, center=7.8, width=0.20, amplitude=0.15)
    # Atmospheric window stays near unity 8-12 um, with a small ozone notch.
    emissivity -= gaussian(wl, center=9.6, width=0.25, amplitude=0.28)
    # A subtle surface emission feature (e.g. silicate reststrahlen).
    emissivity += gaussian(wl, center=11.0, width=0.35, amplitude=0.05)
    # Strong CO2 band around 15 um.
    emissivity -= gaussian(wl, center=15.0, width=0.55, amplitude=0.55)
    # Minor H2O features on the long-wavelength side.
    emissivity -= gaussian(wl, center=13.2, width=0.30, amplitude=0.10)
    emissivity = np.clip(emissivity, 0.02, 1.05)

    signal = continuum * emissivity
    # Add a little measurement noise so it does not look too synthetic.
    noise = rng.normal(0.0, 0.0025 * signal.max(), size=signal.shape)
    signal = signal + noise

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wl, signal, color="#c0392b", lw=1.3, label="Measured radiance")
    ax.plot(wl, continuum, color="#7f8c8d", lw=1.0, ls="--",
            label="288 K blackbody")

    annotations = [
        (7.3, "H$_2$O"),
        (9.6, "O$_3$"),
        (11.0, "surface"),
        (13.2, "H$_2$O"),
        (15.0, "CO$_2$"),
    ]
    ymax = signal.max()
    for center, label in annotations:
        ax.axvline(center, color="#2c3e50", lw=0.5, alpha=0.25)
        ax.text(center, ymax * 1.02, label, ha="center", va="bottom",
                fontsize=9, color="#2c3e50")

    ax.set_xlim(7, 16)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("Wavelength (\u00b5m)")
    ax.set_ylabel("Spectral radiance (W m$^{-2}$ sr$^{-1}$ \u00b5m$^{-1}$)")
    ax.set_title("Synthetic mid-infrared emissive spectrum (7\u201316 \u00b5m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out_path = "emissive_spectrum.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
