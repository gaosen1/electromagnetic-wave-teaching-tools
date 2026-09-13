"""Domain equations used by the dispersion animation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def dispersion_relation(k: FloatArray | float) -> FloatArray | float:
    """Dimensionless normal-dispersion relation used in the beat scene."""

    return 1.15 * k - 0.018 * np.square(k)


def two_frequency_wave(
    z: FloatArray,
    time: float,
    k1: float = 5.8,
    k2: float = 5.2,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return two close-frequency components, their sum, and signed envelope."""

    omega1 = float(dispersion_relation(k1))
    omega2 = float(dispersion_relation(k2))
    wave1 = np.cos(k1 * z - omega1 * time)
    wave2 = np.cos(k2 * z - omega2 * time)
    total = wave1 + wave2
    envelope = 2.0 * np.cos(0.5 * ((k1 - k2) * z - (omega1 - omega2) * time))
    return wave1, wave2, total, envelope


def phase_velocity(k: float) -> float:
    return float(dispersion_relation(k) / k)


def group_velocity(k1: float = 5.8, k2: float = 5.2) -> float:
    return float((dispersion_relation(k1) - dispersion_relation(k2)) / (k1 - k2))


def gaussian_intensity(
    retarded_time: FloatArray,
    tau0: float = 0.72,
    dispersion_strength: float = 0.0,
) -> FloatArray:
    """Lossless Gaussian-pulse intensity after second-order dispersion.

    ``dispersion_strength`` is beta2 * L / tau0**2. The normalization keeps
    the integral of intensity constant while the width grows.
    """

    width_factor = np.sqrt(1.0 + dispersion_strength**2)
    normalized_time = retarded_time / (tau0 * width_factor)
    return np.exp(-np.square(normalized_time)) / width_factor


def gaussian_envelope(
    retarded_time: FloatArray,
    tau0: float = 0.72,
    dispersion_strength: float = 0.0,
) -> NDArray[np.complex128]:
    """Complex Gaussian envelope with quadratic spectral phase."""

    denominator = 1.0 + 1j * dispersion_strength
    return np.exp(-np.square(retarded_time) / (2.0 * tau0**2 * denominator)) / np.sqrt(
        denominator
    )


def gaussian_pulse_train(
    time: FloatArray,
    symbols: tuple[int, ...],
    centers: tuple[float, ...],
    tau0: float,
    dispersion_strength: float,
) -> tuple[FloatArray, FloatArray]:
    """Return each OOK symbol contribution and their received intensity sum."""

    if len(symbols) != len(centers):
        raise ValueError("symbols and centers must have the same length")
    if any(symbol not in (0, 1) for symbol in symbols):
        raise ValueError("OOK symbols must be 0 or 1")

    components = np.asarray(
        [
            symbol * gaussian_intensity(time - center, tau0, dispersion_strength)
            for symbol, center in zip(symbols, centers)
        ]
    )
    return components, np.sum(components, axis=0)


def intensity_energy(time: FloatArray, intensity: FloatArray) -> float:
    return float(np.trapezoid(intensity, time))


def rms_width(time: FloatArray, intensity: FloatArray) -> float:
    energy = intensity_energy(time, intensity)
    mean = float(np.trapezoid(time * intensity, time) / energy)
    variance = float(np.trapezoid(np.square(time - mean) * intensity, time) / energy)
    return float(np.sqrt(variance))
