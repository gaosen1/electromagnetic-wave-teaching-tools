"""Normalized signal models for the wideband-radar teaching animation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def plasma_refractive_index(omega: FloatArray | float, omega_p: float = 1.0) -> FloatArray | float:
    """Cold, collisionless, unmagnetized plasma refractive index."""

    omega_array = np.asarray(omega, dtype=float)
    if np.any(omega_array <= omega_p):
        raise ValueError("omega must exceed the plasma frequency")
    result = np.sqrt(1.0 - np.square(omega_p / omega_array))
    return float(result) if np.ndim(result) == 0 else result


def plasma_group_velocity(omega: FloatArray | float, omega_p: float = 1.0) -> FloatArray | float:
    """Group velocity normalized by c for the same cold-plasma model."""

    return plasma_refractive_index(omega, omega_p)


def lfm_pulse(time: FloatArray, duration: float = 1.0, bandwidth: float = 18.0) -> ComplexArray:
    """Unit-amplitude baseband LFM pulse with a rectangular time gate."""

    gate = np.abs(time) <= duration / 2.0
    chirp_rate = bandwidth / duration
    return gate * np.exp(1j * np.pi * chirp_rate * np.square(time))


def compressed_lfm_profile(
    dispersion_strength: float,
    *,
    samples: int = 1024,
) -> tuple[FloatArray, FloatArray]:
    """Matched-filter power after a phase-only quadratic dispersive channel.

    The channel magnitude is exactly one, so degradation comes from phase
    distortion rather than attenuation.
    """

    time = np.linspace(-1.0, 1.0, samples, endpoint=False)
    transmit = lfm_pulse(time)
    frequency = np.fft.fftfreq(samples)
    phase = -dispersion_strength * np.square(frequency / 0.06)
    received = np.fft.ifft(np.fft.fft(transmit) * np.exp(1j * phase))

    fft_size = 2 * samples
    matched = np.conj(transmit[::-1])
    compressed = np.fft.ifft(np.fft.fft(received, fft_size) * np.fft.fft(matched, fft_size))
    power = np.square(np.abs(compressed))
    ideal_peak = float(np.count_nonzero(np.abs(transmit)) ** 2)
    power /= max(ideal_peak, 1.0)
    delay = (np.arange(fft_size, dtype=float) - (samples - 1)) / samples
    return delay, power.astype(float)
