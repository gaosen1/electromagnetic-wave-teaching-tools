"""Normalized PAM-4 and PCB-channel models for the teaching animation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


PAM4_SYMBOLS = np.array((3, 1, -1, -3, -1, 3, -3, 1, 3, -1, 1, -3), dtype=float)


def pam4_waveform(symbols: FloatArray = PAM4_SYMBOLS, samples_per_symbol: int = 64) -> FloatArray:
    return np.repeat(np.asarray(symbols, dtype=float), samples_per_symbol)


def pam4_transmit_waveform(
    symbols: FloatArray = PAM4_SYMBOLS,
    samples_per_symbol: int = 64,
    bandwidth_per_baud: float = 0.6,
) -> FloatArray:
    """Apply a Gaussian transmitter bandwidth limit to an NRZ PAM-4 sequence."""

    if bandwidth_per_baud <= 0.0:
        raise ValueError("bandwidth_per_baud must be positive")
    waveform = pam4_waveform(symbols, samples_per_symbol)
    frequency_per_baud = np.fft.fftfreq(waveform.size) * samples_per_symbol
    response = np.exp(-0.5 * np.square(frequency_per_baud / bandwidth_per_baud))
    return np.fft.ifft(np.fft.fft(waveform) * response).real


def pcb_channel(
    signal: FloatArray,
    *,
    dispersion_strength: float,
    loss_strength: float = 0.0,
) -> FloatArray:
    """Apply a real, conjugate-symmetric dispersive low-pass channel."""

    frequency = np.fft.fftfreq(signal.size)
    normalized = np.abs(frequency) / 0.5
    magnitude = np.exp(-loss_strength * np.power(normalized, 1.35))
    phase = -dispersion_strength * np.sign(frequency) * np.square(normalized)
    response = magnitude * np.exp(1j * phase)
    return np.fft.ifft(np.fft.fft(signal) * response).real


def pcb_group_delay(normalized_frequency: FloatArray | float, dispersion_strength: float) -> FloatArray | float:
    """Positive normalized group delay implied by the quadratic channel phase."""

    frequency = np.asarray(normalized_frequency, dtype=float)
    result = 2.0 * dispersion_strength * np.abs(frequency)
    return float(result) if np.ndim(result) == 0 else result


def eye_traces(
    signal: FloatArray,
    samples_per_symbol: int,
    *,
    span_symbols: int = 2,
) -> list[FloatArray]:
    width = samples_per_symbol * span_symbols
    traces: list[FloatArray] = []
    half_symbol = samples_per_symbol // 2
    for symbol_start in range(samples_per_symbol, signal.size - 2 * samples_per_symbol, samples_per_symbol):
        start = symbol_start - half_symbol
        traces.append(signal[start : start + width])
    return traces
