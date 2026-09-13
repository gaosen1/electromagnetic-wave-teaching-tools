from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dispersion_demo.pcb_physics import (  # noqa: E402
    eye_traces,
    pam4_transmit_waveform,
    pam4_waveform,
    pcb_channel,
    pcb_group_delay,
)
from dispersion_demo.radar_physics import (  # noqa: E402
    compressed_lfm_profile,
    plasma_group_velocity,
)
from dispersion_demo.rendering import formula_runs  # noqa: E402


def fwhm(x: np.ndarray, values: np.ndarray) -> float:
    indices = np.flatnonzero(values >= 0.5 * np.max(values))
    return float(x[indices[-1]] - x[indices[0]])


class RadarPhysicsTests(unittest.TestCase):
    def test_higher_frequency_has_higher_group_velocity(self) -> None:
        self.assertLess(plasma_group_velocity(1.35), plasma_group_velocity(2.7))
        self.assertLess(plasma_group_velocity(2.7), 1.0)

    def test_phase_dispersion_broadens_matched_filter_peak_without_losing_energy(self) -> None:
        delay, ideal = compressed_lfm_profile(0.0)
        _, dispersed = compressed_lfm_profile(60.0)
        self.assertGreater(fwhm(delay, dispersed), 4.0 * fwhm(delay, ideal))
        self.assertLess(float(np.max(dispersed)), 0.25 * float(np.max(ideal)))
        self.assertAlmostEqual(
            float(np.trapezoid(ideal, delay)),
            float(np.trapezoid(dispersed, delay)),
            places=6,
        )


class PcbPhysicsTests(unittest.TestCase):
    def test_quadratic_phase_channel_is_energy_preserving_without_loss(self) -> None:
        signal = pam4_waveform()
        received = pcb_channel(signal, dispersion_strength=4.0, loss_strength=0.0)
        self.assertAlmostEqual(float(np.sum(np.square(signal))), float(np.sum(np.square(received))), places=8)

    def test_group_delay_increases_with_frequency(self) -> None:
        frequency = np.array((0.1, 0.4, 0.8))
        delay = np.asarray(pcb_group_delay(frequency, 2.0))
        self.assertTrue(np.all(np.diff(delay) > 0.0))

    def test_bandlimited_pam4_transmitter_keeps_three_open_eyes(self) -> None:
        rng = np.random.default_rng(224)
        symbols = rng.choice(np.array((-3.0, -1.0, 1.0, 3.0)), size=256)
        waveform = pam4_transmit_waveform(symbols, samples_per_symbol=64, bandwidth_per_baud=0.6)
        traces = eye_traces(waveform, samples_per_symbol=64)
        center_samples = np.asarray([trace[64] for trace in traces])
        center_symbols = symbols[1:-2]
        ranges = []
        for level in (-3.0, -1.0, 1.0, 3.0):
            samples = center_samples[center_symbols == level]
            ranges.append((float(np.min(samples)), float(np.max(samples))))
        openings = [ranges[index + 1][0] - ranges[index][1] for index in range(3)]
        self.assertGreater(min(openings), 1.45)
        self.assertFalse(np.array_equal(waveform, pam4_waveform(symbols, 64)))

    def test_severe_pcb_channel_closes_at_least_one_pam4_eye(self) -> None:
        rng = np.random.default_rng(224)
        symbols = rng.choice(np.array((-3.0, -1.0, 1.0, 3.0)), size=256)
        transmit = pam4_transmit_waveform(symbols, samples_per_symbol=64, bandwidth_per_baud=0.6)
        received = pcb_channel(transmit, dispersion_strength=4000.0, loss_strength=20.0)
        traces = eye_traces(received, samples_per_symbol=64)
        center_samples = np.asarray([trace[64] for trace in traces])
        center_symbols = symbols[1:-2]
        ranges = []
        for level in (-3.0, -1.0, 1.0, 3.0):
            samples = center_samples[center_symbols == level]
            ranges.append((float(np.min(samples)), float(np.max(samples))))
        openings = [ranges[index + 1][0] - ranges[index][1] for index in range(3)]
        self.assertLess(min(openings), 0.0)


class FormulaLayoutTests(unittest.TestCase):
    def test_formula_parser_separates_real_scripts(self) -> None:
        self.assertEqual(
            formula_runs("β(ω_{0}) = β_{1}Ω + Ω^{2}"),
            [("β(ω", 0), ("0", -1), (") = β", 0), ("1", -1), ("Ω + Ω", 0), ("2", 1)],
        )


if __name__ == "__main__":
    unittest.main()
