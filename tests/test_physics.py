from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dispersion_demo.physics import (  # noqa: E402
    gaussian_envelope,
    gaussian_intensity,
    gaussian_pulse_train,
    group_velocity,
    intensity_energy,
    phase_velocity,
    rms_width,
)


class PhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time = np.linspace(-12.0, 12.0, 100_001)

    def test_group_and_phase_velocity_are_visibly_distinct(self) -> None:
        self.assertLess(group_velocity(), phase_velocity(5.5))

    def test_dispersion_conserves_pulse_energy(self) -> None:
        initial = gaussian_intensity(self.time, dispersion_strength=0.0)
        dispersed = gaussian_intensity(self.time, dispersion_strength=3.1)
        self.assertAlmostEqual(
            intensity_energy(self.time, initial),
            intensity_energy(self.time, dispersed),
            places=6,
        )

    def test_dispersion_increases_width_and_lowers_peak(self) -> None:
        initial = gaussian_intensity(self.time, dispersion_strength=0.0)
        dispersed = gaussian_intensity(self.time, dispersion_strength=3.1)
        self.assertGreater(rms_width(self.time, dispersed), 3.0 * rms_width(self.time, initial))
        self.assertLess(float(np.max(dispersed)), 0.35 * float(np.max(initial)))

    def test_complex_envelope_matches_intensity_formula(self) -> None:
        strength = 1.7
        envelope = gaussian_envelope(self.time, dispersion_strength=strength)
        intensity = gaussian_intensity(self.time, dispersion_strength=strength)
        np.testing.assert_allclose(np.abs(envelope) ** 2, intensity, rtol=1e-12, atol=1e-12)

    def test_dispersed_ook_neighbors_raise_zero_symbol_sample(self) -> None:
        symbols = (1, 0, 1)
        centers = (-1.8, 0.0, 1.8)
        components, received = gaussian_pulse_train(
            self.time,
            symbols,
            centers,
            tau0=0.4,
            dispersion_strength=4.8,
        )
        zero_sample = float(np.interp(0.0, self.time, received))
        self.assertGreater(zero_sample, 0.1)
        np.testing.assert_allclose(np.sum(components, axis=0), received)

    def test_ook_pulse_train_rejects_invalid_symbols(self) -> None:
        with self.assertRaises(ValueError):
            gaussian_pulse_train(self.time, (1, -1), (-1.0, 1.0), 0.4, 1.0)


if __name__ == "__main__":
    unittest.main()
