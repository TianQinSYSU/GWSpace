import unittest

import numpy as np

from gwspace import libFastGB, pyIMRPhenomD
from gwspace.Orbit import TianQinOrbit
from gwspace.Waveform import BasicWaveform
from gwspace.response import tdi_XYZ2AET


class CoreTests(unittest.TestCase):
    def test_compiled_extensions_are_available(self):
        self.assertTrue(callable(libFastGB.Orbits))
        self.assertTrue(callable(pyIMRPhenomD.IMRPhenomD))

    def test_tianqin_arm_lengths(self):
        orbit = TianQinOrbit(np.array([0.0, 12345.0, 86400.0]))

        for first, second in ((0, 1), (1, 2), (2, 0)):
            arm_lengths = np.linalg.norm(
                orbit.orbits[second] - orbit.orbits[first], axis=0
            )
            np.testing.assert_allclose(arm_lengths, orbit.L_T, rtol=1e-12)

    def test_wave_propagation_direction_and_polarization(self):
        longitude = 0.3
        latitude = 0.4
        waveform = BasicWaveform(
            1.0, 1.0, 1.0, Lambda=longitude, Beta=latitude, psi=0.2
        )
        expected = -np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ]
        )

        np.testing.assert_allclose(waveform.vec_k, expected, atol=1e-15)
        p_plus, p_cross = waveform.polarization()
        np.testing.assert_allclose(p_plus @ waveform.vec_k, 0.0, atol=1e-15)
        np.testing.assert_allclose(p_cross @ waveform.vec_k, 0.0, atol=1e-15)

    def test_xyz_to_aet(self):
        X = np.array([1.0, 2.0])
        Y = np.array([3.0, 4.0])
        Z = np.array([5.0, 6.0])

        A, E, T = tdi_XYZ2AET(X, Y, Z)

        np.testing.assert_allclose(A, (Z - X) / np.sqrt(2.0))
        np.testing.assert_allclose(E, (X - 2.0 * Y + Z) / np.sqrt(6.0))
        np.testing.assert_allclose(T, (X + Y + Z) / np.sqrt(3.0))


if __name__ == "__main__":
    unittest.main()
