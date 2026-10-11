"""Regress the native/TD TianQin orbit and a zero-planet binary response."""
import unittest

import numpy as np

from gwspace.Orbit import TianQinOrbit, get_pos
from gwspace.Waveform import GCBWaveform
from gwspace.constants import G_SI, EarthMass, TianQinOrbitRadius_SI
from gwspace.response import get_AET_td, tdi_XYZ2AET


class TianQinFastGBOrbitTests(unittest.TestCase):
    def test_kepler_period(self):
        omega = np.sqrt(G_SI * EarthMass / TianQinOrbitRadius_SI**3)
        period = 2*np.pi / omega
        time = np.array([0., period/4, period])
        orbit = TianQinOrbit(time)
        self.assertAlmostEqual(orbit.f_0, omega/(2*np.pi), delta=1e-20)
        x, y, z, _ = get_pos(time)
        relative = np.array([x, y, z]).transpose(1, 0, 2) - orbit.p_0
        # Quarter-period vectors must be perpendicular; a full orbit closes.
        for spacecraft in relative:
            self.assertLess(abs(np.dot(spacecraft[:, 0], spacecraft[:, 1])), 1e-10)
            np.testing.assert_allclose(spacecraft[:, 2], spacecraft[:, 0], atol=1e-10)

    def test_positions_over_two_and_half_years(self):
        time = np.linspace(0., 2.5*365*86400, 1001)
        orbit = TianQinOrbit(time)
        x, y, z, length = get_pos(time)
        native = np.array([x, y, z]).transpose(1, 0, 2)
        # Check rotation about Earth separately: SSB coordinates could hide it.
        np.testing.assert_allclose(native - native.mean(axis=0),
                                   np.array(orbit.orbits) - orbit.p_0,
                                   rtol=0., atol=1e-10)
        # The independent Earth angular-frequency rounding contributes ~5 ns.
        np.testing.assert_allclose(native, orbit.orbits, rtol=0., atol=6e-9)
        self.assertAlmostEqual(length, orbit.L_T, delta=1e-15)

    def test_zero_planet_ae_against_td_fft(self):
        duration = 2.5*365*86400
        dt = 60.
        waveform = GCBWaveform(
            .55, .27, duration, phi0=.4, f0=.00622,
            fdot=7.484049960353154e-16, fddot=0., DL=.005,
            Lambda=2.1021181843, Beta=-.0821002880,
            iota=.6802563618, psi=-.2062488646)
        time = np.arange(int(duration/dt))*dt
        frequency, *xyz = waveform.get_fastgb_fd_single(dt, oversample=16)
        fast = tdi_XYZ2AET(*xyz)
        bins = np.rint(frequency*duration).astype(int)
        use = abs(frequency-waveform.f0) < 1e-5
        # Physical Fourier normalization is dt*rfft; native FastGB needs dt.
        # No phase fitting, frequency shifting or empirical normalization.
        td = get_AET_td(waveform, time, det='TQ', TDIgen=1)
        for channel, actual, samples in zip('AE', fast[:2], td[:2]):
            reference = (dt*np.fft.rfft(samples))[bins[use]]
            actual = dt*actual[use]
            error = np.linalg.norm(actual-reference)/np.linalg.norm(reference)
            overlap = np.vdot(reference, actual).real / (
                np.linalg.norm(reference)*np.linalg.norm(actual))
            # Frequency-domain TDI vs frozen TD geometry and finite sampling
            # retain small differences. Equality is deliberately not required.
            print(f'zero-planet {channel}: relative L2={error:.8g}, raw overlap={overlap:.10g}')
            with self.subTest(channel=channel):
                self.assertLess(error, .01)
                self.assertGreater(overlap, .9999)


if __name__ == '__main__':
    unittest.main()
