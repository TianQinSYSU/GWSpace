#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Noise.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2022-11-04 11:03:09
# ==================================
"""Space detectors' noises. Support noise in displacement or relative frequency units,
 PSDs in different channels, sensitivity curve, etc."""

import numpy as np
import warnings
from scipy.interpolate import interp1d

from gwspace.constants import C_SI, PI


class BasicNoise(object):
    __slots__ = tuple()
    Na = None  # m^2 s^-4 /Hz, Acceleration noise
    Np = None  # m^2 / Hz, Optical metrology noise
    armLength = None

    @property
    def L_T(self):
        """Arm-length in second"""
        return self.armLength/C_SI

    @property
    def f_star(self):
        return C_SI/(2*PI*self.armLength)

    def noises_displacement(self, freq):
        """ Return Acceleration noise & Optical metrology system noise (in displacement unit).
        Need to implement this in customized subclasses. """
        raise NotImplementedError

    def noises_relative_freq(self, freq):
        """ Return acceleration noise & optical metrology system noise (in relative frequency unit). """
        Sa_d, Sp_d = self.noises_displacement(freq)

        Sa_nu = Sa_d*(2*PI*freq/C_SI)**2
        Sp_nu = Sp_d*(2*PI*freq/C_SI)**2
        return Sa_nu, Sp_nu

    def _check_noise_unit(self, freq, unit):
        if unit == "relative_frequency":
            return self.noises_relative_freq(freq)
        elif unit == "displacement":
            return self.noises_displacement(freq)
        else:
            raise ValueError(f"Unknown unit: {unit}. "
                             f"Supported units: {'|'.join(['displacement', 'relative_frequency'])}")

    def sensitivity(self, freq, wd_foreground=0.):
        """ Sensitivity curve for **1** equivalent Michelson-like detectors, note that the prefactor is 20/3,
         if consider a combined sensitivity, then we should divide it by 2, i.e. 10/3. """
        Sa_d, Sp_d = self.noises_displacement(freq)
        sens = 20/3 / self.armLength**2 * (2*(1+np.cos(freq/self.f_star)**2)*Sa_d + Sp_d)  # low freq limit
        sens *= 1 + 0.6*(freq/self.f_star)**2
        if wd_foreground:
            sens += self.confusion_noise(freq, wd_foreground)
        return sens

    def noise_XYZ(self, freq, unit="relative_frequency", TDIgen=1, wd_foreground=0.):
        Sa, Sp = self._check_noise_unit(freq, unit)
        u = 2*PI * freq * self.L_T
        s_x = 16 * np.sin(u)**2 * (2*(1+np.cos(u)**2)*Sa + Sp)
        s_xy = -8 * np.sin(u)**2 * np.cos(u) * (4*Sa + Sp)
        if wd_foreground:
            s_x += self.wd_foreground_X(freq, wd_foreground)

        if TDIgen == 1:
            return s_x, s_xy
        elif TDIgen == 2:
            fact = 4*np.sin(2*u)**2
            return s_x*fact, s_xy*fact
        else:
            raise NotImplementedError

    def noise_AET(self, freq, unit="relative_frequency", TDIgen=1, wd_foreground=0.):
        Sa, Sp = self._check_noise_unit(freq, unit)
        u = 2*PI * freq * self.L_T
        s_ae = 8 * np.sin(u)**2 * (4*(1+np.cos(u)+np.cos(u)**2)*Sa + (2+np.cos(u))*Sp)
        s_t = 16 * np.sin(u)**2 * (1-np.cos(u)) * (2*(1-np.cos(u))*Sa + Sp)
        # s_x, s_xy = self.noise_XYZ(freq, unit, TDIgen)
        # s_ae = s_x - s_xy
        # s_t = s_x + 2*s_xy
        if wd_foreground:
            s_ae += self.wd_foreground_AE(freq, wd_foreground)

        if TDIgen == 1:
            return s_ae, s_t
        elif TDIgen == 2:  # TODO: check the 2nd generation TDI!!!
            fact = 4*np.sin(2*u)**2
            return s_ae*fact, s_t*fact
        else:
            raise NotImplementedError

    def confusion_noise(self, f, duration):
        """Return the strain sensitivity curve for Galactic confusion noise."""
        warnings.warn("You are trying to call Galactic confusion noise but with it unset, 0 will be returned.")
        return 0

    def wd_foreground_X(self, f, duration):
        """duration: in [yr] """
        u = 2*PI * f * self.L_T
        t = 4. * u**2 * np.sin(u)**2  # TODO: check this!!!
        Sg_sens = self.confusion_noise(f, duration)
        return t * Sg_sens

    def wd_foreground_AE(self, f, duration):
        return 1.5 * self.wd_foreground_X(f, duration)


class TianQinNoise(BasicNoise):
    """ TianQin noise, See Luo et al. https://iopscience.iop.org/article/10.1088/0264-9381/33/3/035010 """
    Na = 1e-30  # m^2 s^-4 /Hz, Acceleration noise
    Np = 1e-24  # m^2 / Hz, Optical metrology noise
    armLength = 3**0.5 * 1.0e8

    def noises_displacement(self, freq):
        """ Return acceleration noise & optical metrology system noise (in displacement unit). """
        # In acceleration
        Sa_a = self.Na * (1. + 1e-4/freq)

        # In displacement
        Sa_d = Sa_a/(2*PI*freq)**4
        Sp_d = self.Np * np.ones_like(freq)
        return Sa_d, Sp_d

    def confusion_noise(self, f, duration):
        """See Table I in arxiv:2403.18709, valid for 0.5 mHz < f < 10 mHz.
         See also https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.063021"""
        t_obs = (0.5, 1, 2, 4, 5)
        a0 = (-18.7, -18.7, -18.7, -18.7, -18.7)
        a1 = (-1.23, -1.34, -1.39, -1.30, -1.32)
        a2 = (-0.801, -0.513, -0.610, -0.872, -0.322)
        a3 = (0.832, 0.0152, 0.577, 0.266, -1.68)
        a4 = (-1.96, -1.53, 0.00242, -5.12, -4.49)
        a5 = (3.09, 4.79, 0.578, 15.6, 21.6)
        a6 = (-2.38, -5.01, -4.39, -15.5, -22.6)
        try:
            index = t_obs.index(duration)
            coefficients = [a[index] for a in (a0, a1, a2, a3, a4, a5, a6)]
        except ValueError:
            warnings.warn(f"Input duration {duration}yr is not in {t_obs} [year(s)], interpolation will be used.")
            coefficients = [interp1d(t_obs, a, kind='cubic')(duration) for a in (a0, a1, a2, a3, a4, a5, a6)]

        sh_confusion = np.zeros_like(f)
        ind = (f >= 5e-4) & (f <= 1e-2)
        # 10/3 is the factor for sky-average, the original fit in the paper is not sky-averaged.
        sh_confusion[ind] = 20./3*np.power(10, np.sum([a_i * np.log10(f[ind]*1e3)**i
                                                       for i, a_i in enumerate(coefficients)], axis=0))**2
        # # avoid the jump of values
        # sh_confusion[(f > 3e-4) & (f < 5e-4)] = sh_confusion[(np.abs(f-5e-4)).argmin()]
        return sh_confusion


class LISANoise(BasicNoise):
    """ LISA noise, the model is SciRDv1 """
    Na = 9e-30  # m^2 s^-4 /Hz, 3e-15**2
    Np = 2.25e-22  # m^2/Hz, 1.5e-11**2
    armLength = 2.5e9  # Arm-length (changed from 5e9 to 2.5e9 after 2017)

    def noises_displacement(self, freq):
        """ Return acceleration noise & optical metrology system noise (in displacement unit). """
        # In acceleration
        Sa_a = self.Na * (1.+(0.4e-3/freq)**2) * (1+(freq/8e-3)**4)

        # In displacement
        Sa_d = Sa_a/(2*PI*freq)**4
        Sp_d = self.Np * (1+(2e-3/freq)**4)
        return Sa_d, Sp_d

    def confusion_noise(self, f, duration):
        """Return analytic fit of the strain sensitivity curve for Galactic confusion noise.
            See 2108.01167 Eq85-86 or Karnesis2021."""
        f1 = np.power(10, -0.25*np.log10(duration) - 2.7)
        fk = np.power(10, -0.27*np.log10(duration) - 2.47)
        return 0.5*1.14e-44 * f**(-7/3) * np.exp(-(f/f1)**1.8) * (1.0+np.tanh((fk-f)/0.31e-3))


class TaijiNoise(LISANoise):
    Na = 9e-30  # m^2 s^-4 /Hz, 3e-15**2
    Np = 6.4e-23  # m^2/Hz, 8e-12**2
    armLength = 3e9

    def confusion_noise(self, f, duration):
        """Eq.(6) and Table(I) in <10.1103/PhysRevD.107.064021>, valid for 0.1 mHz < f < 10 mHz"""
        t_obs = (0.5, 1, 2, 4)
        a0 = (-85.3498, -85.4336, -85.3919, -85.5448)
        a1 = (-2.64899, -2.46276, -2.69735, -3.23671)
        a2 = (-0.0699707, -0.183175, -0.749294, -1.64187)
        a3 = (-0.478447, -0.884147, -1.15302, -1.14711)
        a4 = (-0.334821, -0.427176, -0.302761, 0.0325887)
        a5 = (0.0658353, 0.128666, 0.175521, 0.187854)
        try:
            index = t_obs.index(duration)
            coefficients = [a[index] for a in (a0, a1, a2, a3, a4, a5)]
        except ValueError:
            warnings.warn(f"Input duration {duration}yr is not in {t_obs} [year(s)], interpolation will be used.")
            coefficients = [interp1d(t_obs, a, kind='cubic')(duration) for a in (a0, a1, a2, a3, a4, a5)]

        sh_confusion = np.zeros_like(f)
        ind = (f >= 1e-4) & (f <= 1e-2)
        sh_confusion[ind] = np.exp(np.sum([a_i * np.log(f[ind]*1e3)**i for i, a_i in enumerate(coefficients)], axis=0))
        return sh_confusion


detector_noises = {'TQ': TianQinNoise,
                   'LISA': LISANoise,
                   'Taiji': TaijiNoise,
                   'TianQin': TianQinNoise,
                   }


class WhiteNoise:
    """ White noise generator: for constant powser spectrum density

    :param f_sample: sampling frequencies in Hz.
    :param psd: constant value of the two-sided power spectrum density
    :param seed: for the random number generator
    """

    def __init__(self, f_sample: float, psd: float = 1., seed=None) -> None:
        self._fs = f_sample
        self._rms = np.sqrt(f_sample*psd)
        self._rng = np.random.default_rng(seed)

    @property
    def fs(self) -> float:
        """Get the sampling frequency."""
        return self._fs

    @property
    def rms(self) -> float:
        """Get the noise signal RMS value."""
        return self._rms

    @property
    def get_sample(self) -> float:
        """Retrieve a single sample."""
        return self._rng.normal(loc=0., scale=self.rms)

    def get_series(self, npts: int) -> np.ndarray:
        """Retrieve an array of npts samples."""
        return self._rng.normal(loc=0., scale=self.rms, size=npts)


if __name__ == '__main__':
    tianqin = TianQinNoise()
    re = tianqin.confusion_noise(np.arange(1e-4, 0.02, 1e-4), 4)
    taiji = TaijiNoise()
    re_tj = taiji.confusion_noise(np.arange(1e-4, 0.02, 1e-4), 4)
