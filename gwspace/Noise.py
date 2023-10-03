#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Noise.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2022-11-04 11:03:09
# ==================================

import numpy as np
from scipy import interpolate

from gwspace.constants import C_SI, PI


class TianQinNoise(object):
    Na = 1e-30  # m^2 s^-4 /Hz, Acceleration noise
    Np = 1e-24  # m^2 / Hz, Optical metrology noise
    armLength = np.sqrt(3)*1.0e8

    @property
    def L_T(self):
        """Arm-length in second"""
        return self.armLength/C_SI

    @property
    def f_star(self):
        return C_SI/(2*PI*self.armLength)

    def noises_displacement(self, freq):
        """Acceleration noise & Optical Metrology System"""
        # In acceleration
        Sa_a = self.Na * (1. + 1e-4/freq)

        # In displacement
        Sa_d = Sa_a/(2*PI*freq)**4
        Sp_d = self.Np * np.ones_like(freq)
        return Sa_d, Sp_d

    def noises_relative_freq(self, freq):
        Sa_d, Sp_d = self.noises_displacement(freq)

        # In Relative frequency unit
        Sa_nu = Sa_d*(2*PI*freq/C_SI)**2
        Sp_nu = Sp_d*(2*PI*freq/C_SI)**2
        return Sa_nu, Sp_nu

    def sensitivity(self, freq):  # TODO
        Sa_d, Sp_d = self.noises_displacement(freq)
        sens = (2*(1+np.cos(freq/self.f_star))*Sa_d + Sp_d)
        tmp = (1+(freq/0.41/C_SI*2*self.armLength)**2)
        return 10./3/self.armLength**2*sens*tmp

    def noise_XYZ(self, freq, unit="relative_frequency"):
        if unit == "relative_frequency":  # TODO
            Sa, Sp = self.noises_relative_freq(freq)
        elif unit == "displacement":
            Sa, Sp = self.noises_displacement(freq)
        else:
            raise ValueError(f"Unknown unit: {unit}. "
                             f"Supported units: {'|'.join(['displacement', 'relative_frequency'])}")
        u = 2*PI * freq * self.L_T
        s_x = 16 * np.sin(u)**2 * (2*(1+np.cos(u)**2)*Sa + Sp)
        s_xy = -8 * np.sin(u)**2 * np.cos(u) * (4*Sa + Sp)
        return s_x, s_xy

    def noise_AET(self, freq):
        # s_ae = 8 * np.sin(u)**2 * (4*(1+np.cos(u)+np.cos(u)**2)*Sa + (2+np.cos(u))*Sp)
        # s_t = 16 * np.sin(u)**2 * (1-np.cos(u)) * (2*(1-np.cos(u))*Sa + Sp)
        s_x, s_xy = self.noise_XYZ(freq)
        s_ae = s_x - s_xy
        s_t = s_x + 2*s_xy
        return s_ae, s_t


class LISANoise(TianQinNoise):
    """ For LISA noise, the model is SciRDv1 """
    Na = 9e-30  # m^2 s^-4 /Hz, 3e-15**2
    Np = 2.25e-22  # m^2/Hz, 1.5e-11**2
    armLength = 2.5e9  # Arm-length (changed from 5e9 to 2.5e9 after 2017)

    def noises_displacement(self, freq):
        """ Acceleration noise & Optical Metrology System """
        # In acceleration
        Sa_a = self.Na * (1.+(0.4e-3/freq)**2) * (1+(freq/8e-3)**4)

        # In displacement
        Sa_d = Sa_a/(2*PI*freq)**4
        Sp_d = self.Np * (1+(2e-3/freq)**4)
        return Sa_d, Sp_d

    def sensitivity(self, freq, wd_foreground=0.):  # TODO
        Sa_d, Sp_d = self.noises_displacement(freq)
        All_m = np.sqrt(4*Sa_d + Sp_d)

        # Average the antenna response
        AvResp = np.sqrt(5)

        # projection effect
        Proj = 2./np.sqrt(3)

        # Approximate transfer function
        f0 = 1./(2.*self.L_T)
        a = 0.41
        T = np.sqrt(1+(freq/(a*f0))**2)
        sens = (AvResp*Proj*T*All_m/self.armLength)**2

        if wd_foreground:
            s_gal = self._gal_conf(freq, wd_foreground)
            sens += s_gal
        return sens

    def noise_XYZ(self, freq, wd_foreground=0.):
        sx, sxy = super().noise_XYZ(freq)
        if wd_foreground:
            sx += self.wd_foreground_X(freq, wd_foreground)
        return sx, sxy

    def noise_AET(self, freq, wd_foreground=0.):
        ae, tt = super().noise_AET(freq)
        if wd_foreground:
            ae += self.wd_foreground_AE(freq, wd_foreground)
        return ae, tt

    def _gal_conf(self, f, duration):
        day = 86400.0
        month = 30.5*day
        year = 365.25*day
        if (duration < day/year) or (duration > 10.):
            raise NotImplementedError
        Tobs = duration * year

        Amp = 3.26651613e-44
        alpha = 1.18300266e+00

        Xobs = [1.0*day, 3.0*month, 6.0*month, 1.0*year, 2.0*year, 4.0*year, 10.0*year]
        Slope1 = [9.41315118e+02, 1.36887568e+03, 1.68729474e+03, 1.76327234e+03, 2.32678814e+03, 3.01430978e+03,
                  3.74970124e+03]
        knee = [1.15120924e-02, 4.01884128e-03, 3.47302482e-03, 2.77606177e-03, 2.41178384e-03, 2.09278117e-03,
                1.57362626e-03]
        Slope2 = [1.03239773e+02, 1.03351646e+03, 1.62204855e+03, 1.68631844e+03, 2.06821665e+03, 2.95774596e+03,
                  3.15199454e+03]

        tck1 = interpolate.splrep(Xobs, Slope1, k=1)
        tck2 = interpolate.splrep(Xobs, knee, k=1)
        tck3 = interpolate.splrep(Xobs, Slope2, k=1)
        sl1 = interpolate.splev(Tobs, tck1)
        kn = interpolate.splev(Tobs, tck2)
        sl2 = interpolate.splev(Tobs, tck3)
        return Amp*np.exp(-(f**alpha)*sl1)*(f**(-7./3.))*0.5*(1.0+np.tanh(-(f-kn)*sl2))

    def wd_foreground_X(self, f, duration):
        """duration: in [yr]s """
        u = 2*PI * f * self.L_T
        t = 4. * u**2 * np.sin(u)**2
        Sg_sens = self._gal_conf(f, duration)
        return t * Sg_sens

    def wd_foreground_AE(self, f, duration):
        return 1.5 * self.wd_foreground_X(f, duration)


class TaijiNoise(LISANoise):
    Na = 9e-30  # m^2 s^-4 /Hz, 3e-15**2
    Np = 6.4e-23  # m^2/Hz, 8e-12**2
    armLength = 3e9

    def _gal_conf(self, f, duration):
        raise NotImplementedError

    def wd_foreground_X(self, f, duration):
        raise NotImplementedError


detector_noises = {'TQ': TianQinNoise,
                   'LISA': LISANoise,
                   'Taiji': TaijiNoise,
                   'TianQin': TianQinNoise,
                   }


class WhiteNoise:
    """
    White noise generator
    ---------------------
    for constant powser spectrum density
    """

    def __init__(self, f_sample: float, psd: float = 1., seed=None) -> None:
        """
        Creat a White Noise instance
        ----------------------------
        Parameters:
        - f_sample: sampling frequencies in Hz.
        - psd: constant value of the two-sided power
            spectrum density
        - seed: for the random number generator
        """
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


def white_noise(fs, size, asd):
    """
    Generate a white noise
    ----------------------
    Parameters:
    - fs
    - size
    - asd
    """
    gen = WhiteNoise(fs, asd**2/2)
    return gen.get_series(size)
