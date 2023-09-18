#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Noise.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2022-11-04 11:03:09
# ==================================

import numpy as np
from csgwsim.Constants import C_SI, PI
from scipy import interpolate


class TianQinNoise(object):
    # Sa = 1e-30  # test mass noise
    # Sx = 1e-24  # Position / residual Acceleration sensitivity goals shot noise

    def __init__(self, Na=1e-30, Np=1e-24, armL=1.7e8):
        self.Na = Na
        self.Np = Np
        self.armL = armL
        self.LT = self.armL/C_SI

    def noises(self, freq, unit="relativeFrequency"):
        """
        Acceleration noise & Optical Metrology System
        Sp = self.Np / (2 * self.armL)**2 * np.ones_like(freq)
        Sa = self.Na *(1+1e-4/freq) / (2 * PI * freq)**4 / (2 * self.armL)**2
        """
        omega = 2*PI*freq
        # In acceleration
        Sa_a = self.Na  # * (1. + 0.1e-3/freq ) # without the tail of freq

        # In displacement
        Sa_d = Sa_a/omega**4
        Soms_d = self.Np*np.ones_like(freq)

        if unit == "displacement":
            return Sa_d, Soms_d
        elif unit == "relativeFrequency":
            # In Relative frequency unit
            Sa_nu = Sa_d*(omega/C_SI)**2
            Soms_nu = Soms_d*(omega/C_SI)**2
            return Sa_nu, Soms_nu  # Spm, Sop
        else:
            print(f"No such unit of {self.unit}")

    def sensitivity(self, freq):
        Sa, Sp = self.noises(freq, unit="displacement")
        f_star = C_SI/(2*PI*self.armL)
        sens = (2*(1+np.cos(freq/f_star))*Sa*(1+1e-4/freq)+Sp)
        tmp = (1+(freq/0.41/C_SI*2*self.armL)**2)
        return 10./3/self.armL**2*sens*tmp


class LISANoise(object):
    """
    For LISA noise
    the model is SciRDv1
    """

    # f_star_lisa = C_SI/(2*PI*L_lisa)
    # # # This can be converted to strain spectral density by dividing by the path-length squared:
    # # S_shot = 1.21e-22  # m^2/Hz, 1.1e-11**2
    # # S_s_lisa = S_shot / L_lisa**2
    # # Each inertial sensor is expected to contribute an acceleration noise with spectral density
    # Sacc = 9e-30  # m^2 s^-4 /Hz, 3e-15**2
    # # The single-link optical metrology noise is quoted as:
    # Smos = 2.25e-22  # m^2/Hz, 1.5e-11**2

    def __init__(self, Na=3e-15**2, Np=15.0e-12**2, armL=2.5e9):
        self.Na = Na
        self.Np = Np
        self.armL = armL
        self.LT = self.armL/C_SI

    def noises(self, freq, unit="relativeFrequency"):
        """
        Acceleration noise & Optical Metrology System
        """
        # In acceleration
        Sa_a = self.Na*(1.+(0.4e-3/freq)**2)*(1+(freq/8e-3)**4)

        # In displacement
        Sa_d = Sa_a/(2*PI*freq)**4
        Soms_d = self.Np*(1+(2e-3/freq)**4)

        if unit == "displacement":
            return Sa_d, Soms_d
        elif unit == "relativeFrequency":
            # In Relative frequency unit
            Sa_nu = Sa_d*(2*PI*freq/C_SI)**2
            Soms_nu = Soms_d*(2*PI*freq/C_SI)**2
            return Sa_nu, Soms_nu  # Spm, Sop
        else:
            print(f"No such unit of {self.unit}")

    def sensitivity(self, freq, includewd=None):
        Sa, Sp = self.noises(freq, unit="displacement")
        All_m = np.sqrt(4*Sa+Sp)

        # Average the antenna response
        AvResp = np.sqrt(5)

        # projection effect
        Proj = 2./np.sqrt(3)

        # Approximate transfer function
        f0 = 1./(2.*self.LT)
        a = 0.41
        T = np.sqrt(1+(freq/(a*f0))**2)
        sens = (AvResp*Proj*T*All_m/self.armL)**2

        if includewd is not None:
            day = 86400.0
            year = 365.25*24.0*3600.0
            if (includewd < day/year) or (includewd > 10.0):
                raise NotImplementedError
            Sgal = GalConf(freq, includewd*year)
            sens = sens+Sgal

        return sens


def SGal(fr, pars):
    """
    TODO To be described
    """
    Amp = pars[0]
    alpha = pars[1]
    sl1 = pars[2]
    kn = pars[3]
    sl2 = pars[4]
    Sgal = Amp*np.exp(-(fr**alpha)*sl1)*(fr**(-7./3.))*0.5*(1.0+np.tanh(-(fr-kn)*sl2))

    return Sgal


def GalConf(fr, Tobs):
    """
    TODO To be described
    """
    day = 86400.0
    month = day*30.5
    year = 365.25*24.0*3600.0

    Amp = 3.26651613e-44
    alpha = 1.18300266e+00

    Xobs = [1.0*day, 3.0*month, 6.0*month, 1.0*year, 2.0*year, 4.0*year, 10.0*year]
    Slope1 = [9.41315118e+02, 1.36887568e+03, 1.68729474e+03, 1.76327234e+03, 2.32678814e+03, 3.01430978e+03,
              3.74970124e+03]
    knee = [1.15120924e-02, 4.01884128e-03, 3.47302482e-03, 2.77606177e-03, 2.41178384e-03, 2.09278117e-03,
            1.57362626e-03]
    Slope2 = [1.03239773e+02, 1.03351646e+03, 1.62204855e+03, 1.68631844e+03, 2.06821665e+03, 2.95774596e+03,
              3.15199454e+03]

    Tmax = 10.0*year
    if Tobs > Tmax:
        print('I do not do extrapolation, Tobs > Tmax:', Tobs, Tmax)
        sys.exit(1)

    # Interpolate
    tck1 = interpolate.splrep(Xobs, Slope1, s=0, k=1)
    tck2 = interpolate.splrep(Xobs, knee, s=0, k=1)
    tck3 = interpolate.splrep(Xobs, Slope2, s=0, k=1)
    sl1 = interpolate.splev(Tobs, tck1, der=0)
    kn = interpolate.splev(Tobs, tck2, der=0)
    sl2 = interpolate.splev(Tobs, tck3, der=0)
    # print "interpolated values: slope1, knee, slope2", sl1, kn, sl2
    Sgal_int = SGal(fr, [Amp, alpha, sl1, kn, sl2])

    return Sgal_int


def WDconfusionX(f, armLT, duration):
    """
    TODO To be described
    """
    # duration is assumed to be in years
    day = 86400.0
    year = 365.25*24.0*3600.0
    if (duration < day/year) or (duration > 10.0):
        raise NotImplementedError

    x = 2.0*PI*armLT*f
    t = 4.0*x**2*np.sin(x)**2
    Sg_sens = GalConf(f, duration*year)
    # t = 4 * x**2 * np.sin(x)**2 * (1.0 if obs == 'X' else 1.5)
    return t*Sg_sens


def WDconfusionAE(f, armLT, duration):
    """
    TODO To be described
    """
    SgX = WDconfusionX(f, armLT, duration)
    return 1.5*SgX


def noise_XYZ(freq, Sa, Sp, armL, includewd=None):
    u = freq*(2*PI*armL/C_SI)
    cu = np.cos(u)
    su = np.sin(u)
    su2 = su*su
    sx = 16*su2*(2*(1+cu*cu)*Sa+Sp)
    sxy = -8*su2*cu*(Sp+4*Sa)
    if includewd is not None:
        sx += WDconfusionX(freq, armL/C_SI, includewd)
    return sx, sxy


def noise_AET(freq, Sa, Sp, armL, includewd=None):
    u = freq*(2*PI*armL/C_SI)
    cu = np.cos(u)
    su = np.sin(u)
    su2 = su*su
    ae = 8*su2*((2+cu)*Sp+4*(1+cu+cu**2)*Sa)
    tt = 16*su2*(1-cu)*(Sp+2*(1-cu)*Sa)
    if includewd is not None:
        ae += WDconfusionAE(freq, armL/C_SI, includewd)
    return ae, ae, tt


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
