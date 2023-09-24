#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Orbit.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-08-01 10:23:11
# ==================================
"""Space detectors' orbits, note that the orbits are in nature unit(in second)"""

import numpy as np
from gwspace.Constants import C_SI, PI, PI_3, EarthOrbitFreq_SI, EarthEcc, Perihelion_Ang, AU_T, G_SI, EarthMass

if __package__ or "." in __name__:
    from gwspace import libFastGB
else:
    import libFastGB


class Orbit(object):
    __slots__ = ('kappa0', 'kappa_earth', 'orbits')
    armLength = None
    beta0 = 0.  # angle measured from the \tilde{x} axis to the perigee of the spacecraft orbit

    def __init__(self, kappa_earth=0.):
        self.kappa_earth = kappa_earth  # the longitude measured from the vernal equinox at t = 0

    @property
    def L_T(self):
        """Arm-length in second"""
        return self.armLength/C_SI

    @property
    def R_T(self):
        """semi-major axis of the spacecraft orbit (in second)"""
        return self.armLength/(C_SI*np.sqrt(3))

    @property
    def f_0(self):
        """orbital frequency"""
        raise NotImplementedError("Subclasses should provide a property to calculate orbital frequency")

    @property
    def ecc(self):
        """Eccentricity of the spacecraft."""
        return 0.

    @property
    def p_0(self):
        return np.sum(self.orbits, axis=0) / 3

    def uni_vec_ij(self, i, j):
        """The unit vector between three spacecrafts:
         Here we define uni_vec_12 = (orbit_2-orbit_1)/self.L_T"""
        return (self.orbits[j-1]-self.orbits[i-1])/self.L_T


class TianQinOrbit(Orbit):
    """See Hu et al. https://iopscience.iop.org/article/10.1088/1361-6382/aab52f"""
    __slots__ = '_p_0'
    armLength = np.sqrt(3)*1.0e8
    # ecliptic lon & lat of J0806.3+1527
    theta_s = -4.7/180*PI
    phi_s = 120.5/180*PI

    def __init__(self, time, kappa_earth=0., kappa0=0.):
        # Here we do not store the time series into the class
        Orbit.__init__(self, kappa_earth)
        self.kappa0 = kappa0  # initial orbit phase of the first(n=1) spacecraft measured from \tilde{x} axis

        # Spacecraft orbit phase of the nth TQ satellites
        alp_t1 = self.alpha_detector(time, n=1)
        alp_t2 = self.alpha_detector(time, n=2)
        alp_t3 = self.alpha_detector(time, n=3)

        # 3D coordinate of each spacecraft vector (SSB)
        self._p_0 = self.earth_orbit_xyz(time)
        self.orbits = (self._p_0+self.detector_orbit_xyz(alp_t1),
                       self._p_0+self.detector_orbit_xyz(alp_t2),
                       self._p_0+self.detector_orbit_xyz(alp_t3))

    @property
    def f_0(self):
        """orbital frequency of the TianQin satellites around the Earth (~ 3.18e-6 Hz)"""
        return np.sqrt(G_SI*EarthMass/(self.R_T*C_SI)**3)/(2*PI)

    def alpha_detector(self, time, n):
        """The orbit phase of the n-th spacecraft in the detector plane."""
        kappa_n = 2*PI_3*(n-1) + self.kappa0
        return 2*PI * self.f_0 * time + kappa_n - self.beta0

    def alpha_earth(self, time):
        """the mean orbital ecliptic longitude of the geocenter in the heliocentric-elliptic coordinate system."""
        return 2*PI*EarthOrbitFreq_SI * time + self.kappa_earth - Perihelion_Ang

    def earth_orbit_xyz(self, time):
        alpha_earth = self.alpha_earth(time)
        csa, sia = np.cos(alpha_earth), np.sin(alpha_earth)
        csa2, sia2 = np.cos(alpha_earth*2), np.sin(alpha_earth*2)

        x = AU_T * (csa + 0.5*EarthEcc * (csa2-3) - 3/4*EarthEcc**2 * csa * (1-csa2))
        y = AU_T * (sia + 0.5*EarthEcc * sia2 + 1/4*EarthEcc**2 * sia * (3*csa2-1))
        z = np.zeros(len(csa))
        return np.array([x, y, z])

    @property
    def p_0(self):
        """TQ constellation center: (TQ1+TQ2+TQ3)/3 (Earth barycenter)"""
        return self._p_0

    def detector_orbit_xyz(self, alp_t):
        sin_alp_t, cos_alp_t, cos_alp_2_t = np.sin(alp_t), np.cos(alp_t), np.cos(2*alp_t)
        sn_ps, cs_ps = np.sin(self.phi_s), np.cos(self.phi_s)
        sn_ts, cs_ts = np.sin(self.theta_s), np.cos(self.theta_s)
        R_T = self.R_T

        x = R_T * (cs_ps*sn_ts*sin_alp_t + cos_alp_t*sn_ps)
        y = R_T * (sn_ps*sn_ts*sin_alp_t - cos_alp_t*cs_ps)
        z = -R_T * sin_alp_t * cs_ts

        if self.ecc:
            x += R_T*(self.ecc*(0.5*(cos_alp_2_t-3)*sn_ps + cos_alp_t*cs_ps*sn_ts*sin_alp_t)
                      + self.ecc**2/4*sin_alp_t*((3*cos_alp_2_t-1)*cs_ps*sn_ts-6*cos_alp_t*sin_alp_t*sn_ps))
            y += R_T*(self.ecc*(-0.5*(cos_alp_2_t-3)*cs_ps + cos_alp_t*sn_ps*sn_ts*sin_alp_t)
                      + self.ecc**2/4*sin_alp_t*((3*cos_alp_2_t-1)*sn_ps*sn_ts+6*cos_alp_t*sin_alp_t*cs_ps))
            z += R_T*(-self.ecc * cos_alp_t * sin_alp_t * cs_ts
                      - self.ecc**2/4 * (3*cos_alp_2_t-1) * sin_alp_t * cs_ts)
        return np.array([x, y, z])


class LISAOrbit(Orbit):
    """See "LDC-manual-002.pdf" (Eq. 48-52)"""
    armLength = 2.5e9  # Arm-length (changed from 5e9 to 2.5e9 after 2017)

    def __init__(self, time, kappa_earth=0):
        Orbit.__init__(self, kappa_earth)

        # 3D coordinate of each spacecraft vector (SSB)
        self.orbits = (self.detector_orbit_xyz(time, n=1),
                       self.detector_orbit_xyz(time, n=2),
                       self.detector_orbit_xyz(time, n=3))

    @property
    def ecc(self):
        # ecc_lisa = 0.004824185218078991  # Eccentricity
        return self.L_T/(2 * 3**0.5 * AU_T)

    @property
    def f_0(self):
        """orbital frequency of LISA"""
        return EarthOrbitFreq_SI

    @property
    def kappa0(self):
        """the initial azimuthal position of the guiding center in the ecliptic plane"""
        return self.kappa_earth - Perihelion_Ang - 20/180*PI

    def alpha_detector(self, time):
        return 2*PI * self.f_0 * time + self.kappa0

    def detector_orbit_xyz(self, time, n):  # TODO: add 2nd order
        beta = 2*PI_3*(n-1) + self.beta0
        snb, csb = np.sin(beta), np.cos(beta)
        alpha_lisa = self.alpha_detector(time)
        sin_alp_t, cos_alp_t = np.sin(alpha_lisa), np.cos(alpha_lisa)
        ecc = self.ecc

        x = AU_T * (cos_alp_t + ecc*(sin_alp_t*cos_alp_t*snb - (1+sin_alp_t**2)*csb))
        y = AU_T * (sin_alp_t + ecc*(sin_alp_t*cos_alp_t*csb - (1+cos_alp_t**2)*snb))
        z = -AU_T * np.sqrt(3) * ecc * np.cos(alpha_lisa - beta)
        return np.array([x, y, z])


class TaijiOrbit(LISAOrbit):
    armLength = 3e9

    @property
    def kappa0(self):
        """the initial azimuthal position of the guiding center in the ecliptic plane"""
        return self.kappa_earth - Perihelion_Ang + 20/180*PI


def get_pos(tf, detector="TianQin", toT=True):
    """
    Calculate the orbit position with C code
    ----------------------------------------
    Parameters:
    - tf: array of time
    - detector: string of detector's name
    - toT: bool parameter to determine return
        in length unit or time unit
    ----------------------------------------
    Return:
    - (x,y,z,LT)
    ---> x,y,z: [0],[1],[2] for three spacecrafts' position
    ---> LT = armLength/C_SI
    """
    N = tf.shape[0]
    x = np.zeros(3*N, 'd')
    y = np.zeros(3*N, 'd')
    z = np.zeros(3*N, 'd')
    L = np.zeros(1,  'd')

    libFastGB.Orbits(detector, N, tf, x, y, z, L)

    x, y, z, L = x.copy(), y.copy(), z.copy(), L.copy()

    xr = x.reshape((N, 3)).T
    yr = y.reshape((N, 3)).T
    zr = z.reshape((N, 3)).T

    if toT:
        return xr/C_SI, yr/C_SI, zr/C_SI, L[0]/C_SI
    return xr, yr, zr, L[0]


detectors = {'TQ': TianQinOrbit,
             'LISA': LISAOrbit,
             'Taiji': TaijiOrbit,
             'TianQin': TianQinOrbit,
             }
