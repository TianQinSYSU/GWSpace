#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyOrbits.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 10:23:11
# ==================================
"""Space detectors' orbits, note that the orbits are in nature unit(in second)"""

import numpy as np
from csgwsim.Constants import C_SI, PI, PI_3, EarthOrbitFreq_SI, EarthEccentricity, Perihelion_Ang, AU_T


class Orbit(object):
    __slots__ = ('_get_pos', 'kappa0', 'orbit_1', 'orbit_2', 'orbit_3', 'p_0',
                 'p_12', 'p_23', 'p_31', 'Uni_vec_12', 'Uni_vec_13', 'Uni_vec_23')
    armLength = None
    f_0 = None  # orbital frequency
    ecc = 0.
    beta0 = 0.  # angle measured from the \tilde{x} axis to the perigee of the spacecraft orbit

    @property
    def L_T(self):
        """Arm-length in second"""
        return self.armLength/C_SI

    @property
    def R_T(self):
        """semi-major axis of the spacecraft orbit (in second)"""
        return self.armLength/(C_SI*np.sqrt(3))


class TianQinOrbit(Orbit):
    """See Hu et al. https://iopscience.iop.org/article/10.1088/1361-6382/aab52f"""
    __slots__ = 'kappa_earth'
    armLength = np.sqrt(3)*1.0e8
    f_0 = 3.176e-6  # orbital frequency of the TianQin satellites around the Earth TODO: convert it into a property
    # ecliptic lon & lat of J0806.3+1527
    theta_s = -4.7/180*PI
    phi_s = 120.5/180*PI

    def __init__(self, time, kappa0=0., kappa_earth=0.):
        # TODO: here we do not store the time series into the class
        Orbit.__init__(self)
        self.kappa0 = kappa0  # initial orbit phase of the first(n=1) spacecraft measured from \tilde{x} axis
        self.kappa_earth = kappa_earth  # the longitude measured from the vernal equinox at t = 0

        # Spacecraft orbit phase of the nth TQ satellites
        alp_t1 = self.alpha_detector(time, n=1)
        alp_t2 = self.alpha_detector(time, n=2)
        alp_t3 = self.alpha_detector(time, n=3)

        # 3D coordinate of each spacecraft vector (SSB)
        self.p_0 = self.earth_orbit_xyz(time)
        self.orbit_1 = self.p_0+self.detector_orbit_xyz(alp_t1)
        self.orbit_2 = self.p_0+self.detector_orbit_xyz(alp_t2)
        self.orbit_3 = self.p_0+self.detector_orbit_xyz(alp_t3)

        # For TDI response
        self.p_12 = self.orbit_1+self.orbit_2
        self.p_23 = self.orbit_2+self.orbit_3
        self.p_31 = self.orbit_3+self.orbit_1

        # The unit vector between three spacecrafts
        self.Uni_vec_12 = (self.orbit_2-self.orbit_1)/self.L_T
        self.Uni_vec_13 = (self.orbit_3-self.orbit_1)/self.L_T
        self.Uni_vec_23 = (self.orbit_3-self.orbit_2)/self.L_T
        # self.Uni_vec_21 = -self.Uni_vec_12
        # self.Uni_vec_31 = -self.Uni_vec_13
        # self.Uni_vec_32 = -self.Uni_vec_23

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

        x = AU_T * (csa + 0.5*EarthEccentricity * (csa2-3) - 3/4*EarthEccentricity**2 * csa * (1-csa2))
        y = AU_T * (sia + 0.5*EarthEccentricity * sia2 + 1/4*EarthEccentricity**2 * sia * (3*csa2-1))
        z = np.zeros(len(csa))
        return np.array([x, y, z])

    # def p_0(self, time):
    #     """TQ constellation center: (TQ1+TQ2+TQ3)/3 (Earth barycenter)"""
    #     return self.earth_orbit_xyz(time)

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
    armLength = 25*1e8  # Arm-length (changed from 5e9 to 2.5e9 after 2017)
    f_0 = EarthOrbitFreq_SI  # orbital frequency of LISA

    def __init__(self, time, kappa0=0):
        Orbit.__init__(self)
        self.kappa0 = kappa0  # the initial azimuthal position of the guiding center in the ecliptic plane

        # 3D coordinate of each spacecraft vector (SSB)
        self.orbit_1 = self.detector_orbit_xyz(time, n=1)
        self.orbit_2 = self.detector_orbit_xyz(time, n=2)
        self.orbit_3 = self.detector_orbit_xyz(time, n=3)

        # For TDI response
        self.p_12 = self.orbit_1+self.orbit_2
        self.p_23 = self.orbit_2+self.orbit_3
        self.p_31 = self.orbit_3+self.orbit_1

        # The unit vector between three spacecrafts
        self.Uni_vec_12 = (self.orbit_2-self.orbit_1)/self.L_T
        self.Uni_vec_13 = (self.orbit_3-self.orbit_1)/self.L_T
        self.Uni_vec_23 = (self.orbit_3-self.orbit_2)/self.L_T
        # self.Uni_vec_21 = -self.Uni_vec_12
        # self.Uni_vec_31 = -self.Uni_vec_13
        # self.Uni_vec_32 = -self.Uni_vec_23

    @property
    def ecc(self):
        # ecc_lisa = 0.004824185218078991  # Eccentricity
        return self.L_T/(2 * 3**0.5 * AU_T)

    @property
    def p_0(self):
        return (self.orbit_1+self.orbit_2+self.orbit_3) / 3

    def alpha_detector(self, time):
        return 2*PI * self.f_0 * time + self.kappa0

    def detector_orbit_xyz(self, time, n):
        beta = 2*PI_3*(n-1) + self.beta0
        snb, csb = np.sin(beta), np.cos(beta)
        alpha_lisa = self.alpha_detector(time)
        sin_alp_t, cos_alp_t = np.sin(alpha_lisa), np.cos(alpha_lisa)
        ecc = self.ecc

        x = AU_T * (cos_alp_t + ecc*(sin_alp_t*cos_alp_t*snb - (1+sin_alp_t**2)*csb))
        y = AU_T * (sin_alp_t + ecc*(sin_alp_t*cos_alp_t*csb - (1+cos_alp_t**2)*snb))
        z = -AU_T * np.sqrt(3) * ecc * np.cos(alpha_lisa - beta)
        return np.array([x, y, z])


detectors = {'TQ': TianQinOrbit,
             'LISA': LISAOrbit}
