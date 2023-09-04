#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyINIDetectors.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 14:03:17
# ==================================

import numpy as np
from Constants import *


class INITianQin:

    def __init__(self):
        self.detector = "TianQin"
        self.armLength = np.sqrt(3)*1.0e8
        self.radius = 1.0e8
        self.Omega = 2*PI/(3.65*DAY)
        self.kappa_e = 0.0

        self.ecc = 0.0
        self.perigee = 0.0
        self.kappa_0 = 0

        # direction of orbits J0806
        self.beta_s = -4.7/180*PI
        self.theta_s = 120.5/180*PI

        self.Sa = 1e-30  # test mass noise
        self.Sx = 1e-24  # Position / residual Acceleration sensitivity goals shot noise


class INILISA:

    def __init__(self):
        self.detector = "LISA"
        # LISA
        self.armLength = 25*1e8  # Arm-length (changed from 5e9 to 2.5e9 after 2017)
        self.radius  # FIXME
        # ecc_lisa = 0.004824185218078991  # Eccentricity
        self.ecc = self.armLength/(2*np.sqrt(3)*AU_SI)
        self.kappa_0 = 0

        # f_star_lisa = c / (2*pi*L_lisa)
        # self.radiL_lisa_s = L_lisa / c

        # # This can be converted to strain spectral density by dividing by the path-length squared:
        # S_shot = 1.21e-22  # m^2/Hz, 1.1e-11**2
        # S_s_lisa = S_shot / L_lisa**2
        # Each inertial sensor is expected to contribute an acceleration noise with spectral density
        self.Sacc = 9e-30  # m^2 s^-4 /Hz, 3e-15**2
        # The single-link optical metrology noise is quoted as:
        self.Smos = 2.25e-22  # m^2/Hz, 1.5e-11**2


if __name__ == '__main__':
    print("This is initial pars for detectors")
    tq = INITianQin()
    print(f"Armlength for TianQin is: {tq.armLength}")
