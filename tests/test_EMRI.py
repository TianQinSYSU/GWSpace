#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_EMRI.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 14:13:36
# ==================================

import numpy as np

from csgwsim.Constants import YRSID_SI
from csgwsim import EMRIWaveform

if __name__ == "__main__":
    print("This is a test of loading EMRI waveform")
    # parameters
    T = 0.01  # years
    dt = 15.0  # seconds

    M = 1e6
    a = 0.1  # will be ignored in Schwarzschild waveform
    mu = 1e1
    p0 = 12.0
    e0 = 0.2
    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = 0.2  # polar spin angle
    phiK = 0.2  # azimuthal viewing angle
    qS = 0.3  # polar sky angle
    phiS = 0.3  # azimuthal viewing angle
    dist = 1.0  # distance
    Phi_phi0 = 1.0
    Phi_theta0 = 2.0
    Phi_r0 = 3.0

    emri = EMRIWaveform(M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK,
                        Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0)

    tf = np.arange(0, T*YRSID_SI, dt)

    hp, hc = emri(T, dt)

    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(tf[:2000], hp[:2000])
    plt.plot(tf[:2000], hc[:2000])

    plt.show()
