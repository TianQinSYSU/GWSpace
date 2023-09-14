#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_BHB.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 14:02:42
# ==================================

import numpy as np

from csgwsim.Constants import DAY
from csgwsim import BHBWaveform


def test_BHB():
    print("This is the BHB waveform")
    Tobs = 30*DAY
    delta_f = 1/Tobs

    NF = Tobs
    print(NF)
    freq = np.arange(1, NF+1)*delta_f

    m1 = 35e6
    m2 = 30e6
    chi1 = 0.1
    chi2 = 0.2

    DL = 1.0e3

    import time

    st = time.time()

    bhb = BHBWaveform(m1, m2, chi1, chi2, DL)

    h22 = bhb.h22_FD(freq)

    ed = time.time()

    print(f"Time cost is {ed-st} s")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(h22.freq[:len(h22.amp)], h22.amp)
    plt.ylabel('amplitude')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(h22.phase))
    plt.ylabel('Phase')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(h22.time))
    plt.ylabel('time')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(h22.timep))
    plt.ylabel('dt/df')
    plt.xlabel('freq')

    plt.show()


if __name__ == "__main__":
    print("This is a test of BHB")
    # test_BHB()
