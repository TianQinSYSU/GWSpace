#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_pyResponse.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:56:53
# ==================================

import numpy as np
from csgwsim.pyResponse import FDResponse, TDResponse
from csgwsim.pyWaveForm import BasicWaveform
from csgwsim.TDI import XYZ_FD, AET_FD

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    print("This is a test for frequency domain response")

    print("Testing of BHB waveform")

    pars = {"type": "BHB",
            "m1": 3.5e6,
            "m2": 2.1e5,
            "chi1": 0.2,
            "chi2": 0.1,
            "DL": 1e3,
            "phic": 0.0,
            "MfRef_in": 0,
            "psi": 0.2,
            "iota": 0.3,
            "lambda": 0.4,
            "beta": 1.2,
            "tc": 0,
            }

    fd = FDResponse(pars)

    NF = 10240
    freq = 10**np.linspace(-4, 0, NF)

    BHBwf = BasicWaveform(**pars)

    amp, phase, tf, tfp = BHBwf.amp_phase(freq)

    st = time.time()

    Gslr, zeta = fd.EvaluateGslr(tf[(2, 2)], freq)  # fd.Evaluate_yslr(freq)
    yslr_ = fd.Evaluate_yslr(freq)  # fd.Evaluate_yslr(freq)
    ed = time.time()

    print(f"time cost for the fd response is {ed-st} s")

    mode = [(2, 2)]
    ln = [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)]

    for ll in ln:
        plt.figure()
        gg = Gslr[ll]
        yy = yslr_[mode[0]][ll]
        plt.plot(freq, gg, '-r')
        plt.plot(freq, yy, '--b')
        plt.title(ll)

        plt.xscale('log')

    X, Y, Z = XYZ_FD(yslr_[(2, 2)], freq, LT=fd.LT)
    A, E, T = AET_FD(yslr_[(2, 2)], freq, fd.LT)

    plt.figure()
    plt.loglog(freq, np.abs(X), '-r', label='X')
    plt.loglog(freq, np.abs(Y), '--g', label='Y')
    plt.loglog(freq, np.abs(Z), ':b', label='Z')

    plt.xlabel('f')
    plt.ylabel('X,Y,Z')
    plt.legend(loc='best')

    plt.figure()
    plt.loglog(freq, np.abs(A), '-r', label='A')
    plt.loglog(freq, np.abs(E), '--g', label='E')
    plt.loglog(freq, np.abs(T), ':b', label='T')

    plt.xlabel('f')
    plt.ylabel('A,E,T')
    plt.legend(loc='best')

    plt.show()

    from csgwsim.Constants import DAY
    print("This is TD response generation code")

    Tobs = 4*DAY  # YRSID_SI / 4
    delta_f = 1/Tobs
    delta_T = 1
    f_max = 1/(2*delta_T)

    tf_ = np.arange(0, Tobs, delta_T)

    print("Testing of GCB waveform")
    GCBpars = {"type": "GCB",
               "Mc": 0.5,
               "DL": 0.3,
               "phi0": 0.0,
               "f0": 0.001,
               "psi": 0.2,
               "iota": 0.3,
               "lambda": 0.4,
               "beta": 1.2,
               }

    print("Mc" in GCBpars.keys())

    # GCBwf = BasicWaveform(GCBpars)
    # hpssb, hcssb = GCBwf(tf)

    td = TDResponse(GCBpars, )

    st = time.time()
    yslr_ = td.Evaluate_yslr(tf_)
    ed = time.time()

    print("Time cost is %f s for %d points" % (ed-st, tf_.shape[0]))

    plt.figure()

    tags = [(1, 2), (2, 1), (2, 3), (3, 2), (3, 1), (1, 3)]

    for i, tag in enumerate(tags):
        for j in range(4):
            plt.figure(i*4+j)
            plt.plot(tf_, yslr_[tag][f"{j}L"])
            plt.title(f"y_{tag} [j]L")

    plt.show()
