#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_response.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:56:53
# ==================================

import numpy as np
from csgwsim.response import get_td_response, trans_fd_response
from csgwsim.Waveform import BHBWaveform, GCBWaveform
from csgwsim.Orbit import TianQinOrbit
from csgwsim.TDI import XYZ_FD, AET_FD, XYZ_TD, TDI_XYZ2AET
from csgwsim.Constants import DAY, YRSID_SI

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    print("This is a test for frequency domain response")

    print("Testing of BHB waveform")
    pars = {"mass1": 3.5e6,
            "mass2": 2.1e5,
            'T_obs': YRSID_SI,
            "chi1": 0.2,
            "chi2": 0.1,
            "DL": 1e3,
            "psi": 0.2,
            "iota": 0.3,
            "Lambda": 0.4,
            "Beta": 1.2,
            }
    NF = 10240
    freq = 10**np.linspace(-4, 0, NF)

    BHBwf = BHBWaveform(**pars)
    amp, phase, tf = BHBwf.get_amp_phase(freq)
    # from pyIMRPhenomD import IMRPhenomDh22AmpPhase
    # from scipy.interpolate import InterpolatedUnivariateSpline as Spline
    # wf_phd_class = IMRPhenomDh22AmpPhase(freq, *BHBwf.wave_para_phenomd())
    # freq, amp, phase = wf_phd_class.GetWaveform()  # freq, amp, phase
    # tf_spline = Spline(freq, 1/(2*np.pi)*(phase - phase[0])).derivative()
    # tf = tf_spline(freq)+BHBwf.tc

    det = TianQinOrbit(tf)
    h22 = amp * np.exp(1j*phase) * np.exp(2j*np.pi*freq*BHBwf.tc)

    st = time.time()
    yslr = trans_fd_response(BHBwf.vec_k, BHBwf.p22, det, freq)[0]
    ed = time.time()
    print(f"time cost for the fd response is {ed-st} s")

    ln = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
    plt.figure()
    plt.xscale('log')
    for yy, ll in zip(yslr, ln):
        plt.plot(freq, np.abs(yy*h22), label=ll)
    plt.legend()
    plt.tight_layout()

    X, Y, Z = XYZ_FD(yslr[(2, 2)], freq, det.L_T)
    A, E, T = AET_FD(yslr[(2, 2)], freq, det.L_T)

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

    print("This is TD response generation code")
    Tobs = 4*DAY  # YRSID_SI / 4
    delta_f = 1/Tobs
    delta_T = 1
    f_max = 1/(2*delta_T)

    # tf = np.arange(0, Tobs, delta_T)
    # be careful, the arange method will lose the largest value
    tf_ = np.linspace(0, Tobs, int(Tobs/delta_T))

    print("Testing of GCB waveform")
    GCBpars = {"Mc": 0.5,
               "DL": 0.3,
               "phi0": 0.0,
               "f0": 0.001,
               "psi": 0.2,
               "iota": 0.3,
               "lambda": 0.4,
               "beta": 1.2,
               }

    GCBwf = GCBWaveform(**GCBpars)
    det = TianQinOrbit(tf_)
    st = time.time()
    yslr_ = get_td_response(GCBwf, det, tf_)
    ed = time.time()

    print("Time cost is %f s for %d points" % (ed-st, tf_.shape[0]))

    plt.figure()

    tags = [(1, 2), (2, 1), (2, 3), (3, 2), (3, 1), (1, 3)]

    for i, tag in enumerate(tags):
        plt.figure()
        for j in range(4):
            plt.subplot(4, 1, j+1)
            plt.plot(tf, yslr[tag][f"%sL" % j])
            plt.title(f"y_{tag} [%s]L" % j)

    st = time.time()
    X, Y, Z = XYZ_TD(yslr)
    A, E, T = TDI_XYZ2AET(X, Y, Z)
    ed = time.time()

    print("Time cost for cal XYZ and AET with yslr is ", ed-st)

    plt.figure()
    for i, dd in enumerate(["X", "Y", "Z", "A", "E", "T"]):
        dat = eval(dd)
        plt.subplot(2, 3, i+1)
        plt.plot(tf[:-5], dat[:-5], label=dd)

        plt.xlabel("Time")
        plt.ylabel(dd)

        plt.legend(loc="best")

    plt.show()
