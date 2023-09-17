#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Generate_GW_Data.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-31 09:40:34
# ==================================

import os
import sys
import time
import matplotlib.pyplot as plt

# import numpy as np
# from Constants import *

try:
    import bbh
except:
    bbh_path = './Waveforms/bbh'
    abs_bbh_path = os.path.abspath(bbh_path)

    if abs_bbh_path not in sys.path:
        sys.path.append(abs_bbh_path)

try:
    import few
except:
    few_path = './Waveforms/FastEMRIWaveforms'
    abs_few_path = os.path.abspath(few_path)

    if abs_few_path not in sys.path:
        sys.path.append(abs_few_path)

from pyINIDetectors import INITianQin

from pyWaveForm import WaveForm
from pyFDResponse import FDResponse
from TDI import *


def Generate_TD_Data(pars, show_yslr=False):
    print("This is TD response generation code")
    Tobs = 10*DAY  # YRSID_SI / 4
    delta_f = 1/Tobs
    delta_T = 1
    f_max = 1/(2*delta_T)

    # tf = np.arange(0,Tobs, delta_T)
    # be careful, the arange method will lose the largest value
    tf = np.linspace(0, Tobs, int(Tobs/delta_T))

    print(f"Testing of {pars['type']} waveform")

    TQ = INITianQin()
    td = TDResponse(pars, TQ)

    st = time.time()
    yslr = td.Evaluate_yslr(tf)
    ed = time.time()

    print("Time cost is %f s for %d points" % (ed-st, tf.shape[0]))

    if show_yslr:
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


def Generate_FD_Data(pars, show_yslr=False):
    print("This is a test for frequency domain response")

    print("Testing of BHB waveform")

    TQ = INITianQin()
    fd = FDResponse(pars, TQ)

    NF = 10240
    freq = 10**np.linspace(-4, 0, NF)

    BHBwf = WaveForm(pars)

    amp, phase, tf, tfp = BHBwf.get_amp_phase(freq, )

    st = time.time()

    Gslr, zeta = fd.EvaluateGslr(tf[(2, 2)], freq)  # fd.Evaluate_yslr(freq)
    yslr = fd.Evaluate_yslr(freq)  # fd.Evaluate_yslr(freq)
    ed = time.time()

    print(f"time cost for the fd response is {ed-st} s")

    if show_yslr:
        mode = [(2, 2)]
        ln = [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)]

        for ll in ln:
            plt.figure()
            gg = Gslr[ll]
            yy = yslr[mode[0]][ll]
            plt.plot(freq, gg, '-r')
            plt.plot(freq, yy, '--b')
            plt.title(ll)

            plt.xscale('log')

    X, Y, Z = XYZ_FD(yslr[(2, 2)], freq, LT=fd.LT)
    A, E, T = AET_FD(yslr[(2, 2)], freq, fd.LT)

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


if __name__ == "__main__":
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

    EMRIpars = {"type": "EMRI",
                'M': 1e6,
                'a': 0.1,
                'mu': 1e1,
                'p0': 12.0,
                'e0': 0.2,
                'x0': 1.0,
                'qK': 0.2,
                'phiK': 0.2,
                'qS': 0.3,
                'phiS': 0.3,
                'dist': 1.0,
                'Phi_phi0': 1.0,
                'Phi_theta0': 2.0,
                'Phi_r0': 3.0,
                'psi': 0.4,
                'iota': 0.2,
                }

    BHBpars = {"type": "BHB",
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

    # Generate_TD_Data(GCBpars, show_yslr=True)
    Generate_TD_Data(EMRIpars, show_yslr=True)
    # Generate_FD_Data(BHBpars)
