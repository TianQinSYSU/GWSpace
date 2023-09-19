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
import numpy as np
import matplotlib.pyplot as plt

from csgwsim.Waveform import waveforms
from csgwsim.response import get_td_response
from csgwsim.Orbit import TianQinOrbit, detectors
from csgwsim.Constants import DAY, YRSID_SI, MSUN_SI, MPC_SI
from csgwsim.TDI import XYZ_TD, TDI_XYZ2AET


def Generate_TD_Data(pars, detector='TQ', show_yslr=False):
    print("This is TD response generation code")
    Tobs = 10*DAY  # YRSID_SI / 4
    delta_f = 1/Tobs
    delta_T = 1
    f_max = 1/(2*delta_T)

    # tf = np.arange(0,Tobs, delta_T)
    # be careful, the arange method will lose the largest value
    tf = np.linspace(0, Tobs, int(Tobs/delta_T))

    print(f"Testing of {pars['type']} waveform")

    #help(WF.GCBWaveform)
    WFs = waveforms[pars['type']]
    wf = WFs(**pars)

    orbits = detectors[detector]
    det = orbits(tf)
    
    st = time.time()
    yslr = get_td_response(wf, det, tf)
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

    np.save(detector+pars["type"]+"_X_td.npy", np.array([tf, X]))

    print("Time cost for cal XYZ and AET with yslr is ", ed-st)

    '''

    plt.figure()
    for i, dd in enumerate(["X", "Y", "Z", "A", "E", "T"]):
        dat = eval(dd)
        plt.subplot(2, 3, i+1)
        plt.plot(tf[:-5], dat[:-5], label=dd)

        plt.xlabel("Time")
        plt.ylabel(dd)

        plt.legend(loc="best")

    plt.show()
    '''


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
    f0 = [0.00622028]
    fd = [7.48528554e-16]
    beta = [-0.082205]  # ecliptic latitude [rad]
    Lambda = [2.10225]  # ecliptic longitude [rad]
    Amp = [6.37823e-23]
    iota = [0.645772]
    psi = [2.91617795]
    phi0 = [3.1716561]
    
    GCBpars = {"type": "gcb",
               "mass1": 0.5,
               "mass2": 0.5,
               "DL": 0.3,
               "phi0": 0.0,
               "f0": 0.001,
               "psi": 0.2,
               "iota": 0.3,
               "Lambda": 0.4,
               "Beta": 1.2,
               "T_obs": YRSID_SI,
               }

    EMRIpars = {"type": "emri",
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
    eccpara = {"type": 'bhb_EccFD',
            'delta_f': 0.0001,
            'f_final': 1,
            'f_lower': 0.01,
            'mass1': 10 * MSUN_SI,
            'mass2': 10 * MSUN_SI,
            'inclination': 0.23,
            'eccentricity': 0.4,
            'long_asc_nodes': 0.23,
            'coa_phase': 0,
            'distance': 100 * MPC_SI,
            'obs_time': 365*24*3600}
    
    Generate_TD_Data(GCBpars)
    #Generate_TD_Data(EMRIpars)#, show_yslr=True)
    #Generate_FD_Data(eccpara)
