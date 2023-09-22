#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: gen_response_data.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-31 09:40:34
# ==================================

import time
import numpy as np
import matplotlib.pyplot as plt

from csgwsim.Waveform import waveforms
from csgwsim.response import get_td_response, trans_fd_response
from csgwsim.Orbit import detectors
from csgwsim.Constants import DAY, YRSID_SI
from csgwsim.TDI import XYZ_TD, XYZ_FD, AET_FD, TDI_XYZ2AET


def generate_td_data(pars, s_type='gcb', det='TQ', show_y_slr=False):
    print("This is TD response generation code")
    Tobs = 10*DAY  # YRSID_SI / 4
    delta_T = 1

    # tf = np.arange(0,Tobs, delta_T)
    # be careful, the arange method will lose the largest value
    tf = np.linspace(0, Tobs, int(Tobs/delta_T))

    print(f"Testing of {s_type} waveform")
    wf = waveforms[s_type](**pars)
    det = detectors[det](tf)
    st = time.time()
    y_slr = get_td_response(wf, det, tf)
    ed = time.time()
    print(f"Time cost is {ed-st} s for {tf.shape[0]} points")

    if show_y_slr:
        tags = [(1, 2), (2, 1), (2, 3), (3, 2), (3, 1), (1, 3)]
        for i, tag in enumerate(tags):
            plt.figure()
            for j in range(4):
                plt.subplot(4, 1, j+1)
                plt.plot(tf, y_slr[tag][f"{j}L"])
                plt.title(f"y_{tag} [{j}]L")

    st = time.time()
    X, Y, Z = XYZ_TD(y_slr)
    A, E, T = TDI_XYZ2AET(X, Y, Z)
    ed = time.time()
    print("Time cost for cal XYZ and AET with y_slr is ", ed-st)
    # np.save(det+s_type+"_X_td.npy", np.array([tf, X]))

    plt.subplots(2, 3, sharex='all', sharey='all', figsize=(12, 8))
    for i, dd in enumerate(["X", "Y", "Z", "A", "E", "T"]):
        dat = eval(dd)
        plt.subplot(2, 3, i+1)
        plt.plot(tf[:-5], dat[:-5])
        plt.xlabel("Time")
        plt.ylabel("h")
        plt.title(dd)
    plt.tight_layout()


def generate_fd_data(pars, s_type='bhb_PhenomD', det='TQ', show_y_slr=False):
    print("This is a test for frequency domain response")

    NF = 10240
    freq = 10**np.linspace(-4, 0, NF)

    print(f"Testing of {s_type} waveform")
    BHBwf = waveforms[s_type](**pars)
    amp, phase, tf = BHBwf.get_amp_phase(freq)
    amp, phase, tf = amp[(2, 2)], phase[(2, 2)], tf[(2, 2)]

    det = detectors[det](tf)
    h22 = amp * np.exp(1j*phase) * np.exp(2j*np.pi*freq*BHBwf.tc)

    st = time.time()
    y_slr = trans_fd_response(BHBwf.vec_k, BHBwf.p22, det, freq)[0]
    y_slr = {k: v*h22 for k, v in y_slr.items()}
    ed = time.time()
    print(f"time cost for the fd response is {ed-st} s")

    if show_y_slr:
        ln = [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)]
        plt.figure()
        plt.xscale('log')
        for ll in ln:
            plt.plot(freq, np.abs(y_slr[ll]), label=ll)
        plt.legend()
        plt.tight_layout()

    X, Y, Z = XYZ_FD(y_slr, freq, det.L_T)
    A, E, T = AET_FD(y_slr, freq, det.L_T)

    plt.figure()
    plt.loglog(freq, np.abs(X), '-', label='X')
    plt.loglog(freq, np.abs(Y), '-', label='Y')
    plt.loglog(freq, np.abs(Z), '-', label='Z')
    plt.loglog(freq, np.abs(A), '--', label='A')
    plt.loglog(freq, np.abs(E), '--', label='E')
    plt.loglog(freq, np.abs(T), '--', label='T')
    plt.xlabel('f')
    plt.ylabel('h')
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    GCBpars = {"mass1": 0.5,
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
    EMRIpars = {'M': 1e6,
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
    BHBpars = {"mass1": 3.5e6,
               "mass2": 2.1e5,
               'T_obs': YRSID_SI,
               "chi1": 0.2,
               "chi2": 0.1,
               "DL": 1e3,
               "psi": 0.2,
               "iota": 0.3,
               "Lambda": 0.4,
               "Beta": 1.2,
               "tc": 0,
               }
    ecc_par = {'DL': 49.102,  # Luminosity distance (Mpc)
               'mass1': 21.44,  # Primary mass (solar mass)
               'mass2': 20.09,  # Secondary mass(solar mass)
               'Lambda': 3.44,  # Longitude
               'Beta': -0.074,  # Latitude
               'phi_c': 0,  # Coalescence phase
               'T_obs': YRSID_SI,  # Observation time (s)
               'tc': YRSID_SI,  # Coalescence time (s)
               'iota': 0.6459,  # Inclination angle
               'var_phi': 0,  # Observer phase
               'psi': 1.744,  # Polarization angle
               }

    generate_td_data(GCBpars)
    # generate_td_data(EMRIpars, s_type='emri')
    # generate_fd_data(BHBpars, show_y_slr=True)
