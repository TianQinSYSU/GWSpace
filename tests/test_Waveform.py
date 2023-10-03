#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_Waveform.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-09-05 17:52:12
# ==================================

import numpy as np
import matplotlib.pyplot as plt
from time import time, strftime

from gwspace.Waveform import BHBWaveform, GCBWaveform, BHBWaveformEcc, EMRIWaveform, BurstWaveform
from gwspace.constants import YRSID_SI


def test_Burst():
    print("This is a test of Burst")
    Burst = BurstWaveform(amp=0.5, tau=1000, fc=0.001, tc=5000)

    Tobs = 10000  # YRSID_SI / 4
    delta_T = 1
    tf = np.arange(0, Tobs, delta_T)

    hp, hc = Burst(tf)

    plt.figure()
    plt.plot(tf, hp, 'r-', label=r'$h_+^{\rm S}$')
    plt.plot(tf, hc, 'b--', label=r'$h_\times^{\rm S}$')
    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")
    plt.legend(loc="best")
    plt.show()


def test_GCB():
    Tobs = 10000  # const.YRSID_SI / 4
    delta_T = 1

    tf = np.arange(0, Tobs, delta_T)

    print("Testing of GCB waveform")
    GCBpars = {"mass1": 0.5,
               "mass2": 0.5,
               'T_obs': YRSID_SI,
               "DL": 0.3,
               "phi0": 0.0,
               "f0": 0.001,
               "psi": 0.2,
               "iota": 0.3,
               "Lambda": 0.4,
               "Beta": 1.2,
               }

    GCBwf = GCBWaveform(**GCBpars)
    hpssb, hcssb = GCBwf.get_hphc(tf)

    plt.figure()
    plt.plot(tf, hpssb, 'r-', label=r'$h_+^{\rm SSB}$')
    plt.plot(tf, hcssb, 'b--', label=r'$h_\times^{\rm SSB}$')
    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")
    plt.legend(loc="best")
    plt.tight_layout()


def test_BHB():
    print("Testing of BHB waveform")
    BHBpars = {"mass1": 3.5e6,
               "mass2": 2.1e5,
               'T_obs': YRSID_SI,
               "chi1": 0.2,
               "chi2": 0.1,
               "DL": 1e3,
               "phi_c": 0.0,
               "psi": 0.2,
               "iota": 0.3,
               "Lambda": 0.4,
               "Beta": 1.2,
               "tc": 0,
               }

    BHBwf = BHBWaveform(**BHBpars)

    NF = 1024
    freq = 10**np.linspace(-4, 0, NF)
    amp, phase, tf = BHBwf.get_amp_phase(freq)

    plt.figure()
    plt.loglog(freq, amp[(2, 2)])
    plt.ylabel('amplitude')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(phase[(2, 2)]))
    plt.ylabel('Phase')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(tf[(2, 2)]))
    plt.ylabel('time')
    plt.xlabel('freq')

    plt.show()


def test_BHB_ecc():
    default = {'DL': 49.102,  # Luminosity distance (Mpc)
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

    start_time = time()
    print(strftime("%Y-%m-%d %H:%M:%S"))
    delta_f = 1e-5
    para = BHBWaveformEcc(**default, eccentricity=0.1)
    wf, _ = para.gen_ori_waveform(delta_f=delta_f, hphc=True)
    para_ = BHBWaveformEcc(**default, eccentricity=0.)
    wf_, _ = para_.gen_ori_waveform(delta_f=delta_f, hphc=True)
    print(strftime("%Y-%m-%d %H:%M:%S"), f'Finished in {time() - start_time: .5f}s', '\n')

    freq = delta_f * np.array(range(len(wf)))
    plt.figure()
    plt.loglog(freq, np.abs(wf), label='e=0.1')
    plt.loglog(freq, np.abs(wf_), label='e=0')
    plt.xlim(para.f_min, 1)
    plt.legend()
    plt.tight_layout()


def test_EMRI():
    print("This is a test of loading EMRI waveform")
    # parameters
    Tobs = 0.3*YRSID_SI  # years
    dt = 15.0  # seconds

    pars = {'M': 1e6,
            'a': 0.1,  # will be ignored in Schwarzschild waveform
            'mu': 1e1,
            'p0': 12.0,
            'e0': 0.2,
            'x0': 1.0,  # will be ignored in Schwarzschild waveform
            'qK': 0.2,  # polar spin angle
            'phiK': 0.2,  # azimuthal viewing angle
            'qS': 0.3,  # polar sky angle
            'phiS': 0.3,  # azimuthal viewing angle
            'dist': 1.0,  # distance
            'Phi_phi0': 1.0,
            'Phi_theta0': 2.0,
            'Phi_r0': 3.0,
            'psi': 0.4,
            'iota': 0.2,
            }

    wf = EMRIWaveform(**pars)

    tf = np.arange(0, Tobs, dt)
    hp, hc = wf.get_hphc(tf,)

    plt.figure()
    plt.plot(tf[:2000], hp[:2000])
    plt.plot(tf[:2000], hc[:2000])
    plt.show()


if __name__ == '__main__':
    # test_Burst()
    # test_GCB()
    # test_BHB()
    test_BHB_ecc()
    # test_EMRI()
