#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: gen_response_data.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-08-31 09:40:34
# ==================================

import time
import numpy as np
import matplotlib.pyplot as plt

from gwspace.Waveform import waveforms
from gwspace.response import get_y_slr_td, trans_y_slr_fd, get_XYZ_td, get_XYZ_fd, get_AET_fd, tdi_XYZ2AET
from gwspace.Orbit import detectors
from gwspace.constants import DAY, YRSID_SI, MONTH
from gwspace.Noise import TianQinNoise, LISANoise, TaijiNoise


def generate_td_data(pars, s_type='gcb', det='TQ', show_y_slr=False):
    print("This is TD response generation code")
    Tobs = 10*DAY  # YRSID_SI / 4
    delta_T = 1

    # tf = np.arange(0,Tobs, delta_T)
    # be careful, the arange method will lose the largest value
    tf = np.linspace(0, Tobs, int(Tobs/delta_T))

    print(f"Testing of {s_type} waveform")
    wf = waveforms[s_type](**pars)
    st = time.time()
    y_slr = get_y_slr_td(wf, tf, det)
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
    X, Y, Z = get_XYZ_td(y_slr)
    A, E, T = tdi_XYZ2AET(X, Y, Z)
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
    y_slr = trans_y_slr_fd(BHBwf.vec_k, BHBwf.p22, det, freq)[0]
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

    X, Y, Z = get_XYZ_fd(y_slr, freq, det.L_T)
    A, E, T = get_AET_fd(y_slr, freq, det.L_T)

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


def generate_MBHB_with_PSD_joint(pars, s_type='bhb_PhenomD'):
    tq_noise = TianQinNoise()
    lisa_noise = LISANoise()
    taiji_noise = TaijiNoise()
    freq_ = np.logspace(-5, 0, 10000)
    TQ_A, _ = tq_noise.noise_AET(freq_)
    LISA_A, _ = lisa_noise.noise_AET(freq_, wd_foreground=90/365.25)
    Taiji_A, _ = taiji_noise.noise_AET(freq_)
    
    BHBwf = waveforms[s_type](**pars)
    delta_f = 1e-6  # 1/BHBwf.T_obs
    freq = np.arange(np.ceil(BHBwf.f_min/delta_f)*delta_f, 1., delta_f)
    amp, phase, tf = BHBwf.get_amp_phase(freq)
    amp, phase, tf = amp[(2, 2)], phase[(2, 2)], tf[(2, 2)]
    h22 = amp*np.exp(1j*phase)*np.exp(2j*np.pi*freq*BHBwf.tc)
    SMBBH_A = {}

    for d in ['TQ', 'LISA', 'Taiji']:
        det = detectors[d](tf)
        y_slr = trans_y_slr_fd(BHBwf.vec_k, BHBwf.p22, det, freq)[0]
        y_slr = {k: v*h22 for k, v in y_slr.items()}
        SMBBH_A[d], _, _ = get_AET_fd(y_slr, freq, det.L_T)

    plt.figure(figsize=(9, 6))
    plt.loglog(freq_, np.sqrt(TQ_A), 'k--', label='TQ noise')
    plt.loglog(freq_, np.sqrt(LISA_A), 'k-.', label='LISA noise')
    plt.loglog(freq_, np.sqrt(Taiji_A), 'k:', label='Taiji noise')

    ## to determine where the waveform is zero
    #print(np.where(SMBBH_A['TQ'] == 0))
    print()
    ndim = 10850-1
    plt.loglog(freq[:ndim], np.abs(SMBBH_A['TQ'][:ndim])*np.sqrt(freq[:ndim]), 'r-', label='TQ')
    plt.loglog(freq[:ndim], np.abs(SMBBH_A['LISA'][:ndim])*np.sqrt(freq[:ndim]), 'g-', label='LISA')
    plt.loglog(freq[:ndim], np.abs(SMBBH_A['Taiji'][:ndim])*np.sqrt(freq[:ndim]), 'b-', label='Taiji')

    
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('$\\sqrt{S_n}$ [Hz$^{-1/2}$]')
    # plt.tick_params(labelsize=12)
    # plt.grid(which='both', alpha=0.5)
    # plt.xlim(0.9*BHBwf.f_min, 1e-1)
    plt.xlim(1e-4, 1e-1)
    plt.ylim(1e-23, 1e-16)
    plt.legend(loc="best")  # fontsize=12)
    plt.tight_layout()
    
    #plt.savefig("../../../TQ-SDS/figs/MBHB_fd.pdf")
    plt.show()


def generate_SBHB_with_PSD_joint(par, s_type='bhb_EccFD'):
    tq_noise = TianQinNoise()
    lisa_noise = LISANoise()
    taiji_noise = TaijiNoise()
    freq_ = np.logspace(-5, 0, 10000)
    TQ_A, _ = tq_noise.noise_AET(freq_)
    LISA_A, _ = lisa_noise.noise_AET(freq_, wd_foreground=90/365.25)
    Taiji_A, _ = taiji_noise.noise_AET(freq_)
    
    BHBwf = waveforms['bhb_PhenomD'](**par)
    delta_f = 1e-5  # 1/BHBwf.T_obs
    freq_e0 = np.arange(np.ceil(BHBwf.f_min/delta_f)*delta_f, 1., delta_f)
    amp, phase, tf = BHBwf.get_amp_phase(freq_e0)
    amp, phase, tf = amp[(2, 2)], phase[(2, 2)], tf[(2, 2)]
    h22 = amp*np.exp(1j*phase)*np.exp(2j*np.pi*freq_e0*BHBwf.tc)

    ecc_wf = waveforms[s_type](**par, eccentricity=0.1)
    smBBH_A_e0, smBBH_A_e1 = {}, {}
    for d in ['TQ', 'LISA', 'Taiji']:
        det = detectors[d](tf)
        y_slr = trans_y_slr_fd(BHBwf.vec_k, BHBwf.p22, det, freq_e0)[0]
        y_slr = {k: v*h22 for k, v in y_slr.items()}
        smBBH_A_e0[d], _, _ = get_AET_fd(y_slr, freq_e0, det.L_T)
        smBBH_A_e1[d], freq_e1 = ecc_wf.fd_tdi_response(det=d, delta_f=delta_f)
        
    plt.figure(figsize=(9, 6))
    plt.loglog(freq_, np.sqrt(TQ_A), 'k--', label='TQ noise')
    plt.loglog(freq_, np.sqrt(LISA_A), 'k-.', label='LISA noise')
    plt.loglog(freq_, np.sqrt(Taiji_A), 'k:', label='Taiji noise')
    
    plt.loglog(freq_e1, np.abs(smBBH_A_e1['TQ'])*np.sqrt(freq_e1), 'm-', label='TQ: e=0.1')
    plt.loglog(freq_e1, np.abs(smBBH_A_e1['LISA'])*np.sqrt(freq_e1), 'b-', label='LISA: e=0.1')
    plt.loglog(freq_e1, np.abs(smBBH_A_e1['Taiji'])*np.sqrt(freq_e1), 'c-', label='Taiji: e=0.1')
        
    plt.loglog(freq_e0, np.abs(smBBH_A_e0['TQ'])*np.sqrt(freq_e0), 'r--', label='TQ: e=0')
    plt.loglog(freq_e0, np.abs(smBBH_A_e0['LISA'])*np.sqrt(freq_e0), 'y--', label='LISA: e=0')
    plt.loglog(freq_e0, np.abs(smBBH_A_e0['Taiji'])*np.sqrt(freq_e0), 'g--', label='Taiji: e=0')
        
    plt.xlabel('Frequency [Hz]')  # , fontsize=12)
    plt.ylabel('$\\sqrt{S_n}$ [Hz$^{-1/2}$]')  # , fontsize=12)
    plt.tick_params(labelsize=12)
    # plt.grid(which='both', alpha=0.5)
    plt.xlim(BHBwf.f_min, 1)
    
    # plt.ylim(1e-23, 1e-16)
    
    plt.legend(loc="best", ncols=2)  # fontsize=12)
    plt.tight_layout()
    
    # plt.savefig("../../../TQ-SDS/figs/SMBH_fd.pdf")
    plt.show()


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
                'T_obs': YRSID_SI,
                }
    BHBpars = {"mass1": 3.5e6,
               "mass2": 2.1e5,
               'T_obs': MONTH*3,
               "chi1": 0.2,
               "chi2": 0.1,
               "DL": 1e3,
               "psi": 0.2,
               "iota": 0.3,
               "Lambda": 0.4,
               "Beta": 1.2,
               "tc": 0,
               }
    ecc_par = {'DL': 100,  # Luminosity distance (Mpc)
               'mass1': 35.6,  # Primary mass (solar mass)
               'mass2': 30.6,  # Secondary mass(solar mass)
               'Lambda': 4.7,  # Longitude
               'Beta': -1.5,  # Latitude
               'phi_c': 0,  # Coalescence phase
               'T_obs': MONTH*3,  # Observation time (s)
               'tc': 0,  # Coalescence time (s)
               'iota': 0.3,  # Inclination angle
               'var_phi': 0,  # Observer phase
               'psi': 0.2,  # Polarization angle
               }  # masses of GW150914
    generate_td_data(GCBpars)
    # generate_td_data(EMRIpars, s_type='emri')
    # generate_fd_data(BHBpars, show_y_slr=False)

    # generate_MBHB_with_PSD_joint(BHBpars)
    # generate_SBHB_with_PSD_joint(ecc_par)
