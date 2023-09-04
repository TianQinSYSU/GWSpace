#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: BHB.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-24 09:17:54
# ==================================

import os
import sys
import numpy as np

from Constants import *
from Waveforms.PyIMRPhenomD import IMRPhenomD as pyIMRD
from Waveforms.PyIMRPhenomD import IMRPhenomD_const as PyIMRC

try:
    import bbh
except:
    bbh_path = './Waveforms/bbh'
    abs_bbh_path = os.path.abspath(bbh_path)

    if abs_bbh_path not in sys.path:
        sys.path.append(abs_bbh_path)

    import bbh


class BHBWaveform:
    """
    This is Waveform for BHB
    ------------------------
    Parameters:
    - m1, m2: mass of black holes
    - chi1, chi2: spin of the two black holes
    - DL: in MPC
    """

    def __init__(self, m1, m2, chi1=0., chi2=0., DL=1.0, phic=0, MfRef_in=0):

        # set parameters for a black hole binary system
        self.chi1 = chi1
        self.chi2 = chi2
        self.m1_SI = m1*MSUN_SI
        self.m2_SI = m2*MSUN_SI
        self.distance = DL*MPC_SI
        self.phic = phic
        self.MfRef_in = MfRef_in

    def h22_FD(self, freq, fRef=0, t0=0):
        NF = freq.shape[0]

        amp_imr = np.zeros(NF)
        phase_imr = np.zeros(NF)
        if PyIMRC.findT:
            time_imr = np.zeros(NF)
            timep_imr = np.zeros(NF)
        else:
            time_imr = np.zeros(0)
            timep_imr = np.zeros(0)

        # Create structure for Amp/phase/time FD waveform
        self.h22 = pyIMRD.AmpPhaseFDWaveform(NF, freq, amp_imr, phase_imr, time_imr, timep_imr, fRef, t0)

        # Generate h22 FD amplitude and phse on a given set of frequencies
        self.h22 = pyIMRD.IMRPhenomDGenerateh22FDAmpPhase(
            self.h22, freq,
            self.phic, self.MfRef_in,
            self.m1_SI, self.m2_SI,
            self.chi1, self.chi2,
            self.distance)

        return self.h22


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
