#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_Noise.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-09-05 15:03:28
# ==================================

import numpy as np
import matplotlib.pyplot as plt

from gwspace.Noise import TianQinNoise, LISANoise
from gwspace.Constants import PI, C_SI
from gwspace.wrap import frequency_noise_from_psd

if __name__ == "__main__":
    freq_ = np.logspace(-5, 1, 10000)
    # freq_ = np.linspace(1e-4, 1, 2**10)

    lisa_noise = LISANoise()
    tq_noise = TianQinNoise()

    LA, LT = lisa_noise.noise_AET(freq_, wd_foreground=1.)
    TA, TT = tq_noise.noise_AET(freq_)

    LX, LXY = lisa_noise.noise_XYZ(freq_, wd_foreground=1.)
    TX, TXY = tq_noise.noise_XYZ(freq_)

    noise_TX = frequency_noise_from_psd(TX, freq_[1]-freq_[0])
    # np.save("noise_TX.npy", noise_TX)

    # =======================================
    plt.figure()
    plt.title("Sensitivity of TianQin and LISA---with displacement")
    plt.loglog(freq_, lisa_noise.sensitivity(freq_, wd_foreground=1.), label="LISA")
    plt.loglog(freq_, tq_noise.sensitivity(freq_), label="TianQin")
    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")
    plt.legend(loc="best")

    # =======================================
    plt.figure()
    plt.title("PSD of noise for A E T channel---relativeFrequency")
    plt.loglog(freq_, LA, "-", label="LISA AE")
    plt.loglog(freq_, LT, "-.", label="LISA T")
    plt.loglog(freq_, TA, "--", label="TianQin AE")
    plt.loglog(freq_, TT, "-.", label="TianQin T")
    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")
    plt.legend(loc="best")

    # =======================================
    plt.figure()
    plt.title("PSD of noise for X and X-Y channel---relativeFrequency")
    plt.loglog(freq_, LX, "-", label="LISA X")
    plt.loglog(freq_, LXY, "-.", label="LISA XY")
    plt.loglog(freq_, TX, "--", label="TianQin X")
    # plt.loglog(freq_, noise_TX, "--", label="TianQin X")
    plt.loglog(freq_, TXY, "-.", label="TianQin XY")
    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")
    plt.legend(loc="best")

    # =======================================
    lisa_sa, lisa_sp = lisa_noise.noises_displacement(freq_)
    tq_sa, tq_sp = tq_noise.noises_displacement(freq_)

    plt.figure()
    plt.title("Acc and Pos noise model---displacement")
    plt.loglog(freq_, lisa_sa, ':r', label="LISA Sa")
    plt.loglog(freq_, lisa_sp, '--r', label="LISA Sp")
    plt.loglog(freq_, tq_sa, "-b", label="TianQin Sa")
    plt.plot(freq_, tq_sp, "--b", label="TianQin Sp")
    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")
    plt.legend(loc="best")

    # =======================================
    lisa_sa, lisa_sp = lisa_noise.noises_relative_freq(freq_)
    tq_sa, tq_sp = tq_noise.noises_relative_freq(freq_)

    # Acc and Pos noise from Marsat 2003.00357: eq: A17a and A17b
    M_sop = lambda f: (2*PI*f/C_SI)**2*(8.9**2+1.7**2+4)*1e-24
    M_spm = lambda f: ((2*PI*f*C_SI)**(-2)*9e-30*(1+36*(1e-8/f**2+(3e-5/f)**10))
                       + 0.25*1.7e-12**2*(2*PI*f/C_SI)**2)

    plt.figure()
    plt.title("Acc and Pos noise model---relativeFrequency")
    plt.loglog(freq_, lisa_sa, ':r', label="LISA Sa")
    plt.loglog(freq_, lisa_sp, '--r', label="LISA Sp")
    plt.loglog(freq_, tq_sa, "-b", label="TianQin Sa")
    plt.loglog(freq_, tq_sp, "--b", label="TianQin Sp")
    plt.loglog(freq_, M_spm(freq_), '-g', label="Marsat Sa")
    plt.loglog(freq_, M_sop(freq_), '--g', label="Marsat Sp")
    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")
    plt.legend(loc="best")
