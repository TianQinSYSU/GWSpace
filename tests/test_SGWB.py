#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_SGWB.py
# Author: Zhiyuan Li
# Mail:
# ==================================

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import time

from gwspace.Noise import TianQinNoise
from gwspace.SGWB import SGWB

SGWBpars = {"nside": 8,
            "omega0": 5e-11,
            "alpha": 0.667,
            "T_obs": 63*5000,
            # "blm_vals": (1.0, 0.75, 0.5, 0.7j, 0.7-0.3j, 1.1j),
            # "blmax": 2,
            "theta": 1.3,
            "phi": 1.2,
            }
signal_pars = {"f_max": 0.2,
               "f_min": 0.001,
               "fn": 200,
               "t_segm": 5000,
               }
st = time.time()
SGWB_signal = SGWB(**SGWBpars)
res_signal, frange = SGWB_signal.get_response_signal(**signal_pars)
ed = time.time()
print(f"Time cost: {ed-st} s")

signal_in_gu = SGWB_signal.get_ori_signal(frange)
hp.mollview(signal_in_gu[:, 0], title="The response signal")

tq = TianQinNoise()
TX, TXY = tq.noise_XYZ(frange, unit="displacement")/(2e8*np.sqrt(3))**2

plt.figure()
plt.loglog(frange, TX, label="XYZ channels noise PSD")
plt.loglog(frange, np.abs(res_signal[:, 0, 0, 0]), label="inject signal sample")
plt.xlabel('frequency[Hz]')
plt.ylabel('PSD [Hz]')
plt.legend(loc='best')
plt.show()

SNR = np.sqrt(24*3600*3.64*np.sum(res_signal[:, 0, 0, 0]**2/TX**2).real/frange.size)
print("1 year SNR for SGWB:", SNR)
