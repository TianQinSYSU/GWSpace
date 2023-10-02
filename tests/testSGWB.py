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

SGWBpars = {"fmax": 0.2,
            "fmin": 0.001,
            "fn": 200,
            "tsegmid": 5000,
            "Ttot": 63*5000,
            "omega0": 5e-11,
            "alpha": 0.667,
            "nside_in": 8,
            "blm_vals": (1.0, 0.75, 0.5, 0.7j, 0.7-0.3j, 1.1j),
            "blmax": 2
            }
st = time.time()
SGWB_signal = SGWB(**SGWBpars)
ed = time.time()
print(f"Time cost: {ed-st} s")

hp.mollview(SGWB_signal.signal_in_Gu[0, :], title="The response signal")

tq = TianQinNoise()
TX, TXY = tq.noise_XYZ(SGWB_signal.frange, unit="displacement")/(2e8*np.sqrt(3))**2

plt.figure()
plt.loglog(SGWB_signal.frange, TX, label="XYZ channels noise PSD")
plt.loglog(SGWB_signal.frange, SGWB_signal.signal_SGWB[0, 0, :, 0], label="inject signal sample")
plt.xlabel('frequency[Hz]')
plt.ylabel('PSD [Hz]')
plt.legend(loc='best')
plt.show()
# one year SNR for SGWB
SNR = np.sqrt(24*3600*3.64*np.sum(SGWB_signal.signal_SGWB[0, 0, :, 0]**2/TX**2)/SGWB_signal.frange.size)
print(SNR)
