#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: generate_noise.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-12 16:17:43
#==================================

from csgwsim.Noise import white_noise as wn
import csgwsim.Constants as const

import numpy as np
import matplotlib.pyplot as plt

asd = 1.0
fs = 1

Tobs = const.YRSID_SI / 12
df = 1/Tobs

size = int(Tobs/fs)

ff = np.arange(size) * df

noise_white = wn(fs, size, asd)


plt.plot(ff, noise_white)

plt.show()
