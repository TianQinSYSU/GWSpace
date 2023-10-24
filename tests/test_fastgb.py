#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_fastgb.py
# Author: En-Kun Li
# Mail: lienk@mail.sysu.edu.cn
# Created Time: 2023-09-07 10:05:09
# ==================================

import numpy as np
import matplotlib.pyplot as plt
import time

from gwspace import FastGB as FB


# 两年观测时长，15s采一个点。
# 在fastGB程序中，T_obs=62914560和dt=15是默认值。
# 这个15s的采样间隔是适用于LISA的，实际天琴的采样间隔可能是1s左右，你可以自行修改成更符合实情的T_obs和dt。
dt = 15.0
GCBpars = {"mass1": 0.5,
           "mass2": 0.5,
           'T_obs': 62914560.0,
           "phi0": 3.1716561,
           "f0": 0.00622028,
           "fdot": 7.48528554e-16,
           "psi": 2.91617795,
           "iota": 0.645772,
           "Lambda": 2.10225,  # ecliptic longitude [rad]
           "Beta": -0.082205,  # ecliptic latitude [rad]
           }
# Amp = 6.37823e-23
fastB = FB.FastGB(**GCBpars)

oversample = 16384
# 这里的16384是oversample的值，用于调节对“慢项”的降采样的程度。oversample值必须为2的倍数，否则傅里叶变换时无法用gsl_fft函数。
# 在这里，原采样点数62914560/15=4194304，而“慢项”采样点数为128*oversample
# （“128”的值根据T_obs和f0的不同而不同，具体可以本文档最后一部分）
# 如果不进行降采样，则4194304=128*oversample，算出来oversample=16384刚好也是2的倍数，可以直接用。
# 这样，就得到了不进行降采样情况下的J0806双星引力波频域波形。

st = time.time()
f, X, Y, Z = fastB.get_fastgb_fd(dt=dt, oversample=oversample)
ed = time.time()
print(f"time cost is {ed-st} s")

plt.figure()
plt.loglog(f, np.abs(X**2), label='X')
plt.loglog(f, np.abs(Y**2), label='Y')
plt.loglog(f, np.abs(Z**2), label='Z')
plt.xlim(0.00620, 0.00624)
plt.tight_layout()

# st = time.time()
# t, X, Y, Z = fastB.get_fastgb_td(dt=dt, oversample=oversample)
# ed = time.time()
# print(f"time cost is {ed-st} s")
#
# plt.figure()
# plt.plot(t, X, label='X')
# plt.plot(t, Y, label='Y')
# plt.plot(t, Z, label='Z')
# plt.tight_layout()
