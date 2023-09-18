#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_Orbits.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:55:52
# ==================================

from active_python_path import csgwsim

import time
import numpy as np
import matplotlib.pyplot as plt

YRSID_SI = csgwsim.Constants.YRSID_SI
DAY = csgwsim.Constants.DAY

st = time.time()
Tobs = YRSID_SI
delta_T = 3600
tf = np.arange(0, Tobs, delta_T)

TQOrbit = csgwsim.TianQinOrbit(tf)
ed = time.time()
print(f"Time cost for initial position: {ed-st} s")

plt.figure()
plt.plot(tf/DAY, TQOrbit.orbit_1[0], label='x1')
plt.plot(tf/DAY, TQOrbit.orbit_2[0], label='x2')
plt.plot(tf/DAY, TQOrbit.orbit_3[0], label='x3')
plt.xlabel('Time [Day]')
plt.ylabel('pos (x) [s]')
plt.xlim(90, 115)
plt.ylim(480, 495)
plt.legend()
plt.tight_layout()

plt.figure()
for i in range(3):
    plt.plot(tf/DAY, TQOrbit.p_0[i])
plt.tight_layout()
