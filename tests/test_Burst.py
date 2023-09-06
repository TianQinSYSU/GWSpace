#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_Burst.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 14:21:33
#==================================

import numpy as np
from active_python_path import csgwsim as gws
#from csgwsim import BurstWaveform

if __name__ == "__main__":
    print("This is a test of Burst")
    Burst = gws.BurstWaveform(amp=0.5, tau=1000, fc=0.001, tc=5000)

    Tobs = 10000  # YRSID_SI / 4
    delta_f = 1/Tobs

    delta_T = 1
    f_max = 1/(2*delta_T)

    tf = np.arange(0, Tobs, delta_T)

    hp, hc = Burst(tf)

    import matplotlib.pyplot as plt

    plt.plot(tf, hp, 'r-', label=r'$h_+^{\rm S}$')
    plt.plot(tf, hc, 'b--', label=r'$h_\times^{\rm S}$')

    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")

    plt.legend(loc="best")

    plt.show()
