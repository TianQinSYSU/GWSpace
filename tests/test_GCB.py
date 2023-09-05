#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_GCB.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 14:22:52
#==================================

from csgwsim import GCBWaveform

if __name__ == '__main__':
    print("This is waveform for GCB")
    GCB = GCBWaveform(Mc=0.5, DL=0.3, phi0=0, f0=0.001)
    print(f"Amplitude of GW is: {GCB.amp}")
    print(f'''Frequency and its derivative are:
            f0    = {GCB.f0}
            fdot  = {GCB.fdot},
            fddot = {GCB.fddot}''')

    Tobs = 10000  # YRSID_SI / 4
    delta_f = 1/Tobs

    delta_T = 1
    f_max = 1/(2*delta_T)

    tf = np.arange(0, Tobs, delta_T)

    gcb_hp, gcb_hc = GCB(tf)

    import matplotlib.pyplot as plt

    plt.plot(tf, gcb_hp, 'r-', label=r'$h_+^{\rm S}$')
    plt.plot(tf, gcb_hc, 'b--', label=r'$h_\times^{\rm S}$')

    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")

    plt.legend(loc="best")

    plt.show()
