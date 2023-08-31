#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: Burst.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-31 16:36:40
#==================================

import numpy as np
from Constants import *

class BurstWaveform:
    '''
    A sin-Gaussian waveforms for pooly modelled burst source
    --------------------------------------------------------
    '''

    def __init__(self, amp, tau, fc, tc=0):
        self.amp = amp
        self.tau = tau
        self.fc = fc
        self.tc = tc

    def __call__(self, tf):
        t = tf - self.tc
        h = (2/np.pi)**(0.25) * self.tau**(-0.5) * self.amp 
        h *= np.exp(- (t / self.tau)**2) * np.exp(2j * np.pi * self.fc * t)
        return (h.real, h.imag)


if __name__ == "__main__":
    print("This is a test of Burst")
    Burst = BurstWaveform(amp=0.5, tau=1000, fc=0.001, tc=5000)

    Tobs = 10000 #YRSID_SI / 4
    delta_f = 1/Tobs
    
    delta_T = 1
    f_max = 1/(2*delta_T)
    
    tf = np.arange(0,Tobs, delta_T)
    
    hp, hc = Burst(tf)

    import matplotlib.pyplot as plt

    plt.plot(tf, hp, 'r-', label=r'$h_+^{\rm S}$')
    plt.plot(tf, hc, 'b--', label=r'$h_\times^{\rm S}$')

    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")

    plt.legend(loc="best")

    plt.show()
