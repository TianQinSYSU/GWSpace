#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: GCB.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 10:29:40
#==================================

import numpy as np
from Constants import *

class GCBWaveform:
    '''
    This is Waveform for GCB.
    ------------------------
    Parameters:
    - Mc: chirp mass
    - DL: luminsity distance
    - phi0: initial phase at t = 0
    - f0: frequency of the source
    - fdot: derivative of frequency: df/dt
        - default: None, calculated physically
    - fddot: double derivative of frequency: d^2f/dt^2
        - default: None, calculated physically
    --------------------------
    How to call it:
    ```python
    ```
    tf = np.arange(0,Tobs, delta_T)
    GCB = GCBWaveform(Mc=0.5, DL=0.3, phi0=0, f0=0.001)
    hpS, hcS = GCB(tf)
    '''
    
    def __init__(self, Mc, DL, phi0, f0, fdot=None, fddot=None):
        self.f0 = f0
        #self.fdot = fdot
        if fdot == None:
            self.fdot = (96/5 * PI**(8/3) * 
                          (G_SI * Mc * MSUN_SI/C_SI**3)**(5/3)
                          * f0**(11/3) )
        else:
            self.fdot = fdot
        if fddot == None:
            self.fddot = 11/3 * self.fdot**2/f0
        else:
            self.fddot = fddot
        self.amp = 2 * (G_SI * Mc * MSUN_SI)**(5/3)
        self.amp = self.amp / C_SI**4 / (DL * MPC_SI)
        self.amp = self.amp * (PI * f0)**(2/3)
        self.phi0 = phi0
        
    def __call__(self, t):
        phase = 2* PI * (self.f0 + 0.5 * self.fdot *t + 
                           1/6 * self.fddot * t*t) *t + self.phi0
        hp = self.amp * np.cos( phase )
        hc = self.amp * np.sin( phase )
        return (hp, hc)

if __name__ == '__main__':
    print("This is waveform for GCB")
    GCB = GCBWaveform(Mc=0.5, DL=0.3, phi0=0, f0=0.001)
    print(f"Amplitude of GW is: {GCB.amp}")
    print(f'''Frequency and its derivative are:
            f0    = {GCB.f0}
            fdot  = {GCB.fdot},
            fddot = {GCB.fddot}''')

    Tobs = 10000 #YRSID_SI / 4
    delta_f = 1/Tobs
    
    delta_T = 1
    f_max = 1/(2*delta_T)
    
    tf = np.arange(0,Tobs, delta_T)
    
    gcb_hp, gcb_hc = GCB(tf)

    import matplotlib.pyplot as plt

    plt.plot(tf, gcb_hp, 'r-', label=r'$h_+^{\rm S}$')
    plt.plot(tf, gcb_hc, 'b--', label=r'$h_\times^{\rm S}$')

    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")

    plt.legend(loc="best")

    plt.show()
