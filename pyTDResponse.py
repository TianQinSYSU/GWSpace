#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: pyTDResponse.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 14:55:07
#==================================

import numpy as np
from utils import dot_arr, cal_zeta
from Constants import *
from pyOrbits import Orbit
from pyWaveForm import *



class TDResponse:
    '''
    Response in the time domain
    ---------------------------
    parameter:
    - pars: dict for gravitational wave parameters
    - INI: class of initial detector parameters
    '''
    
    def __init__(self, pars, INI, initial_T=False):
        self.wf = WaveForm(pars)
        self.orbit = Orbit(INI)
        if initial_T:
            if INI.detector == 'TianQin':
                dt = 3600
            tt = np.arange(- 7*dt, YRSID_SI + 7*dt, dt)
            ret = self.orbit.get_position(tt)

        self.LT = self.orbit.armLT        
   
    def H(self, tf, nl):
        '''
        Calculate n^i h_{ij} n^j
        ------------------------
        Parameters
        ----------
        - tf: time array
        - nl: unit vector from sender to receiver

        Return
        ------
        - 
        '''
        u = self.wf.u
        v = self.wf.v
        hpssb, hcssb = self.wf(tf)
                
        xi_p, xi_c = cal_zeta(u,v, nl)
        return hpssb * xi_p + hcssb * xi_c
    
    def Evaluate_yslr(self, tf, TDIgen=1):
        if TDIgen == 1:
            TDIdelay = 4

        p0 = self.orbit.get_position_px(tf, pp="p0")
        p1L, p2L, p3L = self.orbit.get_position_px(tf, pp="all")
        #p2L = self.orbit.get_position_px(tf, pp="p2")
        #p3L = self.orbit.get_position_px(tf, pp="p3")

        L = self.LT
        n1 = (p2L - p3L)/L;  n2 = (p3L - p1L)/L; n3 = (p1L - p2L)/L
        p1 = p0 + p1L; p2 = p0 + p2L; p3 = p0 + p3L

        k = self.wf.k

        kp1 = dot_arr(k, p1); kn1 = dot_arr(k, n1)
        kp2 = dot_arr(k, p2); kn2 = dot_arr(k, n2)
        kp3 = dot_arr(k, p3); kn3 = dot_arr(k, n3)
                
        H3_p2 = {}; H3_p1 = {}
        H1_p3 = {}; H1_p2 = {}
        H2_p3 = {}; H2_p1 = {} 
        
        tt = [tf - kp1, tf - kp2, tf - kp3]

        
        for i in range(TDIdelay+1):
            tag = self.LT * i
            H3_p2[i] = self.H(tf - kp2 - tag, n3)
            H3_p1[i] = self.H(tf - kp1 - tag, n3)
            H1_p3[i] = self.H(tf - kp3 - tag, n1)
            H1_p2[i] = self.H(tf - kp2 - tag, n1)
            H2_p3[i] = self.H(tf - kp3 - tag, n2)
            H2_p1[i] = self.H(tf - kp1 - tag, n2)

        yslr = {}
        y12 = {}; y23 = {}; y31 = {}
        y21 = {}; y32 = {}; y13 = {}

        for i in range(TDIdelay):
            y12["%sL"%i] = (H3_p1[i+1] - H3_p2[i])/2/(1 + kn3)
            y21["%sL"%i] = (H3_p2[i+1] - H3_p1[i])/2/(1 - kn3)
            
            y23["%sL"%i] = (H1_p2[i+1] - H1_p3[i])/2/(1 + kn1)
            y32["%sL"%i] = (H1_p3[i+1] - H1_p2[i])/2/(1 - kn1)

            y31["%sL"%i] = (H2_p3[i+1] - H2_p1[i])/2/(1 + kn2)
            y13["%sL"%i] = (H2_p1[i+1] - H2_p3[i])/2/(1 - kn2)

        yslr[(1,2)] = y12; yslr[(2,1)] = y21
        yslr[(2,3)] = y23; yslr[(3,2)] = y32
        yslr[(3,1)] = y31; yslr[(1,3)] = y13

        return yslr
       
if __name__ == "__main__":
    print("This is TD response generation code")

    Tobs = 4* DAY #YRSID_SI / 4
    delta_f = 1/Tobs    
    delta_T = 1
    f_max = 1/(2*delta_T)
    
    tf = np.arange(0,Tobs, delta_T)

    print("Testing of GCB waveform")
    GCBpars = {"type": "GCB",
           "Mc": 0.5,
           "DL": 0.3,
           "phi0": 0.0,
           "f0": 0.001,
           "psi": 0.2,
           "iota": 0.3,
           "lambda": 0.4,
           "beta": 1.2,
          }

    print("Mc" in GCBpars.keys())
    
    #GCBwf = WaveForm(GCBpars)
    #hpssb, hcssb = GCBwf(tf)

    from pyINIDetectors import INITianQin

    TQ = INITianQin()
    td = TDResponse(GCBpars, TQ)

    import matplotlib.pyplot as plt
    import time

    st = time.time()
    yslr = td.Evaluate_yslr(tf)
    ed = time.time()

    print("Time cost is %f s for %d points"%(ed - st, tf.shape[0]))

    plt.figure()

    tags = [(1,2), (2,1), (2,3), (3,2), (3,1), (1,3)]

    for i, tag in enumerate(tags):
        for j in range(4):
            plt.figure(i*4 + j)
            plt.plot(tf, yslr[tag][f"{j}L"])
            plt.title(f"y_{tag} [j]L")

    plt.show()

