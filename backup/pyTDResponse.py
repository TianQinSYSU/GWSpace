#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: pyTDResponse.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 14:55:07
#==================================

import numpy as np
from utils import dot_arr
from Constants import *
from pyOrbits import Orbit
from pyWaveForm import *


def cal_xi(u,v,nl):
    '''
    Calculate xi^+ and xi^x
    ------------------------------
    Parameters
    ----------
    - u, v: polarization coordinates
    - nl: unit vector from sender to receiver 

    Return
    ------
    - n otimes (\epsilon_+ \epsilon_x) otimes n
    '''
    xi_p = (dot_arr(u, nl))**2 - (dot_arr(v, nl))**2
    xi_c = 2 * dot_arr(u, nl) * dot_arr(v, nl)
    return (xi_p, xi_c)

class TDResponse:
    '''
    Response in the time domain
    ---------------------------
    parameter:
    - pars: dict for gravitational wave parameters
    - INI: class of initial detector parameters
    '''
    
    def __init__(self, pars, INI):
        self.wf = WaveForm(pars)
        self.orbit = Orbit(INI)
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
                
        xi_p, xi_c = cal_xi(u,v, nl)
        return hpssb * xi_p + hcssb * xi_c
        
    
    def y_gw_sr(self, tf, send, rece):
        '''
        Calculate the relative frequency deviation
        ------------------------------------------
        Parameters
        ----------
        - tf: time array
        - send: int sender number
        - rece: int receiver number
        
        Return
        ------
        - (Hs - Hr)/2/(1- k \cdot nl)
        '''
        k = self.wf.k
        
        ps = self.orbit.get_pos(tf, 'p%s'%send)
        ts = tf - self.LT
        pr = self.orbit.get_pos(ts, 'p%s'%rece)
                
        kps = dot_arr(k, ps)
        kpr = dot_arr(k, pr)
        
        nl = (pr - ps) / self.LT
        knl = dot_arr(k, nl)
        
        Hs = self.H(ts - kps, nl)
        Hr = self.H(tf - kpr, nl)
        
        return 0.5 * (Hs - Hr)/(1 - knl)

    def generate_TDI_tags(self, tf, TDIdelay=3):
        '''
        Calculate TDI tags
        ------------------
        Parameters
        ----------
        - tf: time array
        - TDIdelay: int number for the time delay
            - default: 3 ==> for the first generation TDI

        Return
        ------
        - (y12, y23, y31, y21, y32, y13)
        '''
        
        tags = [i * self.LT for i in range(TDIdelay+1)]
        y12 = {}; y23 = {}; y31 = {}
        y21 = {}; y32 = {}; y13 = {}
        
        for i, tag in enumerate(tags):
            ts = tf - tag
            y12["%sL"%i] = self.y_gw_sr(ts, 1, 2)
            y23["%sL"%i] = self.y_gw_sr(ts, 2, 3)
            y31["%sL"%i] = self.y_gw_sr(ts, 3, 1)
            y21["%sL"%i] = self.y_gw_sr(ts, 2, 1)
            y32["%sL"%i] = self.y_gw_sr(ts, 3, 2)
            y13["%sL"%i] = self.y_gw_sr(ts, 1, 3)

        return (y12, y23, y31, y21, y32, y13)
        
#"""        
#    def TDI_XYZ(self, tf):
#        
#        tags = [0, self.LT, 2* self.LT, 3 * self.LT]
#        y12 = {}; y23 = {}; y31 = {}
#        y21 = {}; y32 = {}; y13 = {}
#        
#        for i, tag in enumerate(tags):
#            ts = tf - tag
#            y12["%sL"%i] = self.y_gw_sr(ts, 1, 2)
#            y23["%sL"%i] = self.y_gw_sr(ts, 2, 3)
#            y31["%sL"%i] = self.y_gw_sr(ts, 3, 1)
#            y21["%sL"%i] = self.y_gw_sr(ts, 2, 1)
#            y32["%sL"%i] = self.y_gw_sr(ts, 3, 2)
#            y13["%sL"%i] = self.y_gw_sr(ts, 1, 3)
#        
#        X = (y31["0L"] + y13["1L"] + y21["2L"] + y12["3L"]
#             - y21["0L"] - y12["1L"] - y31["2L"] - y13["3L"])
#        Y = (y12["0L"] + y21["1L"] + y32["2L"] + y23["3L"]
#             - y32["0L"] - y23["1L"] - y12["2L"] - y21["3L"])
#        Z = (y23["0L"] + y32["1L"] + y13["2L"] + y31["3L"]
#             - y13["0L"] - y12["1L"] - y31["2L"] - y13["3L"])
#        return (X,Y,Z)
#        
#"""

if __name__ == "__main__":
    print("This is TD response generation code")

    Tobs = 10000 #YRSID_SI / 4
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

    y12 = td.y_gw_sr(tf[:1000000], 1,2)
    #y12.shape

    import matplotlib.pyplot as plt

    plt.plot(tf, y12, 'r-', label=r'$y_{12}$')

    plt.xlabel("Time")
    plt.ylabel(r"$y^{GW}_{12}$")

    plt.legend(loc="best")

    #=====================
    import time

    st = time.time()
    yslr = td.generate_TDI_tags(tf)
    ed = time.time()

    print("Time cost is %f s for 1000 points"%(ed - st))

    plt.figure()

    for i in range(6):
        plt.plot(tf, yslr[i]["1L"])



    plt.show()

