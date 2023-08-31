#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: TDI.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-11 23:11:40
#==================================

import numpy as np
from Constants import *
from pyTDResponse import TDResponse
from pyFDResponse import FDResponse


def TDI_XYZ2AET(X,Y,Z):
    """
    Calculate AET channel from XYZ
    """
    A = 1/np.sqrt(2) *(Z-X)
    E = 1/np.sqrt(6) * (X-2*Y+Z)
    T = 1/np.sqrt(3) * (X+Y+Z)
    return (A,E,T)

def XYZ_TD(yslr, TDIgen=1):
    """
    Generate TDI XYZ in the TDIgen generation
    -----------------------------------------
    Parameters
    ----------
    """
    y31 = yslr[(3,1)]; y13 = yslr[(1,3)]
    y12 = yslr[(1,2)]; y21 = yslr[(2,1)]
    y23 = yslr[(2,3)]; y32 = yslr[(3,2)]

    if TDIgen == 1:
        X = (y31["0L"] + y13["1L"] + y21["2L"] + y12["3L"]
                - y21["0L"] - y12["1L"] - y31["2L"] - y13["3L"])
        Y = (y12["0L"] + y21["1L"] + y32["2L"] + y23["3L"]
                - y32["0L"] - y23["1L"] - y12["2L"] - y21["3L"])
        Z = (y23["0L"] + y32["1L"] + y13["2L"] + y31["3L"]
                - y13["0L"] - y31["1L"] - y23["2L"] - y32["3L"])
        
    return (X,Y,Z)

def AET_TD(yslr, TDIgen=1):
    X,Y,Z = XYZ_TD(yslr, TDIgen)
    A,E,T = TDI_XYZ2AET(X,Y,Z)
    return (A,E,T)

def XYZ_FD(yslr, freq, LT, TDIgen=1):
    '''
    Calculate XYZ from yslr in frequency domain
    -------------------------------------------
    Parameters:
    - yslr: single link response of GW
    - freq: frequency 
    - LT: arm length
    '''

    Dt = np.exp(2j * np.pi * freq * LT)
    Dt2 = Dt * Dt

    X = yslr[(3,1)] + Dt*yslr[(1,3)] - yslr[(2,1)] - Dt*yslr[(1,2)]
    Y = yslr[(1,2)] + Dt*yslr[(2,1)] - yslr[(3,2)] - Dt*yslr[(2,3)]
    Z = yslr[(2,3)] + Dt*yslr[(3,2)] - yslr[(1,3)] - Dt*yslr[(3,1)]

    return np.array([X,Y,Z]) * (1. - Dt2)

def AET_FD(yslr, freq, LT, TDIgen=1):
    '''
    Calculate AET from yslr in frequency domain
    -------------------------------------------
    Parameters:
    - yslr: single link response of GW
    - freq: frequency 
    - LT: arm length
    '''
    Dt = np.exp(2j * np.pi * freq * LT)
    Dt2 = Dt * Dt

    A = ( (1+Dt) * (yslr[(3,1)] + yslr[(1,3)]) 
            - yslr[(2,3)] - Dt*yslr[(3,2)] 
            - yslr[(2,1)] - Dt*yslr[(1,2)] )
    E = ( (1-Dt) * (yslr[(1,3)] - yslr[(3,1)]) 
            +(1+2*Dt) *(yslr[(2,1)] - yslr[(2,3)])
            +(2+Dt) * (yslr[(1,2)] - yslr[(3,2)]) )
    T = (1-Dt)*(yslr[(1,3)] - yslr[(3,1)]
            + yslr[(2,1)] - yslr[(1,2)]
            + yslr[(3,2)] - yslr[(2,3)] )

    A = 1/np.sqrt(2) * (Dt2 - 1) * A
    E = 1/np.sqrt(6) * (Dt2 - 1) * E
    T = 1/np.sqrt(3) * (Dt2 - 1) * T

    return np.array([A,E,T])


if __name__ == "__main__":
    print("This is TDI TD response generation code")

    Tobs = 4*DAY #YRSID_SI / 4
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

    X,Y,Z = XYZ_TD(yslr)

    plt.figure()

    plt.plot(tf, X, '-r')

    plt.figure()
    plt.plot(tf, Y, '--b')

    plt.figure()
    plt.plot(tf, Z, ':g')

    A,E,T = TDI_XYZ2AET(X,Y,Z)

    plt.figure()
    plt.plot(tf, A, '-r')

    plt.figure()
    plt.plot(tf, E, '--b')

    plt.figure()
    plt.plot(tf, T, ':g')

    plt.show()

