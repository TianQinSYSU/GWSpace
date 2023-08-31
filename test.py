#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-25 10:54:38
#==================================

import sys
from utils import yaml_readinpars
import numpy as np

import sys, os
try:
    from FastEMRI import *
except:
    few_path = './Waveforms/FastEMRIWaveforms'
    abs_few_path = os.path.abspath(few_path)
    
    if abs_few_path not in sys.path:
        sys.path.append(abs_few_path)
        
    from FastEMRI import *

from EMRI import EMRIWaveform

if __name__ == "__main__":
    print("This is a test of loading EMRI waveform")
    # parameters
    T = 0.01  # years
    dt = 15.0  # seconds

    pars = {"type": "EMRI",
            'M': 1e6,
            'a': 0.1,
            'mu': 1e1,
            'p0': 12.0,
            'e0': 0.2,
            'x0': 1.0,
            'qK': 0.2,
            'phiK': 0.2,
            'qS': 0.3,
            'phiS': 0.3,
            'dist': 1.0,
            'Phi_phi0': 1.0,
            'Phi_theta0': 2.0,
            'Phi_r0': 3.0,
          }
    M = pars['M']
    mu = pars['mu']
    a = pars['a']
    p0 = pars['p0']
    e0 = pars['e0']
    x0 = pars['x0']
    dist = pars['dist']
    
    qS = pars['qS']
    phiS = pars['phiS']
    qK = pars['qK']
    phiK = pars['phiK']
    Phi_phi0 = pars['Phi_phi0']
    Phi_theta0 = pars['Phi_theta0']
    Phi_r0 = pars['Phi_r0']

    
    emri = EMRIWaveform(M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK,
            Phi_phi0=0, Phi_theta0=0, Phi_r0=0)

    tf = np.arange(0, T * YRSID_SI, dt)

    hp, hc = emri(T, dt)

    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(tf[:2000], hp[:2000])
    plt.plot(tf[:2000], hc[:2000])

    plt.show()
    





