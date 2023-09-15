#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: cosmology.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-15 10:00:09
#==================================

import numpy as np
from Constants import C_SI, H0_SI, Omega_m

def luminosity_distance_approx(z, Omega_m = Omega_m):
    '''
    An analytical approxmimation of the luminosity distance
    in flat cosmologies
    -------------------------
    axiv:1111.6396
    '''
    x_z = (1-Omega_m)/Omega_m /(1+z)**3
    Phix = lambda x: ((1+320*x +0.4415 *x*x + 0.02656 * x**3) / 
            (1+392*x+0.5121*x*x + 0.03944*x**2))
    return 2*C_SI/H0_SI * (1+z)/np.sqrt(Omega_m) * (
            Phix(0) - Phix(x_z)/np.sqrt(1+z))


