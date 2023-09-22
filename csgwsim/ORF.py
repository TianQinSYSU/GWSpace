#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: ORF.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-21 15:45:38
#==================================

import numpy as np
from .Constants import PI
from .Orbit import get_pos
from .utils import dot_arr, get_uvk, cal_zeta


def transfer_Dslr(tf, freq, lambd, beta, detector="TianQin"):
    '''
    Calculate the transfer function of single arm
    ---------------------------------------------
    - tf: time
    - freq: frequency
    - lambd: 
    - beta: position of source in SSB coordinate
    - detector: 
    '''
    u,v,k = get_uvk(lambd, beta)

    xt,yt,zt,LT = get_pos(tf, detector, toT=True)

    nij = lambda i,j: np.array([xt[j-1]-xt[i-1], yt[j-1]-yt[i-1], zt[j-1]-zt[i-1]])/LT
    n21 = nij(2,1);  n13 = nij(1,3);  n32 = nij(3,2)
    kn21 = dot_arr(k, n21)
    kn13 = dot_arr(k, n13)
    kn32 = dot_arr(k, n32)

    kri = lambda i: dot_arr(k, np.array([xt[i-1],yt[i-1],zt[i-1]]))
    kr1 = kri(1);  kr2 = kri(2);  kr3 = kri(3)

    sinc12 = np.sinc(freq*LT*(1+kn21))
    sinc21 = np.sinc(freq*LT*(1-kn21))
    sinc31 = np.sinc(freq*LT*(1+kn13))
    sinc13 = np.sinc(freq*LT*(1-kn13))
    sinc23 = np.sinc(freq*LT*(1+kn32))
    sinc32 = np.sinc(freq*LT*(1-kn32))

    exp12 = np.exp(-1j*PI*freq*(LT+kr1+kr2))
    exp23 = np.exp(-1j*PI*freq*(LT+kr2+kr3))
    exp31 = np.exp(-1j*PI*freq*(LT+kr3+kr1))

    fac = -1j*2*np.pi*freq*LT
    yslr = {
            (1,2): fac*sinc12*exp12,
            (2,1): fac*sinc21*exp12,
            (2,3): fac*sinc23*exp23,
            (3,2): fac*sinc32*exp23,
            (3,1): fac*sinc31*exp31,
            (1,3): fac*sinc13*exp31,
            }

    zeta_p12, zeta_c12 = cal_zeta(u,v, n21)
    zeta_p23, zeta_c23 = cal_zeta(u,v, n32)
    zeta_p31, zeta_c31 = cal_zeta(u,v, n13)

    zeta_l = {
            "p12": zeta_p12,
            "c12": zeta_c12,
            "p23": zeta_p23,
            "c23": zeta_c23,
            "p31": zeta_p31,
            "c31": zeta_c31,
            }

    return yslr, zeta_l

