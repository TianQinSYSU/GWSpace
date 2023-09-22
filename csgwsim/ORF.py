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
from .Orbits import get_pos
from .utils import dot_arr, get_uvk


def transfer_Dslr(tf, freq, lambd, beta, detector="TianQin", nopol=True):
    u,v,k = get_uvk(lambd, beta)

    xt.yt,zt,LT = get_pos(tf, detector, toT=True)

    nij = lambda i,j: np.array([xt[j-1]-xt[i-1], yt[j-1]-yt[i-1], zt[j-1]-zt[i-1]])/LT
    n21 = nij(2,1);  n13 = nij(1,3);  n32 = nij(3,2)

    kri = lambda i: dot_arr(k, np.array([xt[i-1],yt[i-1],zt[i-1]]))
    kr1 = kri(1);  kr2 = kri(2);  kr3 = kri(3)

    _xip = lambda n: dot_arr(n, u)**2 - dot_arr(n, v)**2
    _xic = lambda n: 2*dot_arr(n, u)*dot_arr(n, v)

    if nopol:
        xi12 = _xip(-n21) + _xic(-n12)
        xi21 = _xip(n21) + _xic(n21)
        xi13 = _xip(n13) + _xic(n13)
        xi31 = _xip(-n13) + _xic(-n13)
        xi32 = _xip(n32) + _xic(n32)
        xi23 = _xip(-n32) + _xic(-n32)
    else:
        NotImplementedError

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
            (1,2): fac*sinc12*exp12*xi12,
            (2,1): fac*sinc21*exp12*xi21,
            (2,3): fac*sin23*exp23*xi23,
            (3,2): fac*sinc32*exp23*xi32,
            (3,1): fac*sinc31*exp31*xi31,
            (1,3): fac*sinc13*exp13*xi13,
            }


    return yslr





