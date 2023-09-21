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
        xi13 = _xip(n13) + _xi
    else:
        NotImplementedError

    # in numpy sinc(x) = sin(pi x)/pi x
    fact_sinc = np.sinc(freq * LT *(1-kn))
    fact_exp = np.exp(-1j * PI*freq*(LT+krs+krr))
    return 0.5 * fact_sinc * fact_exp





