#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: TDI.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-11 23:11:40
# ==================================

import numpy as np


def TDI_XYZ2AET(X, Y, Z):
    """
    Calculate AET channel from XYZ
    """
    A = 1/np.sqrt(2)*(Z-X)
    E = 1/np.sqrt(6)*(X-2*Y+Z)
    T = 1/np.sqrt(3)*(X+Y+Z)
    return A, E, T


def XYZ_TD(yslr, TDIgen=1):
    """
    Generate TDI XYZ in the TDIgen generation
    -----------------------------------------
    Parameters
    ----------
    """
    y31 = yslr[(3, 1)]
    y13 = yslr[(1, 3)]
    y12 = yslr[(1, 2)]
    y21 = yslr[(2, 1)]
    y23 = yslr[(2, 3)]
    y32 = yslr[(3, 2)]

    if TDIgen == 1:
        X = (y31["0L"]+y13["1L"]+y21["2L"]+y12["3L"]
             - y21["0L"]-y12["1L"]-y31["2L"]-y13["3L"])
        Y = (y12["0L"]+y21["1L"]+y32["2L"]+y23["3L"]
             - y32["0L"]-y23["1L"]-y12["2L"]-y21["3L"])
        Z = (y23["0L"]+y32["1L"]+y13["2L"]+y31["3L"]
             - y13["0L"]-y31["1L"]-y23["2L"]-y32["3L"])
    elif TDIGen == 2:
        X = 
    else:
        raise NotImplementedError

    return X, Y, Z


def AET_TD(yslr, TDIgen=1):
    X, Y, Z = XYZ_TD(yslr, TDIgen)
    A, E, T = TDI_XYZ2AET(X, Y, Z)
    return A, E, T


def XYZ_FD(yslr, freq, LT, TDIgen=1):
    """
    Calculate XYZ from yslr in frequency domain
    -------------------------------------------
    Parameters:
    - yslr: single link response of GW
    - freq: frequency
    - LT: arm length
    """

    Dt = np.exp(2j*np.pi*freq*LT)
    Dt2 = Dt*Dt

    X = yslr[(3, 1)]+Dt*yslr[(1, 3)]-yslr[(2, 1)]-Dt*yslr[(1, 2)]
    Y = yslr[(1, 2)]+Dt*yslr[(2, 1)]-yslr[(3, 2)]-Dt*yslr[(2, 3)]
    Z = yslr[(2, 3)]+Dt*yslr[(3, 2)]-yslr[(1, 3)]-Dt*yslr[(3, 1)]

    return np.array([X, Y, Z])*(1.-Dt2)


def AET_FD(yslr, freq, LT, TDIgen=1):
    """
    Calculate AET from yslr in frequency domain
    -------------------------------------------
    Parameters:
    - yslr: single link response of GW
    - freq: frequency
    - LT: arm length
    """
    Dt = np.exp(2j*np.pi*freq*LT)
    Dt2 = Dt*Dt

    A = ((1+Dt)*(yslr[(3, 1)]+yslr[(1, 3)])
         - yslr[(2, 3)]-Dt*yslr[(3, 2)]
         - yslr[(2, 1)]-Dt*yslr[(1, 2)])
    E = ((1-Dt)*(yslr[(1, 3)]-yslr[(3, 1)])
         + (1+2*Dt)*(yslr[(2, 1)]-yslr[(2, 3)])
         + (2+Dt)*(yslr[(1, 2)]-yslr[(3, 2)]))
    T = (1-Dt)*(yslr[(1, 3)]-yslr[(3, 1)]
                + yslr[(2, 1)]-yslr[(1, 2)]
                + yslr[(3, 2)]-yslr[(2, 3)])

    A = 1/np.sqrt(2)*(Dt2-1)*A
    E = 1/np.sqrt(6)*(Dt2-1)*E
    T = 1/np.sqrt(3)*(Dt2-1)*T

    return np.array([A, E, T])
