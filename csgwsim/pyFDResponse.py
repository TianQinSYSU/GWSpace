#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyFDResponse.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 14:55:20
# ==================================

# import numpy as np
# from Constants import *
from utils import dot_arr, cal_zeta, sYlm
from pyOrbits import Orbit
from pyWaveForm import *


class FDResponse:
    """
    Response in the frequency domain
    --------------------------------
    """

    def __init__(self, pars, INI, initial_T=False):
        self.wf = BasicWaveform(pars)
        self.orbit = Orbit(INI)
        if initial_T:
            if INI.detector == "TianQin":
                dt = 3600
            tt = np.arange(-7*dt, YRSID_SI+7*dt, dt)
            ret = self.orbit.get_position(tt)

        self.LT = self.orbit.armLT

        self.u = self.wf.vec_u
        self.v = self.wf.vec_v
        self.k = self.wf.vec_k

    def EvaluateGslr(self, tf, freq):

        k = self.k

        p0 = self.orbit.get_position_px(tf, pp="p0")
        p1L, p2L, p3L = self.orbit.get_position_px(tf, pp="all")
        # p2L = self.orbit.get_position_px(tf, pp="p2")
        # p3L = self.orbit.get_position_px(tf, pp="p3")

        L = self.LT
        n1 = (p2L-p3L)/L
        n2 = (p3L-p1L)/L
        n3 = (p1L-p2L)/L
        p1 = p0+p1L
        p2 = p0+p2L
        p3 = p0+p3L

        kn1 = dot_arr(k, n1)
        kn2 = dot_arr(k, n2)
        kn3 = dot_arr(k, n3)

        # n1Hn1 = dot_arr_H_arr(n1, H, n1)
        # n2Hn2 = dot_arr_H_arr(n2, H, n2)
        # n3Hn3 = dot_arr_H_arr(n3, H, n3)
        zeta = {}
        zeta['p3'], zeta['c3'] = cal_zeta(self.u, self.v, n3)
        zeta['p2'], zeta['c2'] = cal_zeta(self.u, self.v, n2)
        zeta['p1'], zeta['c1'] = cal_zeta(self.u, self.v, n1)

        kp1p2 = dot_arr(k, (p1+p2))
        kp2p3 = dot_arr(k, (p2+p3))
        kp3p1 = dot_arr(k, (p3+p1))

        prefact = np.pi*freq*self.LT

        exp12 = np.exp(1j*(prefact+kp1p2))
        exp23 = np.exp(1j*(prefact+kp2p3))
        exp31 = np.exp(1j*(prefact+kp3p1))

        sinc32 = np.sinc(prefact*(1.-kn1))
        sinc23 = np.sinc(prefact*(1.+kn1))
        sinc13 = np.sinc(prefact*(1.-kn2))
        sinc31 = np.sinc(prefact*(1.+kn2))
        sinc21 = np.sinc(prefact*(1.-kn3))
        sinc12 = np.sinc(prefact*(1.+kn3))

        prefacts = 1j/2.*prefact
        yy12 = prefacts*exp12  # * n3Hn3
        yy23 = prefacts*exp23  # * n1Hn1
        yy31 = prefacts*exp31  # * n2Hn2

        Gslr = {(3, 2): yy23*sinc32,
                (2, 3): yy23*sinc23,
                (1, 3): yy31*sinc13,
                (3, 1): yy31*sinc31,
                (2, 1): yy12*sinc21,
                (1, 2): yy12*sinc12}

        return Gslr, zeta

    def Evaluate_yslr(self, freq, mode=[(2, 2)]):
        """
        Calculate yslr for all the modes
        --------------------------------
        - h: h(f) --> h_22 = h[(2,2)]
        """
        exp_2psi = np.exp(-1j*2*self.wf.psi)
        exp2psi = np.exp(1j*2*self.wf.psi)

        amp, phase, tf, dtf = self.wf.amp_phase(freq, mode)
        yslr = {}

        for lm in mode:
            l, m = lm

            hlm = np.exp(1j*phase[lm])  # without amp
            hl_m = np.exp(-1j*phase[lm])  # without amp
            Gslr, zeta = self.EvaluateGslr(tf[lm], freq)

            ylm = sYlm(-2, l, m, self.wf.iota, self.wf.varphi)
            yl_m = sYlm(-2, l, -m, self.wf.iota, self.wf.varphi)

            def niPlxni(i, y1, y2):
                zp = zeta['p%s' % i]
                zc = zeta['c%s' % i]
                return 0.5*(y1*exp_2psi*(zp+1j*zc)
                            + (-1)**l*y2*exp2psi*(zp-1j*zc))

            # n1Plmn1 = 0.5 * (ylm * exp_2psi * (zeta['p1'] + 1j * zeta['c1'])
            #        + (-1)**l * yl_m * exp2psi * (zeta['p1'] -1j * zeta['c1']))
            # n1pl_mn1 = 0.5 * (yl_m * exp_2psi * (zeta['p1'] + 1j * zeta['c1'])
            #        + (-1)**l * ylm * exp2psi * (zeta['p1'] -1j * zeta['c1']))

            n1Plmn1 = niPlxni(1, ylm, yl_m)
            n1Pl_mn1 = niPlxni(1, yl_m, ylm)
            n2Plmn2 = niPlxni(2, ylm, yl_m)
            n2Pl_mn2 = niPlxni(2, yl_m, ylm)
            n3Plmn3 = niPlxni(3, ylm, yl_m)
            n3Pl_mn3 = niPlxni(3, yl_m, ylm)

            yslr[lm] = {}
            yslr[lm][(1, 2)] = Gslr[(1, 2)]*(n3Plmn3*hlm+n3Pl_mn3*hl_m)*amp[lm]
            yslr[lm][(2, 1)] = Gslr[(2, 1)]*(n3Plmn3*hlm+n3Pl_mn3*hl_m)*amp[lm]

            yslr[lm][(2, 3)] = Gslr[(2, 3)]*(n1Plmn1*hlm+n1Pl_mn1*hl_m)*amp[lm]
            yslr[lm][(3, 2)] = Gslr[(3, 2)]*(n1Plmn1*hlm+n1Pl_mn1*hl_m)*amp[lm]

            yslr[lm][(3, 1)] = Gslr[(3, 1)]*(n2Plmn2*hlm+n2Pl_mn2*hl_m)*amp[lm]
            yslr[lm][(1, 3)] = Gslr[(1, 3)]*(n2Plmn2*hlm+n2Pl_mn2*hl_m)*amp[lm]
        return yslr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    from pyINIDetectors import INITianQin
    from TDI import XYZ_FD, AET_FD

    print("This is a test for frequency domain response")

    print("Testing of BHB waveform")

    pars = {"type": "BHB",
            "m1": 3.5e6,
            "m2": 2.1e5,
            "chi1": 0.2,
            "chi2": 0.1,
            "DL": 1e3,
            "phic": 0.0,
            "MfRef_in": 0,
            "psi": 0.2,
            "iota": 0.3,
            "lambda": 0.4,
            "beta": 1.2,
            "tc": 0,
            }

    TQ = INITianQin()
    fd = FDResponse(pars, TQ)

    NF = 10240
    freq = 10**np.linspace(-4, 0, NF)

    BHBwf = BasicWaveform(pars)

    amp, phase, tf, tfp = BHBwf.amp_phase(freq)

    st = time.time()

    Gslr, zeta = fd.EvaluateGslr(tf[(2, 2)], freq)  # fd.Evaluate_yslr(freq)
    yslr_ = fd.Evaluate_yslr(freq)  # fd.Evaluate_yslr(freq)
    ed = time.time()

    print(f"time cost for the fd response is {ed-st} s")

    mode = [(2, 2)]
    ln = [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)]

    for ll in ln:
        plt.figure()
        gg = Gslr[ll]
        yy = yslr_[mode[0]][ll]
        plt.plot(freq, gg, '-r')
        plt.plot(freq, yy, '--b')
        plt.title(ll)

        plt.xscale('log')

    X, Y, Z = XYZ_FD(yslr_[(2, 2)], freq, LT=fd.LT)
    A, E, T = AET_FD(yslr_[(2, 2)], freq, fd.LT)

    plt.figure()
    plt.loglog(freq, np.abs(X), '-r', label='X')
    plt.loglog(freq, np.abs(Y), '--g', label='Y')
    plt.loglog(freq, np.abs(Z), ':b', label='Z')

    plt.xlabel('f')
    plt.ylabel('X,Y,Z')
    plt.legend(loc='best')

    plt.figure()
    plt.loglog(freq, np.abs(A), '-r', label='A')
    plt.loglog(freq, np.abs(E), '--g', label='E')
    plt.loglog(freq, np.abs(T), ':b', label='T')

    plt.xlabel('f')
    plt.ylabel('A,E,T')
    plt.legend(loc='best')

    plt.show()
