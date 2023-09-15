#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyResponse.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 14:55:20
# ==================================

import numpy as np
from Constants import YRSID_SI
from utils import dot_arr, cal_zeta, sYlm
from pyOrbits import detectors
from pyWaveForm import BasicWaveform


class FDResponse:
    """
    Response in the frequency domain
    --------------------------------
    """

    def __init__(self, pars, det='TQ', initial_T=False):
        self.wf = BasicWaveform(**pars)  # TODO
        self.orbit = detectors[det]()
        if initial_T:
            if det == "TQ":
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

        exp12 = np.exp(1j*np.pi*freq*(self.LT+kp1p2))
        exp23 = np.exp(1j*np.pi*freq*(self.LT+kp2p3))
        exp31 = np.exp(1j*np.pi*freq*(self.LT+kp3p1))

        sinc32 = np.sinc(prefact*(1.-kn1))
        sinc23 = np.sinc(prefact*(1.+kn1))
        sinc13 = np.sinc(prefact*(1.-kn2))
        sinc31 = np.sinc(prefact*(1.+kn2))
        sinc21 = np.sinc(prefact*(1.-kn3))
        sinc12 = np.sinc(prefact*(1.+kn3))

        prefacts = -1j * prefact
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


class TDResponse(object):
    """
    Response in the time domain
    ---------------------------
    parameter:
    - pars: dict for gravitational wave parameters
    """

    def __init__(self, pars, time, det='TQ'):
        self.wf = BasicWaveform(**pars)  # TODO
        self.orbit = detectors[det](time)
        self.LT = self.orbit.armLength

    def H(self, tf, nl):
        """
        Calculate n^i h_{ij} n^j
        ------------------------
        Parameters
        ----------
        - tf: time array
        - nl: unit vector from sender to receiver

        Return
        ------
        -
        """
        u = self.wf.vec_u
        v = self.wf.vec_v
        hpssb, hcssb = self.wf.get_hphc(tf)
        tf_size = tf.shape[0]
        h_size = hpssb.shape[0]
        if tf_size > h_size:
            hp = np.zeros_like(tf)
            hc = np.zeros_like(tf)
            hp[:h_size] = hpssb
            hc[:h_size] = hcssb
        elif tf_size < h_size:
            hp = hpssb[-tf_size:]
            hc = hcssb[-tf_size:]
        else:
            hp = hpssb
            hc = hcssb

        xi_p, xi_c = cal_zeta(u, v, nl)
        return hp*xi_p+hc*xi_c

    def Evaluate_yslr(self, tf, TDIgen=1):
        if TDIgen == 1:
            TDIdelay = 4

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

        k = self.wf.vec_k

        kp1 = dot_arr(k, p1)
        kn1 = dot_arr(k, n1)
        kp2 = dot_arr(k, p2)
        kn2 = dot_arr(k, n2)
        kp3 = dot_arr(k, p3)
        kn3 = dot_arr(k, n3)

        H3_p2 = {}
        H3_p1 = {}
        H1_p3 = {}
        H1_p2 = {}
        H2_p3 = {}
        H2_p1 = {}

        tt = [tf-kp1, tf-kp2, tf-kp3]

        for i in range(TDIdelay+1):
            tag = self.LT*i
            H3_p2[i] = self.H(tf-kp2-tag, n3)
            H3_p1[i] = self.H(tf-kp1-tag, n3)
            H1_p3[i] = self.H(tf-kp3-tag, n1)
            H1_p2[i] = self.H(tf-kp2-tag, n1)
            H2_p3[i] = self.H(tf-kp3-tag, n2)
            H2_p1[i] = self.H(tf-kp1-tag, n2)

        yslr = {}
        y12 = {}
        y23 = {}
        y31 = {}
        y21 = {}
        y32 = {}
        y13 = {}

        for i in range(TDIdelay):
            y12["%sL" % i] = (H3_p1[i+1]-H3_p2[i])/2/(1+kn3)
            y21["%sL" % i] = (H3_p2[i+1]-H3_p1[i])/2/(1-kn3)

            y23["%sL" % i] = (H1_p2[i+1]-H1_p3[i])/2/(1+kn1)
            y32["%sL" % i] = (H1_p3[i+1]-H1_p2[i])/2/(1-kn1)

            y31["%sL" % i] = (H2_p3[i+1]-H2_p1[i])/2/(1+kn2)
            y13["%sL" % i] = (H2_p1[i+1]-H2_p3[i])/2/(1-kn2)

        yslr[(1, 2)] = y12
        yslr[(2, 1)] = y21
        yslr[(2, 3)] = y23
        yslr[(3, 2)] = y32
        yslr[(3, 1)] = y31
        yslr[(1, 3)] = y13

        return yslr
