#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyResponse.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 14:55:20
# ==================================

import numpy as np
from numba import jit
from .utils import dot_arr, get_uvk, cal_zeta, sYlm
from .pyOrbits import detectors
from .pyWaveForm import waveforms


@jit(nopython=True)
def _matrix_res_pro(n, p, m):
    """TensorProduct(n, m) : P,  where A:B = A_ij B_ij"""
    return (n[0] * p[0, 0] * m[0] + n[0] * p[0, 1] * m[1] + n[0] * p[0, 2] * m[2]
            + n[1] * p[1, 0] * m[0] + n[1] * p[1, 1] * m[1] + n[1] * p[1, 2] * m[2]
            + n[2] * p[2, 0] * m[0] + n[2] * p[2, 1] * m[1] + n[2] * p[2, 2] * m[2])


def trans_fd_response(vec_k, p, det, f):
    """See Marsat et al. (Eq. 21, 28) https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011"""
    u12 = det.Uni_vec_12
    u23 = det.Uni_vec_23
    u13 = det.Uni_vec_13
    ls = det.L_T
    p_1 = det.orbit_1
    p_2 = det.orbit_2
    p_3 = det.orbit_3

    x = np.pi * f * ls
    # com_f = 1j/2 * x in (Marsat et al.) is because it was using the A,E,T which are 1/2 of their LDC definitions
    # See (Eq. 2) of McWilliams et al. https://journals.aps.org/prd/abstract/10.1103/PhysRevD.81.064014
    com_f = 1j * x

    vk12 = np.dot(vec_k, u12)
    vk23 = np.dot(vec_k, u23)
    vk13 = np.dot(vec_k, u13)

    # In numpy, the sinc function is sin(pi x)/(pi x)
    sin12 = np.sinc(f * ls * (1 - vk12))
    sin23 = np.sinc(f * ls * (1 - vk23))
    sin13 = np.sinc(f * ls * (1 - vk13))

    sin21 = np.sinc(f * ls * (1 + vk12))
    sin32 = np.sinc(f * ls * (1 + vk23))
    sin31 = np.sinc(f * ls * (1 + vk13))

    exp12 = np.exp(1j * np.pi * f * (ls + np.dot(vec_k, p_1+p_2)))
    exp23 = np.exp(1j * np.pi * f * (ls + np.dot(vec_k, p_2+p_3)))
    exp31 = np.exp(1j * np.pi * f * (ls + np.dot(vec_k, p_3+p_1)))

    y12_pre = com_f * sin12 * exp12
    y21_pre = com_f * sin21 * exp12
    y13_pre = com_f * sin13 * exp31
    y31_pre = com_f * sin31 * exp31
    y23_pre = com_f * sin23 * exp23
    y32_pre = com_f * sin32 * exp23

    def trans_response(p_):
        n12pn12 = _matrix_res_pro(u12, p_, u12)
        n23pn23 = _matrix_res_pro(u23, p_, u23)
        n31pn31 = _matrix_res_pro(u13, p_, u13)

        y12 = y12_pre * n12pn12
        y21 = y21_pre * n12pn12
        y13 = y13_pre * n31pn31
        y31 = y31_pre * n31pn31
        y23 = y23_pre * n23pn23
        y32 = y32_pre * n23pn23

        return y12, y21, y13, y31, y23, y32

    if type(p) != tuple:
        p = (p, )
    return tuple(trans_response(p_0) for p_0 in p)


def get_fd_response(vec_k, p, det, f, channel='A'):
    """

    :param vec_k:
    :param p: p is a tuple, i.e. a series of P_lm (or P_x, P_+) in it
    :param det:
    :param f:
    :param channel:
    :return:
    """
    # if channel not in 'XYZAET':  # <if not all([c in 'XYZAET' for c in channel])> for multi-channel mode
    #     raise ValueError(f"[SpaceResponse] Unknown channel {channel}. "
    #                      f"Supported channels: {'|'.join(['X', 'Y', 'Z', 'A', 'E', 'T'])}")
    yslr = trans_fd_response(vec_k, p, det, f)
    x = np.pi * f * det.L_T
    z = np.exp(2j * x)  # Time delay factor
    res_list = []

    for (y12, y21, y13, y31, y23, y32) in yslr:  # TODO: to be polished
        if channel == 'A':
            # See (Eq. 29) in arXiv:2003.00357v1, the factor of rescaling a,e,t to the original A,E,T
            factor_ae = 1j*np.sqrt(2)*np.sin(2.*x)*np.exp(2j*x)
            res = factor_ae * ((1.+z) * (y31 + y13) - y23 - z*y32 - y21 - z*y12)
        elif channel == 'E':
            factor_ae = 1j*np.sqrt(2)*np.sin(2.*x)*np.exp(2j*x)
            res = factor_ae * 1./np.sqrt(3) * ((1.-z)*(y13 - y31) + (2.+z)*(y12 - y32) + (1.+2*z)*(y21 - y23))
        elif channel == 'T':
            factor_t = 2.*np.sqrt(2)*np.sin(2.*x)*np.sin(x)*np.exp(3j*x)
            res = factor_t * np.sqrt(2/3) * (y21 - y12 + y32 - y23 + y13 - y31)
        else:
            raise ValueError(f"[SpaceResponse] Unknown channel {channel}. "
                             f"Supported channels: {'|'.join(['A', 'E', 'T'])}")
        res_list.append(res)

    return res_list


# def Evaluate_yslr(self, freq, mode=None):
#     """
#     Calculate yslr for all the modes
#     --------------------------------
#     - h: h(f) --> h_22 = h[(2,2)]
#     """
#     if mode is None:
#         mode = [(2, 2)]
#     exp_2psi = np.exp(-1j*2*self.wf.psi)
#     exp2psi = np.exp(1j*2*self.wf.psi)
#
#     amp, phase, tf = self.wf.get_amp_phase(freq, mode)
#     yslr = {}
#
#     for lm in mode:
#         l, m = lm
#
#         hlm = np.exp(1j*phase[lm])  # without amp
#         hl_m = np.exp(-1j*phase[lm])  # without amp
#         Gslr, zeta = self.EvaluateTslr(tf[lm], freq, self.wf.Lambda, self.wf.Beta)
#
#         ylm = sYlm(-2, l, m, self.wf.iota, self.wf.varphi)
#         yl_m = sYlm(-2, l, -m, self.wf.iota, self.wf.varphi)
#
#         def niPlxni(i, y1, y2):
#             zp = zeta['p%s' % i]
#             zc = zeta['c%s' % i]
#             return 0.5*(y1*exp_2psi*(zp+1j*zc)
#                         + (-1)**l*y2*exp2psi*(zp-1j*zc))
#
#         # n1Plmn1 = 0.5 * (ylm * exp_2psi * (zeta['p1'] + 1j * zeta['c1'])
#         #        + (-1)**l * yl_m * exp2psi * (zeta['p1'] -1j * zeta['c1']))
#         # n1pl_mn1 = 0.5 * (yl_m * exp_2psi * (zeta['p1'] + 1j * zeta['c1'])
#         #        + (-1)**l * ylm * exp2psi * (zeta['p1'] -1j * zeta['c1']))
#
#         n1Plmn1 = niPlxni(1, ylm, yl_m)
#         n1Pl_mn1 = niPlxni(1, yl_m, ylm)
#         n2Plmn2 = niPlxni(2, ylm, yl_m)
#         n2Pl_mn2 = niPlxni(2, yl_m, ylm)
#         n3Plmn3 = niPlxni(3, ylm, yl_m)
#         n3Pl_mn3 = niPlxni(3, yl_m, ylm)
#
#         yslr[lm] = {}
#         yslr[lm][(1, 2)] = Gslr[(1, 2)]*(n3Plmn3*hlm+n3Pl_mn3*hl_m)*amp[lm]
#         yslr[lm][(2, 1)] = Gslr[(2, 1)]*(n3Plmn3*hlm+n3Pl_mn3*hl_m)*amp[lm]
#
#         yslr[lm][(2, 3)] = Gslr[(2, 3)]*(n1Plmn1*hlm+n1Pl_mn1*hl_m)*amp[lm]
#         yslr[lm][(3, 2)] = Gslr[(3, 2)]*(n1Plmn1*hlm+n1Pl_mn1*hl_m)*amp[lm]
#
#         yslr[lm][(3, 1)] = Gslr[(3, 1)]*(n2Plmn2*hlm+n2Pl_mn2*hl_m)*amp[lm]
#         yslr[lm][(1, 3)] = Gslr[(1, 3)]*(n2Plmn2*hlm+n2Pl_mn2*hl_m)*amp[lm]
#     return yslr


class TDResponse(object):
    """
    Response in the time domain
    ---------------------------
    parameter:
    - pars: dict for gravitational wave parameters
    """

    def __init__(self, pars, time, wf='GCB', det='TQ'):
        self.wf = waveforms[wf](**pars)
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
        else:
            raise NotImplementedError

        p1, p2, p3 = self.orbit.orbit_1, self.orbit.orbit_2, self.orbit.orbit_3

        L = self.LT
        n1 = (p2-p3)/L
        n2 = (p3-p1)/L
        n3 = (p1-p2)/L

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
            y12[f"{i}L"] = (H3_p1[i+1]-H3_p2[i])/2/(1+kn3)
            y21[f"{i}L"] = (H3_p2[i+1]-H3_p1[i])/2/(1-kn3)

            y23[f"{i}L"] = (H1_p2[i+1]-H1_p3[i])/2/(1+kn1)
            y32[f"{i}L"] = (H1_p3[i+1]-H1_p2[i])/2/(1-kn1)

            y31[f"{i}L"] = (H2_p3[i+1]-H2_p1[i])/2/(1+kn2)
            y13[f"{i}L"] = (H2_p1[i+1]-H2_p3[i])/2/(1-kn2)

        yslr[(1, 2)] = y12
        yslr[(2, 1)] = y21
        yslr[(2, 3)] = y23
        yslr[(3, 2)] = y32
        yslr[(3, 1)] = y31
        yslr[(1, 3)] = y13

        return yslr
