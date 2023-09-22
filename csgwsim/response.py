#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: response.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 14:55:20
# ==================================

import numpy as np
from numba import jit
from .Orbit import detectors


@jit(nopython=True)
def _matrix_res_pro(n, p):
    """TensorProduct(n, n) : P,  where A:B = A_ij B_ij"""
    return (n[0] * p[0, 0] * n[0] + n[0] * p[0, 1] * n[1] + n[0] * p[0, 2] * n[2]
            + n[1] * p[1, 0] * n[0] + n[1] * p[1, 1] * n[1] + n[1] * p[1, 2] * n[2]
            + n[2] * p[2, 0] * n[0] + n[2] * p[2, 1] * n[1] + n[2] * p[2, 2] * n[2])


def trans_fd_response(vec_k, p, det, f):
    """
    See Marsat et al. (Eq. 21, 28) https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011
    Parameters:
    - vec_k: the prop direction of GWs (x,y,z), which is determined by (lambd, beta)
    - p: tensor of polar for GW, h = h_+ e^+ + h_x e^x in time domain, but h_ij (f) = p_ij h(f)
    """

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
        n12pn12 = _matrix_res_pro(u12, p_)
        n23pn23 = _matrix_res_pro(u23, p_)
        n31pn31 = _matrix_res_pro(u13, p_)

        y_slr = {(1, 2): y12_pre * n12pn12,
                 (2, 1): y21_pre * n12pn12,
                 (1, 3): y13_pre * n31pn31,
                 (3, 1): y31_pre * n31pn31,
                 (2, 3): y23_pre * n23pn23,
                 (3, 2): y32_pre * n23pn23}
        return y_slr

    if type(p) != tuple:
        p = (p, )
    return tuple(trans_response(p_0) for p_0 in p)


def get_fd_response(vec_k, p, det, f, channel='A'):
    """TODO: deprecated

    :param vec_k:
    :param p: p is a tuple, i.e. a series of P_lm (or P_x, P_+) in it
    :param det:
    :param f:
    :param channel:
    :return:
    """
    # if channel not in 'XYZAET':  # <if not all([c in 'XYZAET' for c in channel])> for multichannel mode
    #     raise ValueError(f"[SpaceResponse] Unknown channel {channel}. "
    #                      f"Supported channels: {'|'.join(['X', 'Y', 'Z', 'A', 'E', 'T'])}")
    y_slr = trans_fd_response(vec_k, p, det, f)
    x = np.pi * f * det.L_T
    z = np.exp(2j * x)  # Time delay factor
    res_list = []

    for y_p in y_slr:
        y12, y21, y13, y31, y23, y32 = list(y_p.values())
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


def get_td_response(wf, tf, det='TQ', TDIgen=1):
    """TODO: here we calculate orbits of the detectors only once at tf, but considering TDI_delay here,
         their positions need to be **recalculated**, e.g. at tf-L, tf-2*L, ..."""
    if TDIgen == 1:
        TDI_delay = 4
    else:
        raise NotImplementedError

    det = detectors[det](tf)
    p1, p2, p3 = det.orbit_1, det.orbit_2, det.orbit_3
    L = det.L_T
    n1 = -det.Uni_vec_23
    n2 = det.Uni_vec_13
    n3 = -det.Uni_vec_12

    k = wf.vec_k
    p_p, p_c = wf.polarization()
    xi1 = (_matrix_res_pro(n1, p_p), _matrix_res_pro(n1, p_c))
    xi2 = (_matrix_res_pro(n2, p_p), _matrix_res_pro(n2, p_c))
    xi3 = (_matrix_res_pro(n3, p_p), _matrix_res_pro(n3, p_c))

    kp1 = np.dot(k, p1)
    kp2 = np.dot(k, p2)
    kp3 = np.dot(k, p3)
    kn1 = np.dot(k, n1)
    kn2 = np.dot(k, n2)
    kn3 = np.dot(k, n3)

    tf_kp1, tf_kp2, tf_kp3 = tf-kp1, tf-kp2, tf-kp3

    # Here `i` is for `i*L` TDI_delay, in 1st generation TDI we consider delay up to 4,
    # i.e. t-0L, t-1L, t-2L, t-3L, t-4L. And then we calculate difference between each L delay
    def h_tdi_delay(tf_s, xi_p, xi_c):
        h_list = [wf.get_hphc(tf_s - i_*L) for i_ in range(TDI_delay+1)]
        return [hp*xi_p+hc*xi_c for (hp, hc) in h_list]

    h3_p2 = h_tdi_delay(tf_kp2, *xi3)
    h3_p1 = h_tdi_delay(tf_kp1, *xi3)
    h2_p3 = h_tdi_delay(tf_kp3, *xi2)
    h2_p1 = h_tdi_delay(tf_kp1, *xi2)
    h1_p3 = h_tdi_delay(tf_kp3, *xi1)
    h1_p2 = h_tdi_delay(tf_kp2, *xi1)

    def get_y(hi_pj, hi_pk, denominator):
        return [(hi_pj[i+1]-hi_pk[i])/denominator for i in range(TDI_delay)]

    y_slr = {(1, 2): get_y(h3_p1, h3_p2, 2*(1+kn3)),
             (2, 1): get_y(h3_p2, h3_p1, 2*(1-kn3)),
             (1, 3): get_y(h2_p1, h2_p3, 2*(1-kn2)),
             (3, 1): get_y(h2_p3, h2_p1, 2*(1+kn2)),
             (2, 3): get_y(h1_p2, h1_p3, 2*(1+kn1)),
             (3, 2): get_y(h1_p3, h1_p2, 2*(1-kn1))}
    return y_slr
