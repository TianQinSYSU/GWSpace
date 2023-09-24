#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: response.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
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


def trans_y_slr_fd(vec_k, p, det, f):
    """ See Marsat et al. (Eq. 21, 28) https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011

    :param vec_k: the prop direction of GWs (x,y,z), which is determined by (lambda, beta)
    :param p: p is a tuple, i.e. a series of P_lm (or P_x, P_+, i.e. e^+, e^x in $h = h_+ e^+ + h_x e^x$)
    :param det:
    :param f:
    :return:
    """

    u12 = det.uni_vec_ij(1, 2)
    u23 = det.uni_vec_ij(2, 3)
    u13 = det.uni_vec_ij(1, 3)
    ls = det.L_T
    p_1, p_2, p_3 = det.orbits

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


def get_y_slr_td(wf, tf, det='TQ', TDIgen=1):
    """TODO: here we calculate orbits of the detectors only once at tf, but considering TDI_delay here,
         their positions need to be **recalculated**, e.g. at tf-L, tf-2*L, ..."""
    if TDIgen == 1:
        TDI_delay = 4
    else:
        raise NotImplementedError

    det = detectors[det](tf)
    p1, p2, p3 = det.orbits
    L = det.L_T
    n1 = det.uni_vec_ij(3, 2)
    n2 = det.uni_vec_ij(1, 3)
    n3 = det.uni_vec_ij(2, 1)

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


def tdi_XYZ2AET(X, Y, Z):
    """
    Calculate AET channel from XYZ
    """
    A = 1/np.sqrt(2)*(Z-X)
    E = 1/np.sqrt(6)*(X-2*Y+Z)
    T = 1/np.sqrt(3)*(X+Y+Z)
    return A, E, T


def get_XYZ_td(y_slr, TDIgen=1):
    """
    Generate TDI XYZ in the TDIgen generation
    -----------------------------------------
    Parameters
    ----------
    """
    y31 = y_slr[(3, 1)]
    y13 = y_slr[(1, 3)]
    y12 = y_slr[(1, 2)]
    y21 = y_slr[(2, 1)]
    y23 = y_slr[(2, 3)]
    y32 = y_slr[(3, 2)]

    if TDIgen == 1:
        X = (y31[0]+y13[1]+y21[2]+y12[3]
             - y21[0]-y12[1]-y31[2]-y13[3])
        Y = (y12[0]+y21[1]+y32[2]+y23[3]
             - y32[0]-y23[1]-y12[2]-y21[3])
        Z = (y23[0]+y32[1]+y13[2]+y31[3]
             - y13[0]-y31[1]-y23[2]-y32[3])
    else:
        raise NotImplementedError

    return X, Y, Z


def get_AET_td(y_slr, TDIgen=1):
    X, Y, Z = get_XYZ_td(y_slr, TDIgen)
    A, E, T = tdi_XYZ2AET(X, Y, Z)
    return A, E, T


def get_XYZ_fd(y_slr, freq, L_T):
    """
    Calculate XYZ from y_slr in frequency domain
    -------------------------------------------
    Parameters:
    - y_slr: single link response of GW
    - freq: frequency
    - L_T: arm length
    """

    Dt = np.exp(2j*np.pi*freq*L_T)
    Dt2 = Dt*Dt

    X = y_slr[(3, 1)]+Dt*y_slr[(1, 3)]-y_slr[(2, 1)]-Dt*y_slr[(1, 2)]
    Y = y_slr[(1, 2)]+Dt*y_slr[(2, 1)]-y_slr[(3, 2)]-Dt*y_slr[(2, 3)]
    Z = y_slr[(2, 3)]+Dt*y_slr[(3, 2)]-y_slr[(1, 3)]-Dt*y_slr[(3, 1)]

    return np.array([X, Y, Z])*(1.-Dt2)


def get_AET_fd(y_slr, freq, L_T):
    """
    Calculate AET from y_slr in frequency domain
    -------------------------------------------
    Parameters:
    - y_slr: single link response of GW
    - freq: frequency
    - L_T: arm length
    """
    Dt = np.exp(2j*np.pi*freq*L_T)  # Time delay factor
    Dt2 = Dt*Dt

    A = ((1+Dt)*(y_slr[(3, 1)]+y_slr[(1, 3)])
         - y_slr[(2, 3)]-Dt*y_slr[(3, 2)]
         - y_slr[(2, 1)]-Dt*y_slr[(1, 2)])
    E = ((1-Dt)*(y_slr[(1, 3)]-y_slr[(3, 1)])
         + (1+2*Dt)*(y_slr[(2, 1)]-y_slr[(2, 3)])
         + (2+Dt)*(y_slr[(1, 2)]-y_slr[(3, 2)]))
    T = (1-Dt)*(y_slr[(1, 3)]-y_slr[(3, 1)]
                + y_slr[(2, 1)]-y_slr[(1, 2)]
                + y_slr[(3, 2)]-y_slr[(2, 3)])

    A = 1/np.sqrt(2)*(Dt2-1)*A
    E = 1/np.sqrt(6)*(Dt2-1)*E
    T = 1/np.sqrt(3)*(Dt2-1)*T

    return np.array([A, E, T])
