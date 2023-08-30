#
# Copyright (C) 2020 Sylvain Marsat.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with with program; see the file COPYING. If not, write to the
#  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA  02111-1307  USA
#

"""
    Standalone python implementation of TaylorF2 phasing.
"""

import numpy as np
import pytools as pytools
import pyconstants as pyconstants

PN_order_max = 7

# c.f. notebook TF2.nb, end of section "Checks for SEOBNRv4T"
# NOTE: constants used (pi, gamma) are numpy constants here
def tf2_phasing_coeffs(q, chi1, chi2, kappa1=1., kappa2=2., PNorderNS=7, PNorderSO=7, PNorderSS=6):
    coeffs = {}

    eta = pytools.etaofq(q)
    delta = pytools.deltaofq(q)
    chi_s = 1./2 * (chi1 + chi2)
    chi_a = 1./2 * (chi1 - chi2)
    kappa_s = 1./2 * (kappa1 + kappa2)
    kappa_a = 1./2 * (kappa1 - kappa2)

    # Maximal orders
    coeffs['PNorderNS'] = PNorderNS
    coeffs['PNorderSO'] = PNorderSO
    coeffs['PNorderSS'] = PNorderSS

    # Newtonian coefficient, prefactor
    coeffs['prefactor'] = 3./128/eta

    # Non-spinning PN coefficients
    coeffs['NS'] = {}
    coeffs['NS'][0] = 1.
    coeffs['NS'][1] = 0.
    coeffs['NS'][2] = eta*(55/9.) + (3715/756.)
    coeffs['NS'][3] = -16*np.pi
    coeffs['NS'][4] = eta*eta*(3085/72.) + eta*(27145/504.) + (15293365/508032.)
    coeffs['NS'][5] = eta*np.pi*(-65/9.) + np.pi*(38645/756.)
    coeffs['NS'][6] = eta*(-15737765635/3048192.) + eta*eta*eta*(-127825/1296.) + np.log(2)*(-13696/21.) + np.euler_gamma*(-6848/21.) + np.pi*np.pi*(-640/3.) + eta*np.pi*np.pi*(2255/12.) + eta*eta*(76055/1728.) + (11583231236531/4694215680.)
    coeffs['NS'][7] = eta*eta*np.pi*(-74045/756.) + eta*np.pi*(378515/1512.) + np.pi*(77096675/254016.)
    # Non-spinning PN coefficients with factor log(v)
    coeffs['NSlogv'] = {}
    coeffs['NSlogv'][0] = 0.
    coeffs['NSlogv'][1] = 0.
    coeffs['NSlogv'][2] = 0.
    coeffs['NSlogv'][3] = 0.
    coeffs['NSlogv'][4] = 0.
    coeffs['NSlogv'][5] = eta*np.pi*(-65/3.) + np.pi*(38645/252.)
    coeffs['NSlogv'][6] = (-6848/21.)
    coeffs['NSlogv'][7] = 0.

    # Spin-orbit PN coefficients
    coeffs['SO'] = {}
    coeffs['SO'][0] = 0.
    coeffs['SO'][1] = 0.
    coeffs['SO'][2] = 0.
    coeffs['SO'][3] = chi_a*delta*(113/3.) + chi_s*(eta*(-76/3.) + (113/3.))
    coeffs['SO'][4] = 0.
    coeffs['SO'][5] = chi_a*(delta*(-732985/2268.) + delta*eta*(-140/9.)) + chi_s*((-732985/2268.) + eta*eta*(340/9.) + eta*(24260/81.))
    coeffs['SO'][6] = chi_a*delta*np.pi*(2270/3.) + chi_s*(-520*eta*np.pi + np.pi*(2270/3.))
    coeffs['SO'][7] = chi_a*(delta*(-25150083775/3048192.) + delta*eta*eta*(-1985/48.) + delta*eta*(26804935/6048.)) + chi_s*((-25150083775/3048192.) + eta*eta*(-1042165/3024.) + eta*eta*eta*(5345/36.) + eta*(10566655595/762048.))
    # Spin-orbit PN coefficients with factor log(v)
    coeffs['SOlogv'] = {}
    coeffs['SOlogv'][0] = 0.
    coeffs['SOlogv'][1] = 0.
    coeffs['SOlogv'][2] = 0.
    coeffs['SOlogv'][3] = 0.
    coeffs['SOlogv'][4] = 0.
    coeffs['SOlogv'][5] = chi_a*(delta*(-732985/756.) + delta*eta*(-140/3.)) + chi_s*((-732985/756.) + eta*eta*(340/3.) + eta*(24260/27.))
    coeffs['SOlogv'][6] = 0.
    coeffs['SOlogv'][7] = 0.

    # Spin-square PN coefficients
    coeffs['SS'] = {}
    coeffs['SS'][0] = 0.
    coeffs['SS'][1] = 0.
    coeffs['SS'][2] = 0.
    coeffs['SS'][3] = 0.
    coeffs['SS'][4] = chi_a*chi_s*(-100*kappa_a + 200*eta*kappa_a - 100*delta*kappa_s + delta*(-5/4.)) + chi_a*chi_a*(100*eta - 50*delta*kappa_a - 50*kappa_s + 100*eta*kappa_s + (-5/8.)) + chi_s*chi_s*(-50*delta*kappa_a - 50*kappa_s + 100*eta*kappa_s + eta*(-195/2.) + (-5/8.))
    coeffs['SS'][5] = 0.
    coeffs['SS'][6] = chi_a*chi_s*(-480*eta*eta*kappa_a + delta*(-1344475/1008.) + eta*kappa_a*(-88510/21.) + delta*eta*kappa_s*(-1495/3.) + delta*eta*(745/18.) + kappa_a*(26015/14.) + delta*kappa_s*(26015/14.)) + chi_a*chi_a*(-240*eta*eta - 240*eta*eta*kappa_s + (-1344475/2016.) + eta*kappa_s*(-44255/21.) + delta*eta*kappa_a*(-1495/6.) + delta*kappa_a*(26015/28.) + kappa_s*(26015/28.) + eta*(267815/252.)) + chi_s*chi_s*(-240*eta*eta*kappa_s + (-1344475/2016.) + eta*kappa_s*(-44255/21.) + delta*eta*kappa_a*(-1495/6.) + eta*eta*(3415/9.) + delta*kappa_a*(26015/28.) + kappa_s*(26015/28.) + eta*(829705/504.))
    # Spin-square PN coefficients with factor log(v)
    coeffs['SSlogv'] = {}
    coeffs['SSlogv'][0] = 0.
    coeffs['SSlogv'][1] = 0.
    coeffs['SSlogv'][2] = 0.
    coeffs['SSlogv'][3] = 0.
    coeffs['SSlogv'][4] = 0.
    coeffs['SSlogv'][5] = 0.
    coeffs['SSlogv'][6] = 0.

    # Precompute sum of NS, SO, SS coeffs
    coeffs['v'] = np.zeros(PN_order_max+1, dtype=float)
    coeffs['vlogv'] = np.zeros(PN_order_max+1, dtype=float)
    for i in range(PNorderNS+1):
        coeffs['v'][i] = coeffs['NS'][i]
        coeffs['vlogv'][i] = coeffs['NSlogv'][i]
    for i in range(PNorderSO+1):
        coeffs['v'][i] += coeffs['SO'][i]
        coeffs['vlogv'][i] += coeffs['SOlogv'][i]
    for i in range(PNorderSS+1):
        coeffs['v'][i] += coeffs['SS'][i]
        coeffs['vlogv'][i] += coeffs['SSlogv'][i]

    return coeffs

# The solar mass is from pyconstants
def tf2_phasing_eval(coeffs, M, f):

    M_s = M * pyconstants.MTSUN_SI

    v = np.power(np.pi * M_s * f, 1./3)
    logv = np.log(v)

    v_minus5 = np.power(v, -5.)
    v_n = {}
    v_n[1] = v
    for i in range(2, PN_order_max+1):
        v_n[i] = v * v_n[i-1]

    # Leading term -- for GR this is 1., coeffs['v'][0]=1. and coeffs['vlogv'][0]=0.
    phase_scaled = (coeffs['v'][0] + coeffs['vlogv'][0]*logv)
    for i in range(1, PN_order_max+1):
        phase_scaled += (coeffs['v'][i] + coeffs['vlogv'][i]*logv) * v_n[i]
    phase = coeffs['prefactor'] * v_minus5 * phase_scaled

    return phase

def tf2_phasing_eval_deriv(coeffs, M, f):

    M_s = M * pyconstants.MTSUN_SI

    v = np.power(np.pi * M_s * f, 1./3)
    logv = np.log(v)

    v_minus8 = np.power(v, -8.)
    v_n = {}
    v_n[0] = 1.
    for i in range(1, PN_order_max+1):
        v_n[i] = v * v_n[i-1]

    # Leading term -- for GR this is -5., coeffs['v'][0]=1. and coeffs['vlogv'][0]=0.
    dfphase_scaled = (((-5)*coeffs['v'][0] + coeffs['vlogv'][0]) + (-5)*coeffs['vlogv'][0]*logv)
    for i in range(1,PN_order_max+1):
        dfphase_scaled += (((i-5)*coeffs['v'][i] + coeffs['vlogv'][i]) + (i-5)*coeffs['vlogv'][i]*logv) * v_n[i]

    dfphase = np.pi*M_s/3. * coeffs['prefactor'] * v_minus8 * dfphase_scaled

    return dfphase

def tf2_phasing_eval_phase_and_deriv(coeffs, M, f):

    M_s = M * pyconstants.MTSUN_SI

    v = np.power(np.pi * M_s * f, 1./3)
    logv = np.log(v)

    v_minus8 = np.power(v, -8.)
    v_n = {}
    v_n[0] = 1.
    for i in range(1, PN_order_max+1):
        v_n[i] = v * v_n[i-1]

    # Leading term -- for GR this is 1., coeffs['v'][0]=1. and coeffs['vlogv'][0]=0.
    phase_scaled = (coeffs['v'][0] + coeffs['vlogv'][0]*logv)
    for i in range(1, PN_order_max+1):
        phase_scaled += (coeffs['v'][i] + coeffs['vlogv'][i]*logv) * v_n[i]
    phase = coeffs['prefactor'] * v_minus5 * phase_scaled

    # Leading term -- for GR this is -5., coeffs['v'][0]=1. and coeffs['vlogv'][0]=0.
    dfphase_scaled = (((-5)*coeffs['v'][0] + coeffs['vlogv'][0]) + (-5)*coeffs['vlogv'][0]*logv)
    for i in range(1,PN_order_max+1):
        dfphase_scaled += (((i-5)*coeffs['v'][i] + coeffs['vlogv'][i]) + (i-5)*coeffs['vlogv'][i]*logv) * v_n[i]
    dfphase = np.pi*M_s/3. * coeffs['prefactor'] * v_minus8 * dfphase_scaled

    return phase, dfphase

def tf2_phasing(freqs, M, q, chi1, chi2, kappa1=1., kappa2=2., f0=None, t0=0., phi0=0., PNorderNS=7, PNorderSO=7, PNorderSS=6):

    if f0 is None or f0<=0.:
        f0 = freqs[0]

    coeffs = tf2_phasing_coeffs(q, chi1, chi2, kappa1=kappa1, kappa2=kappa2, PNorderNS=PNorderNS, PNorderSO=PNorderSO, PNorderSS=PNorderSS)

    phi_f0 = tf2_phasing_eval(coeffs, M, f0)
    dfphi_f0 = tf2_phasing_eval_deriv(coeffs, M, f0)

    deltaphi_align = 2*np.pi*(t0 - dfphi_f0/(2*np.pi))*(freqs - f0) + phi0 - phi_f0

    phi = tf2_phasing_eval(coeffs, M, freqs) + deltaphi_align

    return phi

def tf2_phasing_tf(freqs, M, q, chi1, chi2, kappa1=1., kappa2=2., f0=None, t0=0., phi0=0., PNorderNS=7, PNorderSO=7, PNorderSS=6):

    if f0 is None or f0<=0.:
        f0 = freqs[0]

    coeffs = tf2_phasing_coeffs(q, chi1, chi2, kappa1=kappa1, kappa2=kappa2, PNorderNS=PNorderNS, PNorderSO=PNorderSO, PNorderSS=PNorderSS)

    phi_f0 = tf2_phasing_eval(coeffs, M, f0)
    dfphi_f0 = tf2_phasing_eval_deriv(coeffs, M, f0)

    deltaphi_align = 2*np.pi*(t0 - dfphi_f0/(2*np.pi))*(freqs - f0) + phi0 - phi_f0
    dfdeltaphi_align = 2*np.pi*(t0 - dfphi_f0/(2*np.pi))

    phi, dfphi = tf2_phasing_eval_and_deriv(coeffs, M, freqs)

    phi = phi + deltaphi_align
    tf = (dfphi + dfdeltaphi_align) / (2*np.pi)

    return phi, tf

# NOTE: for now, only Newtonian amplitude implemented
# DL in Mpc
def tf2_amp(freqs, M, q, chi1, chi2, DL, kappa1=1., kappa2=2., PNorderNS=7, PNorderSO=7, PNorderSS=6):

    if PNorderNS>0 or PNorderSO>0 or PNorderSS>0:
        raise ValueError("TF2 amplitude only implemented at Newtonian order for now.")

    eta = pytools.etaofq(q)
    M_s = M * pyconstants.MTSUN_SI
    M_m = M * pyconstants.MRSUN_SI

    v = np.power(np.pi * M_s * freqs, 1./3)

    aN = M_m / (DL * 1e6*pyconstants.PC_SI) * M_s * np.pi * np.sqrt(2*eta/3.) * np.power(v, -7./2)

    return aN
