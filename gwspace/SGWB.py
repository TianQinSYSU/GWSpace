#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_SGWB.py
# Author: Zhiyuan Li, Han Wang
# Mail: wangh657@mail2.sysu.edu.cn
# ==================================
"""Generate the Gaussian-like SGWB signal and signal after being responsed in a space detector
 in frequency domain. Support isotropy SGWB and SGWB from a specific orientation. """

import numpy as np
import healpy as hp
from healpy import Alm
from sympy.physics.wigner import clebsch_gordan

from gwspace.response import trans_XYZ_fd
from gwspace.Orbit import detectors
from gwspace.wrap import frequency_noise_from_psd
from gwspace.Waveform import p0_plus_cross
from gwspace.constants import H0_SI, PI, PI_2
from scipy.special import sph_harm


class SGWB(object):
    """ Generate the SGWB signal in frequency domain.

    :param nside: `nside` for `healpy`, the pixel number will be then determined by (12 * nside**2)
    :param omega0: Intensity of the signal at the reference frequency
    :param alpha: Power-law index
    :param T_obs: Observation time (s)
    :param theta: Coordinate for anisotropy SGWB source. [0, pi]. If BOTH None, give an isotropy SGWB.
    :param phi: Coordinate for anisotropy SGWB source. [0, 2pi]. If BOTH None, give an isotropy SGWB.
    """

    def __init__(self, nside, omega0, alpha, T_obs, theta=None, phi=None):
        self.omega0 = omega0
        self.alpha = alpha
        self.T_obs = T_obs
        self.npix = hp.nside2npix(nside)
        pix_idx = np.arange(self.npix)
        self.thetas, self.phis = hp.pix2ang(nside, pix_idx)

        if theta is None and phi is None:
            self.skymap_inj = np.ones(self.npix)/self.npix
        elif 0 <= theta <= PI and 0 <= phi <= 2*PI:
            self.blmax = 2
            self.blm_size = Alm.getsize(self.blmax)
            self.almax = 2*self.blmax
            self.alm_size = (self.almax+1)**2
            self.bl_bm_idx = [self.idx_2_alm(self.blmax, ii) for ii in range(2*self.blm_size-self.blmax-1)]

            l_m_val = [self.idx_2_alm(self.blmax, ii) for ii in range(self.blm_size)]
            self.blms = np.array([sph_harm(m, l, phi, theta) for (l, m) in l_m_val], dtype=np.complex128)
            beta_vals = self.calc_beta()
            blm_full = self.calc_blm_full()
            alms_inj = np.dot(np.dot(beta_vals, blm_full), blm_full)
            alms_inj2 = alms_inj/(alms_inj[0]*np.sqrt(4*PI))
            alms_non_neg = alms_inj2[0:hp.Alm.getsize(self.almax)]
            self.skymap_inj = hp.alm2map(alms_non_neg, nside)
        else:
            raise ValueError(f"Invalid theta({theta}) and/or phi({phi})")

    def get_ori_signal(self, frange, fref=0.01, seed=123):
        """ Generate a SGWB signal follows the Gaussian noise.

        :param frange: Frequency range
        :param fref: (Hz) Reference frequency
        :param seed: seed for np.random.seed
        :return: ndarray: shape(self.npix, frange.size)
        """
        Omegaf = self.omega0*(frange/fref)**self.alpha
        Sgw = Omegaf*(3/(4*frange**3))*(H0_SI/PI)**2
        Sgw = frequency_noise_from_psd(Sgw, 1/self.T_obs, seed=seed)
        return np.outer(self.skymap_inj, (2/self.T_obs) * np.real(Sgw*Sgw.conj()))

    def get_response_signal(self, f_min, f_max, fn, t_segm, det='TQ'):
        """ Generate a responsed (XYZ) signal for a given GW detector.

        :param f_min: (Hz) Minimum frequency
        :param f_max: (Hz) Maximum frequency
        :param fn: The number of frequency segments
        :param t_segm: Length of time segments (in seconds), usually choose ~ 3600s for TQ,
         the ORF error can be less than 3%
        :param det: str, for the detector type
        :return: (res_signal, frange): (ndarray: shape(fn, tf.size, 3, 3), ndarray: shape(fn, ))
        """
        frange = np.linspace(f_min, f_max, fn)
        gaussian_signal = self.get_ori_signal(frange)

        tf = np.arange(0, self.T_obs, t_segm)
        vec_k = self.vec_k
        e_plus_cross = [p0_plus_cross(p, PI_2-t) for p, t in np.column_stack((self.phis, self.thetas))]
        det = detectors[det](tf)

        res_signal = np.zeros((fn, tf.size, 3, 3), dtype=np.complex128)
        for i in range(self.npix):
            v_k, e_p_c = vec_k[i], e_plus_cross[i]
            for j in range(fn):
                res_p, res_c = trans_XYZ_fd(v_k, e_p_c, det, frange[j])  # both with shape(3, tf.size)
                det_ORF_temp = (np.einsum("ml,nl->lmn", res_p.conj(), res_p)
                                + np.einsum("ml,nl->lmn", res_c.conj(), res_c))
                # res_signal = det_ORF * gaussian_signal
                res_signal[j] += gaussian_signal[i, j] * det_ORF_temp / (8*PI) / (2*PI*frange[j]*det.L_T)**2

        res_signal *= (4*PI)/self.npix
        return res_signal, frange

    def calc_beta(self):
        ii, jj, kk = np.meshgrid(range(self.alm_size),
                                 range(2*self.blm_size-self.blmax-1),
                                 range(2*self.blm_size-self.blmax-1), indexing='ij')

        v_idx_2_alm = np.vectorize(lambda lmax, i: self.idx_2_alm(lmax, i))
        l1, m1 = v_idx_2_alm(self.blmax, jj)
        l2, m2 = v_idx_2_alm(self.blmax, kk)
        L, M = v_idx_2_alm(self.almax, ii)

        cg = np.vectorize(lambda l_1, m_1, l_2, m_2, L_, M_: float(clebsch_gordan(l_1, l_2, L_, m_1, m_2, M_)))
        beta_vals = (np.sqrt((2*l1+1)*(2*l2+1)/(4*PI*(2*L+1))) *
                     cg(l1, 0, l2, 0, L, 0)*cg(l1, m1, l2, m2, L, M))
        return beta_vals

    def calc_blm_full(self):
        """ Convert blm array into a full blm array with -m values too """
        # Array of blm values for both +ve and -ve indices
        blms_full = np.zeros(2*self.blm_size-self.blmax-1, dtype=np.complex128)

        for jj in range(blms_full.size):
            lval, mval = self.bl_bm_idx[jj]

            if mval >= 0:
                blms_full[jj] = self.blms[Alm.getidx(self.blmax, lval, mval)]
            else:
                mval = -mval
                blms_full[jj] = (-1)**mval * np.conj(self.blms[Alm.getidx(self.blmax, lval, mval)])

        return blms_full

    @staticmethod
    def idx_2_alm(lmax, ii):
        """ Index --> (l, m) function which works for negative indices too """
        alm_size = Alm.getsize(lmax)
        if ii >= (2*alm_size-lmax-1):
            raise ValueError('Index larger than acceptable')
        elif ii < alm_size:
            l, m = Alm.getlm(lmax, ii)
        else:
            l, m = Alm.getlm(lmax, ii-alm_size+lmax+1)

            if m == 0:
                raise ValueError('Something wrong with ind -> (l, m) conversion')
            else:
                m = -m
        return l, m

    @property
    def vec_k(self):
        return np.array([-np.sin(self.thetas)*np.cos(self.phis),
                         -np.sin(self.thetas)*np.sin(self.phis),
                         -np.cos(self.thetas)]).T  # Vector of sources
