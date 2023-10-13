import numpy as np
import healpy as hp
from healpy import Alm
from sympy.physics.wigner import clebsch_gordan

from gwspace.response import trans_XYZ_fd
from gwspace.Orbit import detectors
from gwspace.wrap import frequency_noise_from_psd
from gwspace.Waveform import _p0  # FIXME
from gwspace.constants import H0_SI
from scipy.special import sph_harm


class SGWB(object):
    """ Generate the anisotropy SGWB signal in frequency domain

    :param nside: we usually use 8,12,16... The pixel number will be determined by (12 * nside**2)
    :param omega0: the intensity of the signal at reference frequency
    :param alpha: the power-law index
    :param T_obs: Observation time (s)
    :param theta:
    :param phi:
    """

    def __init__(self, nside, omega0, alpha, T_obs, theta=None, phi=None):
        self.omega0 = omega0
        self.alpha = alpha
        self.T_obs = T_obs
        self.npix = hp.nside2npix(nside)
        pix_idx = np.arange(self.npix)
        self.thetas, self.phis = hp.pix2ang(nside, pix_idx)

        if theta is None and phi is None:
            self.skymap_inj = 1/self.npix*np.ones(self.npix)
        elif 0 <= theta <= np.pi and 0 <= phi <= 2*np.pi:
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
            alms_inj2 = alms_inj/(alms_inj[0]*np.sqrt(4*np.pi))
            alms_non_neg = alms_inj2[0:hp.Alm.getsize(self.almax)]
            self.skymap_inj = hp.alm2map(alms_non_neg, nside)
        else:
            raise ValueError("the range of theta or phi is wrong")

    def get_ori_signal(self, frange, fref=0.01, seed=123):
        """ Generate a SGWB signal follows the Gaussian noise.

        :param frange: the frequency range
        :param fref: (Hz) the reference frequency
        :param seed: seed for np.random.seed
        :return: ndarray: shape(self.npix, frange.size)
        """
        Omegaf = self.omega0*(frange/fref)**self.alpha
        Sgw = Omegaf*(3/(4*frange**3))*(H0_SI/np.pi)**2
        Sgw = frequency_noise_from_psd(Sgw, 1/self.T_obs, seed=seed)
        return np.outer(self.skymap_inj, (2/self.T_obs) * np.real(Sgw*Sgw.conj()))

    def get_response_signal(self, f_min, f_max, fn, t_segm, det='TQ'):
        """ Generate a responsed (XYZ) signal for a given GW detector.

        :param f_min: (Hz) the minimum frequency we generate
        :param f_max: (Hz) the maximum frequency we generate
        :param fn: the number of frequency segments
        :param t_segm: the length of time segments (in seconds), usually choose ~ 3600s for TQ,
         the ORF error can be less than 3%
        :param det: str, the detector type
        :return: (res_signal, frange): (ndarray: shape(fn, tf.size, 3, 3), ndarray: shape(fn, ))
        """
        frange = np.linspace(f_min, f_max, fn)
        gaussian_signal = self.get_ori_signal(frange)

        tf = np.arange(0, self.T_obs, t_segm)
        vec_k = self.vec_k
        e_plus_cross = [_p0(p, np.pi/2-t) for p, t in np.column_stack((self.phis, self.thetas))]
        det = detectors[det](tf)

        res_signal = np.zeros((fn, tf.size, 3, 3), dtype=np.complex128)
        for i in range(self.npix):
            v_k, e_p_c = vec_k[i], e_plus_cross[i]
            for j in range(fn):
                res_p, res_c = trans_XYZ_fd(v_k, e_p_c, det, frange[j])  # both with shape(3, tf.size)
                det_ORF_temp = (np.einsum("ml,nl->lmn", res_p.conj(), res_p)
                                + np.einsum("ml,nl->lmn", res_c.conj(), res_c))
                # res_signal = det_ORF * gaussian_signal
                res_signal[j] += gaussian_signal[i, j] * det_ORF_temp / (8*np.pi) / (2*np.pi*frange[j]*det.L_T)**2

        res_signal *= (4*np.pi)/self.npix
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
        beta_vals = (np.sqrt((2*l1+1)*(2*l2+1)/(4*np.pi*(2*L+1))) *
                     cg(l1, 0, l2, 0, L, 0)*cg(l1, m1, l2, m2, L, M))
        return beta_vals

    def calc_blm_full(self):
        """convert blm array into a full blm array with -m values too"""
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
        """ index --> (l, m) function which works for negative indices too """
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
