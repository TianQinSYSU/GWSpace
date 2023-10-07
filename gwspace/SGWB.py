import numpy as np
import healpy as hp
from healpy import Alm
from sympy.physics.quantum.cg import CG

from gwspace.response import trans_y_slr_fd, get_XYZ_fd
from gwspace.Orbit import detectors
from gwspace.wrap import frequency_noise_from_psd
from gwspace.constants import H0_SI
from scipy.special import sph_harm


class SGWB(object):
    """ Generate the anisotropy SGWB signal in frequency domain

    :param nside: we usually use 8,12,16... The pixel number will be determined by (12 * nside**2)
    :param omega0: the intensity of the signal at reference frequency
    :param alpha: the power-law index
    :param blm_vals: A series of spherical coefficients
    :param blmax: the order of blm_vals
    :param H0: Hubble constant
    """

    def __init__(self, nside, omega0, alpha, theta, phi, H0=H0_SI):
        #if len(blm_vals) != self.blm_size:
        #    raise ValueError('The size of the input blm array does not match the size defined by lmax')

        self.omega0 = omega0
        self.alpha = alpha
        self.H0 = H0
        self.npix = hp.nside2npix(nside)
        pix_idx = np.arange(self.npix)
        self.thetas, self.phis = hp.pix2ang(nside, pix_idx)
        if theta == None and phi==None:
            self.skymap_inj = 1/self.npix*np.ones(self.npix)
        elif theta>=0 and theta<= np.pi and phi>=0 and phi<=2*np.pi:        
            self.blmax = 2
            self.blm_size = Alm.getsize(self.blmax)
            self.almax = 2*self.blmax
            self.alm_size = (self.almax+1)**2
            self.bl_bm_idx = [self.idx_2_alm(self.blmax, ii) for ii in range(2*self.blm_size-self.blmax-1)]
            Ylms = np.zeros(self.blm_size, dtype='complex')
            for ii in range(self.blm_size):
                lval, mval = self.idx_2_alm(self.blmax, ii)
                Ylms[ii] = sph_harm(mval, lval, phi, theta)
            self.blms = np.array(Ylms, dtype=np.complex128)
            beta_vals = self.calc_beta()
            blm_full = self.calc_blm_full()
            alms_inj = np.einsum('ijk,j,k', beta_vals, blm_full, blm_full)
            alms_inj2 = alms_inj/(alms_inj[0]*np.sqrt(4*np.pi))
            alms_non_neg = alms_inj2[0:hp.Alm.getsize(self.almax)]
            self.skymap_inj = hp.alm2map(alms_non_neg, nside)
        else:
            print("the range of theta or phi is wrong")
    
    def get_ori_signal(self, frange, Ttot, fref=0.01):
        """

        :param frange: the frequency range
        :param Ttot: the total observation time (in seconds)
        :param fref: (Hz) the reference frequency
        :return:
        """
        Omegaf = self.omega0*(frange/fref)**self.alpha
        Sgw = Omegaf*(3/(4*frange**3))*(self.H0/np.pi)**2
        Sgw_Gaussian = frequency_noise_from_psd(Sgw, 1/Ttot, seed=123)
        return np.einsum('i,j->ij', (2/Ttot)*Sgw_Gaussian*Sgw_Gaussian.conj(), self.skymap_inj)  # Gaussian signal

    def get_response_signal(self, fmin, fmax, fn, tsegmid, Ttot, det='TQ'):
        """

        :param fmin: (Hz) the minimum frequency we generate
        :param fmax: (Hz) the maximum frequency we generate
        :param fn: the frequency segment number
        :param tsegmid: the time segment length (in seconds), for TQ we usually use 5000s or 3600s,
         The ORF error is less than 3%
        :param Ttot: the total observation time (in seconds)
        :param det: the detector type
        :return:
        """
        frange = np.linspace(fmin, fmax, fn)
        tf = np.arange(0, Ttot, tsegmid)
        e_plus = np.einsum("ik,jk->ijk", self.vec_v, self.vec_v)-np.einsum("ik,jk->ijk", self.vec_u, self.vec_u)
        e_cross = np.einsum("ik,jk->ijk", self.vec_u, self.vec_v)+np.einsum("ik,jk->ijk", self.vec_v, self.vec_u)
        det = detectors[det](tf)
        det_res_plus = np.zeros((3, fn, tf.size, self.npix), dtype="complex")
        det_res_cross = np.zeros((3, fn, tf.size, self.npix), dtype="complex")
        for i in range(self.npix):
            for j in range(fn):  # TODO: polish this
                y_plus, y_cross = trans_y_slr_fd(self.vec_k[:, i], (e_plus[:, :, i], e_cross[:, :, i]), det, frange[j])
                det_res_plus[:, j, :, i] = get_XYZ_fd(y_plus, frange[j], det.L_T)
                det_res_cross[:, j, :, i] = get_XYZ_fd(y_cross, frange[j], det.L_T)
        # det_ORF is 3*3*frequency*t*pixel
        det_ORF = (1/(8*np.pi))*(np.einsum("mjkl,njkl->mnjkl", np.conj(det_res_plus), det_res_plus)
                                 + np.einsum("mjkl,njkl->mnjkl", np.conj(det_res_cross), det_res_cross))/(
                                  2*np.pi*frange[None, None, :, None, None]*det.L_T)**2
        signal_in_gaussian = self.get_ori_signal(frange, Ttot)
        res_signal = signal_in_gaussian[None, None, :, None, :]*det_ORF
        res_signal = (4*np.pi)*np.sum(res_signal, axis=4)/self.npix
        #return res_signal, frange,det_ORF
        return res_signal,frange

    def calc_beta(self):
        beta_vals = np.zeros((self.alm_size, 2*self.blm_size-self.blmax-1, 2*self.blm_size-self.blmax-1))

        for ii in range(beta_vals.shape[0]):
            for jj in range(beta_vals.shape[1]):
                for kk in range(beta_vals.shape[2]):
                    l1, m1 = self.idx_2_alm(self.blmax, jj)
                    l2, m2 = self.idx_2_alm(self.blmax, kk)
                    L, M = self.idx_2_alm(self.almax, ii)

                    # Clebsch-Gordan coefficient
                    cg0 = (CG(l1, 0, l2, 0, L, 0).doit()).evalf()
                    cg1 = (CG(l1, m1, l2, m2, L, M).doit()).evalf()

                    beta_vals[ii, jj, kk] = np.sqrt((2*l1+1)*(2*l2+1)/((4*np.pi)*(2*L+1))) * cg0 * cg1

        return beta_vals

    def calc_blm_full(self):
        """convert blm array into a full blm array with -m values too"""
        # Array of blm values for both +ve and -ve indices
        blms_full = np.zeros(2*self.blm_size-self.blmax-1, dtype='complex')

        for jj in range(blms_full.size):
            lval, mval = self.bl_bm_idx[jj]

            if mval >= 0:
                blms_full[jj] = self.blms[Alm.getidx(self.blmax, lval, mval)]

            elif mval < 0:
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

    #@property
    #def vec_u(self):
    #    return np.array([-np.sin(self.phis), np.cos(self.phis), np.zeros(len(self.phis))])

    #@property
    #def vec_v(self):
    #    return np.array([np.cos(self.thetas)*np.cos(self.phis),
    #                     np.cos(self.thetas)*np.sin(self.phis),
    #                     -np.sin(self.thetas)])

    #@property
    #def vec_k(self):
    #    return np.array([-np.sin(self.thetas)*np.cos(self.phis),
    #                     -np.sin(self.thetas)*np.sin(self.phis),
    #                     -np.cos(self.thetas)])  # Vector of sources
    @property
    def vec_u(self):
        return np.array([np.sin(self.phis), -np.cos(self.phis), np.zeros(len(self.phis))])

    @property
    def vec_v(self):
        return np.array([-np.sin(np.pi/2-self.thetas)*np.cos(self.phis),
                         -np.sin(np.pi/2-self.thetas)*np.sin(self.phis),
                         np.cos(np.pi/2-self.thetas)])

    @property
    def vec_k(self):
        return np.array([-np.cos(np.pi/2-self.thetas)*np.cos(self.phis),
                         -np.cos(np.pi/2-self.thetas)*np.sin(self.phis),
                         -np.sin(np.pi/2-self.thetas)])  # Vector of sources