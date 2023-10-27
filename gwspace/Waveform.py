#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Waveform.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-08-01 12:32:36
# ==================================
"""All available waveforms for different GW sources."""

import numpy as np
from numpy import sin, cos, sqrt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from gwspace.Orbit import detectors
from gwspace.utils import sYlm
from gwspace.constants import MSUN_SI, MTSUN_SI, MPC_SI, YRSID_SI, PI, PI_2, C_SI
from gwspace.response import trans_AET_fd

from gwspace.eccentric_fd import gen_ecc_fd_and_tf, gen_ecc_fd_waveform
try:
    from PyIMRPhenomD import IMRPhenomD as pyIMRD
    from PyIMRPhenomD import IMRPhenomD_const as pyIMRc
    use_py_phd = True
except ImportError:
    from gwspace.pyIMRPhenomD import IMRPhenomDh22AmpPhase as pyIMRD
    use_py_phd = False

if __package__ or "." in __name__:
    from gwspace import libFastGB
else:
    import libFastGB


def p0_plus_cross(Lambda, Beta):
    """See Marsat et al. (Eq. 14)"""
    sib, csb = sin(Beta), cos(Beta)
    sil, csl = sin(Lambda), cos(Lambda)
    sil2, csl2 = sin(2*Lambda), cos(2*Lambda)

    p0_plus = np.array([-sib**2*csl**2+sil**2, (sib**2+1)*(-sil*csl), sib*csb*csl,
                        (sib**2+1)*(-sil*csl), -sib**2*sil**2+csl**2, sib*csb*sil,
                        sib*csb*csl, sib*csb*sil, -csb**2]).reshape(3, 3)
    p0_cross = np.array([-sib*sil2, sib*csl2, csb*sil,
                         sib*csl2, sib*sil2, -csb*csl,
                         csb*sil, -csb*csl, 0]).reshape(3, 3)
    return p0_plus, p0_cross  # uu-vv, uv+vu


# Note1: One can use __slots__=('mass1', 'mass2', 'etc') to fix the attributes
#        then the class will not have __dict__ anymore, and attributes in __slots__ are read-only.
# Note2: One can use @DynamicAttrs to avoid warnings of 'no attribute'.
class BasicWaveform(object):
    """ The waveform class that contains basic parameters & methods,
    it is not a real waveform but only for inheriting. (see classes below)

    :param mass1: Primary mass (solar mass)
    :param mass2: Secondary mass(solar mass)
    :param T_obs: Observation time (s)
    :param DL: Luminosity distance (Mpc)
    :param Lambda: Longitude [0, 2pi]
    :param Beta: Latitude **[pi/2, -pi/2]** [instead of [0, pi]]
    :param phi_c: Coalescence phase [0, 2pi]
    :param tc: Coalescence time (s)
    :param iota: Inclination angle [0, pi]
    :param var_phi: Observer phase [0, 2pi]
    :param psi: Polarization angle [0, pi]
    :param kwargs: Additional parameters need to save
    """
    __slots__ = ('mass1', 'mass2', 'T_obs', 'DL', 'Lambda', 'Beta',
                 'phi_c', 'tc', 'iota', 'var_phi', 'psi', 'add_para')

    def __init__(self, mass1, mass2, T_obs, DL=1., Lambda=None, Beta=None,
                 phi_c=0., tc=0., iota=0., var_phi=0., psi=0., **kwargs):
        self.DL = DL
        self.mass1 = mass1
        self.mass2 = mass2
        self.Lambda = Lambda
        self.Beta = Beta
        self.phi_c = phi_c
        self.T_obs = T_obs
        self.tc = tc
        self.iota = iota
        self.var_phi = var_phi
        self.psi = psi
        self.add_para = kwargs

    # @property
    # def redshift(self):
    #     return float(dl_to_z(self.DL))

    # @property
    # def z(self):
    #     return self.redshift

    @property
    def Mt(self):
        return self.mass1 + self.mass2  # Total mass (solar mass)

    @property
    def eta(self):
        return self.mass1 * self.mass2 / self.Mt**2  # Symmetric mass ratio

    @property
    def Mc(self):
        return self.eta**(3/5) * self.Mt  # Chirp mass (solar mass)

    @property
    def vec_u(self):
        return np.array([sin(self.Lambda), -cos(self.Lambda), 0])

    @property
    def vec_v(self):
        return np.array([-sin(self.Beta)*cos(self.Lambda),
                         -sin(self.Beta)*sin(self.Lambda),
                         cos(self.Beta)])

    @property
    def vec_k(self):
        return np.array([-cos(self.Beta)*cos(self.Lambda),
                         -cos(self.Beta)*sin(self.Lambda),
                         -sin(self.Beta)])  # Vector of sources

    def polarization(self):
        p0_plus, p0_cross = p0_plus_cross(self.Lambda, self.Beta)
        p_plus = p0_plus*cos(2*self.psi) + p0_cross*sin(2*self.psi)
        p_cross = - p0_plus*sin(2*self.psi) + p0_cross*cos(2*self.psi)
        return p_plus, p_cross


class BurstWaveform(object):
    """ A sin-Gaussian waveforms for poorly modelled burst source. """

    def __init__(self, amp, tau, fc, tc=0):
        self.amp = amp
        self.tau = tau
        self.fc = fc
        self.tc = tc

    def get_hphc(self, tf):
        t = tf-self.tc
        h = (2/PI)**0.25 * self.tau**(-0.5) * self.amp
        h *= np.exp(- (t/self.tau)**2)*np.exp(2j*PI*self.fc*t)
        return h.real, h.imag


class BHBWaveform(BasicWaveform):
    """ Waveform for binary black holes (BHB).

    :param mass1: Primary mass (solar mass)
    :param mass2: Secondary mass(solar mass)
    :param T_obs: Observation time (s)
    :param DL: Luminosity distance (Mpc)
    :param Lambda: Longitude [0, 2pi]
    :param Beta: Latitude **[pi/2, -pi/2]** [instead of [0, pi]]
    :param phi_c: Coalescence phase [0, 2pi]
    :param tc: Coalescence time (s)
    :param iota: Inclination angle [0, pi]
    :param var_phi: Observer phase [0, 2pi]
    :param psi: Polarization angle [0, pi]
    :param chi1: Spin of the primary black hole (-1, 1)
    :param chi2: Spin of the secondary black hole (-1, 1)
    :param kwargs: Additional parameters need to save
    """
    __slots__ = ('chi1', 'chi2')
    # _true_para_key = ('DL', 'Mc', 'eta', 'chi1', 'chi2', 'phi_c', 'iota', 'tc', 'var_phi', 'psi', 'Lambda', 'Beta')
    # fisher_key = ('Mc', 'eta', 'chi1', 'chi2', 'DL', 'phi_c', 'iota', 'tc', 'Lambda', 'Beta', 'psi')

    def __init__(self, mass1, mass2, T_obs, DL=1., Lambda=None, Beta=None,
                 phi_c=0., tc=0., iota=0., var_phi=0., psi=0., chi1=0., chi2=0., **kwargs):

        BasicWaveform.__init__(self, mass1, mass2, T_obs, DL, Lambda, Beta,
                               phi_c, tc, iota, var_phi, psi, **kwargs)
        self.chi1 = chi1
        self.chi2 = chi2
        # if not det_frame_para:
        #     self.raw_source_masses = {'mass1': self.mass1, 'mass2': self.mass2,
        #                               'M': self.M, 'Mc': self.Mc}
        #     self.mass1 *= 1+self.z
        #     self.mass2 *= 1+self.z

    # def __eq__(self, other):
    #     return all([getattr(self, key) == getattr(other, key) for key in self._true_para_key])

    @property
    def f_min(self):
        return 5**(3/8)/(8*PI) * (MTSUN_SI*self.Mc)**(-5/8) * self.T_obs**(-3/8)

    def p_lm(self, l=2, m=2):
        """See Marsat et al. (Eq. 16) https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011"""
        p0_plus, p0_cross = p0_plus_cross(self.Lambda, self.Beta)
        y_lm = sYlm(-2, l, m, self.iota, self.var_phi)
        y_l_m_conj = sYlm(-2, l, -m, self.iota, self.var_phi).conjugate()
        return (1/2 * y_lm * np.exp(-2j*self.psi) * (p0_plus + 1j*p0_cross) +
                1/2 * (-1)**l * y_l_m_conj * np.exp(2j*self.psi) * (p0_plus - 1j*p0_cross))

    @property
    def p_22(self):
        """Equals to self.p_lm(2, 2)"""
        p0_plus, p0_cross = p0_plus_cross(self.Lambda, self.Beta)
        y22 = sqrt(5/4/PI) * cos(self.iota/2)**4 * np.exp(2j*self.var_phi)
        y2_2_conj = sqrt(5/4/PI) * sin(self.iota/2)**4 * np.exp(2j*self.var_phi)

        return (1/2 * y22 * np.exp(-2j*self.psi) * (p0_plus + 1j*p0_cross) +
                1/2 * y2_2_conj * np.exp(2j*self.psi) * (p0_plus - 1j*p0_cross))

    def wave_para_phenomd(self, f_ref=0.):
        """Convert parameters to a list, specially for getting waveform from `pyIMRPhenomD`."""
        phi_ref = self.phi_c
        m1_si = self.mass1 * MSUN_SI
        m2_si = self.mass2 * MSUN_SI
        chi1, chi2 = self.chi1, self.chi2
        dl_si = self.DL * MPC_SI
        return phi_ref, f_ref, m1_si, m2_si, chi1, chi2, dl_si

    def get_amp_phase(self, freq, f_ref=0.):
        """ Generate amp and phase in frequency domain.

        :param freq: Frequency list
        :param f_ref: Reference frequency (default: 0.)
        :return: (amp, phase, tf (, tfp)): (amplitude, phase, t of f (, dt/df))
        """
        if use_py_phd:
            NF = freq.shape[0]

            amp_imr = np.zeros(NF)
            phase_imr = np.zeros(NF)
            if pyIMRc.findT:
                time_imr = np.zeros(NF)
                timep_imr = np.zeros(NF)
            else:
                time_imr = np.zeros(0)
                timep_imr = np.zeros(0)
            
            t0 = self.tc
            # Create structure for Amp/phase/time FD waveform
            h22 = pyIMRD.AmpPhaseFDWaveform(NF, freq, amp_imr, phase_imr, time_imr, timep_imr, f_ref, t0)
            # Generate h22 FD amplitude and phase on a given set of frequencies
            h22 = pyIMRD.IMRPhenomDGenerateh22FDAmpPhase(h22, freq, *self.wave_para_phenomd(f_ref))

            amp = {(2, 2): h22.amp}
            phase = {(2, 2): h22.phase}
            tf = {(2, 2): h22.time}
            # tfp = {(2, 2): h22.timep}
        else:
            wf_phd_class = pyIMRD(freq, *self.wave_para_phenomd(f_ref))
            freq, amp_22, phase_22 = wf_phd_class.GetWaveform()
            # Note: these are actually cython pointers, we should also use .copy() to acquire ownership
            freq, amp_22, phase_22 = freq.copy(), amp_22.copy(), phase_22.copy()
            tf_spline = Spline(freq, 1/(2*PI)*(phase_22 - phase_22[0])).derivative()
            tf_22 = tf_spline(freq) + self.tc
            amp = {(2, 2): amp_22}
            phase = {(2, 2): phase_22}
            tf = {(2, 2): tf_22}

        return amp, phase, tf


class BHBWaveformEcc(BasicWaveform):
    """ BHBWaveform including eccentricity, using `EccentricFD` Waveform.

    :param mass1: Primary mass (solar mass)
    :param mass2: Secondary mass(solar mass)
    :param T_obs: Observation time (s)
    :param DL: Luminosity distance (Mpc)
    :param Lambda: Longitude [0, 2pi]
    :param Beta: Latitude **[pi/2, -pi/2]** [instead of [0, pi]]
    :param phi_c: Coalescence phase [0, 2pi]
    :param tc: Coalescence time (s)
    :param iota: Inclination angle [0, pi]
    :param var_phi: Observer phase [0, 2pi]
    :param psi: Polarization angle [0, pi]
    :param eccentricity: initial eccentricity at f_min, [0, 0.4)
    :param kwargs: Additional parameters need to save
    """
    __slots__ = 'eccentricity'

    def __init__(self, mass1, mass2, T_obs, DL=1., Lambda=None, Beta=None,
                 phi_c=0., tc=0., iota=0., var_phi=0., psi=0., eccentricity=0., **kwargs):
        BasicWaveform.__init__(self, mass1, mass2, T_obs, DL, Lambda, Beta,
                               phi_c, tc, iota, var_phi, psi, **kwargs)
        self.eccentricity = eccentricity

    @property
    def f_min(self):
        return 5**(3/8)/(8*PI) * (MTSUN_SI*self.Mc)**(-5/8) * self.T_obs**(-3/8)

    def wave_para(self):
        args = {'mass1': self.mass1*MSUN_SI,
                'mass2': self.mass2*MSUN_SI,
                'distance': self.DL*MPC_SI,
                'coa_phase': self.phi_c,
                'inclination': self.iota,
                'long_asc_nodes': self.var_phi,
                'eccentricity': self.eccentricity}
        return args

    def gen_ori_waveform(self, delta_f=None, f_min=None, f_max=1., hphc=False):
        """ Generate F-Domain eccentric waveform for TDI response. (EccentricFD) """
        if not f_min:
            f_min = self.f_min
        if delta_f is None:
            delta_f = 1/self.T_obs

        if hphc:
            return gen_ecc_fd_waveform(**self.wave_para(), delta_f=delta_f,
                                       f_lower=f_min, f_final=f_max, obs_time=0)
        return gen_ecc_fd_and_tf(self.tc, **self.wave_para(), delta_f=delta_f,
                                 f_lower=f_min, f_final=f_max, obs_time=0)

    def fd_tdi_response(self, channel='A', det='TQ', delta_f=None, f_min=None, f_max=1., **kwargs):
        """ Generate F-Domain TDI response for eccentric waveform (EccentricFD).
         Although the eccentric waveform also have (l, m)=(2,2), it has eccentric harmonics,
         which should also calculate separately like what we should do for spherical harmonics."""
        if det not in detectors.keys():
            raise ValueError(f"Unknown detector {det}. "
                             f"Supported detectors: {'|'.join(detectors.keys())}")
        # if channel not in 'XYZAET':  # <if not all([c in 'XYZAET' for c in channel])> for multichannel mode
        #     raise ValueError(f"Unknown channel {channel}. "
        #                      f"Supported channels: {'|'.join(['X', 'Y', 'Z', 'A', 'E', 'T'])}")
        det_class = detectors[det]
        wf, freq = self.gen_ori_waveform(delta_f, f_min, f_max)

        gw_tdi = np.zeros(shape=(len(freq), ), dtype=np.complex128)
        t_delay = np.exp(2j*PI*freq*self.tc)
        p_p, p_c = self.polarization()
        for i in range(10):
            h_p, h_c, tf_vec = wf[i]
            index = (h_p != 0).argmax()

            det = det_class(tf_vec[index:], **kwargs)
            # FIXME: use channel!!!, here only store A channel
            gw_tdi_p, gw_tdi_c = trans_AET_fd(self.vec_k, (p_p, p_c), det, freq[index:])
            gw_tdi[index:] += gw_tdi_p[0]*h_p[index:] + gw_tdi_c[0]*h_c[index:]

        return gw_tdi*t_delay, freq


class GCBWaveform(BasicWaveform):
    """ Waveform for GCB.

    :param mass1: Primary mass (solar mass)
    :param mass2: Secondary mass(solar mass)
    :param T_obs: Observation time (s)
    :param phi0: Initial phase at t = 0
    :param f0: Frequency of the source
    :param fdot: Derivative of frequency, df/dt (default: None, calculated physically)
    :param fddot: Double derivative of frequency, d^2f/dt^2 (default: None, calculated physically)
    :param DL: Luminosity distance (Mpc)
    :param Lambda: Longitude [0, 2pi]
    :param Beta: Latitude **[pi/2, -pi/2]** [instead of [0, pi]]
    :param iota: Inclination angle [0, pi]
    :param var_phi: Observer phase [0, 2pi]
    :param psi: Polarization angle [0, pi]
    :param kwargs: Additional parameters need to save
    """
    __slots__ = ('phi0', 'f0', 'fdot', 'fddot')

    def __init__(self, mass1, mass2, T_obs, phi0, f0, fdot=None, fddot=None,
                 DL=1., Lambda=None, Beta=None, iota=0., var_phi=0., psi=0, **kwargs):
        BasicWaveform.__init__(self, mass1, mass2, T_obs, DL, Lambda, Beta,
                               np.inf, np.inf, iota, var_phi, psi, **kwargs)
        self.phi0 = phi0
        self.f0 = f0
        if fdot is None:
            self.fdot = 96/5*PI**(8/3) * (self.Mc*MTSUN_SI)**(5/3) * f0**(11/3)
        else:
            self.fdot = fdot
        if fddot is None:
            self.fddot = 11/3*self.fdot**2/f0
        else:
            self.fddot = fddot

    @property
    def amplitude(self):
        return 2*(self.Mc*MTSUN_SI)**(5/3) * C_SI/(self.DL*MPC_SI) * (PI*self.f0)**(2/3)

    def get_hphc(self, t):
        # FIXME: use self.T_obs
        phase = 2*PI*(self.f0 + 0.5*self.fdot*t + 1/6*self.fddot*t*t)*t + self.phi0
        csi = cos(self.iota)

        hp = self.amplitude*cos(phase)*(1+csi*csi)
        hc = self.amplitude*sin(phase)*2*csi
        return hp, hc

    def _buffer_size(self, det, oversample=1):
        if det == 'TianQin':
            fm = 3.1771266198541054e-6  # fsc_tq
        elif det in ('LISA', 'Taiji'):
            fm = 3.168753578692357e-8  # EarthOrbitFreq_SI
        else:
            raise ValueError(f"Unknown detector {det}.")
        # N=f*T, minimal f should be >=2*f_max according to Nyquist sampling theorem,
        # choose 3 for a conservative estimate.
        N = (3*2*fm) * self.T_obs
        N = 1 << int(np.ceil(np.log2(N)))  # next power of 2
        return N*oversample

    def get_fastgb_fd_single(self, dt, oversample=1, detector='TianQin', buffer=None):
        """ Calculate the GCB waveform using fast/slow decomposition.

        :param dt: Sampling step of time
        :param oversample: Should be a power of 2, or it will be not able to use gsl_fft
        :param detector:
        :param buffer: Should be a tuple with
        """
        # FIXME: assume T=T_obs below
        N = self._buffer_size(detector, oversample)

        XLS = np.zeros(2*N, 'd')
        YLS = np.zeros(2*N, 'd')
        ZLS = np.zeros(2*N, 'd')

        XSL = np.zeros(2*N, 'd')
        YSL = np.zeros(2*N, 'd')
        ZSL = np.zeros(2*N, 'd')

        params = np.array([self.f0, self.fdot, self.Beta, self.Lambda, self.amplitude, self.iota, self.psi, self.phi0])

        if np.all(params) is not None:
            libFastGB.ComputeXYZ_FD(params, N, self.T_obs, dt, XLS, YLS, ZLS, XSL, YSL, ZSL,
                                    len(params), detector=detector)
            # TODO Need to transform to SL if required
            # Xf, Yf, Zf = XLS, YLS, ZLS
            Xf, Yf, Zf = XSL, YSL, ZSL
        else:
            raise ValueError

        Xf, Yf, Zf = Xf.view(np.complex128), Yf.view(np.complex128), Zf.view(np.complex128)
        if buffer is None:
            kmin = int(self.f0*self.T_obs-N/2)
            df = 1.0/self.T_obs
            f_range = np.linspace(kmin*df, (kmin+N-1)*df, N)
            return f_range, Xf, Yf, Zf
        else:
            blen, alen = len(buffer[0]), N

            # for a full buffer, "a" begins and ends at these indices
            beg, end = int(self.f0*self.T_obs-N/2), int(self.f0*self.T_obs+N/2)
            # alignment of partial buffer with "a"
            begb, bega = max(beg, 0), max(0, -beg)
            endb, enda = min(end, blen), alen-max(0, end-blen)

            for i, a in enumerate((Xf, Yf, Zf)):
                buffer[i][begb:endb] += a[bega:enda]

    def get_fastgb_fd(self, dt, oversample=1, detector='TianQin'):
        length = int(0.5*self.T_obs/dt)+1  # NFFT=int(T/dt), length=NFFT/2+1
        buffer = tuple(np.zeros(length, dtype=np.complex128) for _ in range(3))

        # for _ in table: TODO: make it support multiple sources?
        self.get_fastgb_fd_single(dt, oversample, detector, buffer)
        f = np.linspace(0, (length-1)*1.0/self.T_obs, length)
        return (f, ) + buffer

    def get_fastgb_td(self, dt, oversample=1, detector='TianQin'):
        f, X, Y, Z = self.get_fastgb_fd(dt, oversample, detector)
        df = 1.0/self.T_obs
        kmin = round(f[0]/df)

        def ifft(arr):
            # n = int(1.0/(dt*df))
            n = round(1.0/(dt*df))
            # by liyn (in case the int() function would cause loss of n)

            ret = np.zeros(int(n/2+1), dtype=arr.dtype)
            ret[kmin:kmin+len(arr)] = arr[:]
            ret *= n  # normalization, ehm, found empirically

            return np.fft.irfft(ret)

        X, Y, Z = ifft(X), ifft(Y), ifft(Z)
        t = np.arange(len(X))*dt
        return t, X, Y, Z


class EMRIWaveform(BasicWaveform):
    r""" Waveform for EMRI.

    :param M: (double) Mass of larger black hole in solar masses.
    :param mu: (double) Mass of compact object in solar masses.
    :param a: (double) Dimensionless spin of massive black hole.
    :param p0: (double) Initial semilatus rectum (Must be greater than
       the separatrix at the given e0 and x0).
       See documentation for more information on :math:`p_0<10`.
    :param e0: (double) Initial eccentricity.
    :param x0: (double) Initial cosine of the inclination angle.
       (:math:`x_I=\cos{I}`). This differs from :math:`Y=\cos{\iota}\equiv L_z/\sqrt{L_z^2 + Q}`
       used in the semi-relativistic formulation. When running kludge waveforms,
       :math:`x_{I,0}` will be converted to :math:`Y_0`.
    :param dist: (double) Luminosity distance in Gpc.
    :param qS: (double) Sky location polar angle in ecliptic coordinates.
    :param phiS: (double) Sky location azimuthal angle in ecliptic coordinates.
    :param qK: (double) Initial BH spin polar angle in ecliptic coordinates.
    :param phiK: (double) Initial BH spin azimuthal angle in ecliptic coordinates.
    :param Phi_phi0: (double, optional) Initial phase for :math:`\Phi_\phi`. Default is 0.0.
    :param Phi_theta0: (double, optional) Initial phase for :math:`\Phi_\Theta`. Default is 0.0.
    :param Phi_r0: (double, optional) Initial phase for :math:`\Phi_r`. Default is 0.0.
    :param kwargs: (dict, optional) Additional parameters need to save
    """

    def __init__(self, M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, T_obs,
                 Phi_phi0=0, Phi_theta0=0, Phi_r0=0, **kwargs):
        self.M = M
        self.mu = mu
        self.a = a
        self.p0 = p0
        self.e0 = e0
        self.x0 = x0
        self.dist = dist

        self.qS = qS
        self.phiS = phiS
        self.qK = qK
        self.phiK = phiK
        self.Phi_phi0 = Phi_phi0
        self.Phi_theta0 = Phi_theta0
        self.Phi_r0 = Phi_r0

        self.wave_func = self._gen_wave_func()
        self.theta, self.phi = self.wave_func._get_viewing_angles(qS, phiS, qK, phiK)  # get view angle
        BasicWaveform.__init__(self, M, mu, T_obs, dist*1000., Lambda=self.phi, Beta=PI_2-self.theta, **kwargs)

    @staticmethod
    def _gen_wave_func():
        from few.waveform import GenerateEMRIWaveform

        model = "FastSchwarzschildEccentricFlux"
        use_gpu = False
        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {"DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
                           "max_init_len": int(1e3)}  # all the trajectories will be well under len = 1000
        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {"use_gpu": use_gpu,
                            "max_init_len": int(1e3)}  # all the trajectories will be well under len = 1000
        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {"assume_positive_m": False}  # if we assume positive m, it will generate negative m for all m>0
        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"use_gpu": use_gpu, "pad_output": False}
        return GenerateEMRIWaveform(model, use_gpu=use_gpu, inspiral_kwargs=inspiral_kwargs,
                                    amplitude_kwargs=amplitude_kwargs, Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs)

    def get_harmonic_mode(self, eps=1e-5, model_insp="SchwarzEccFlux", use_gpu=False):
        """ Calculate harmonic modes

        :param eps: tolerance on mode contribution to total power
        :param model_insp: str (default: "SchwarzEccFlux")
        :param use_gpu: bool (default: False)
        """
        from few.trajectory.inspiral import EMRIInspiral
        from few.amplitude.romannet import RomanAmplitude
        from few.utils.ylm import GetYlms
        from few.utils.modeselector import ModeSelector

        # first, lets get amplitudes for a trajectory
        traj = EMRIInspiral(func=model_insp)
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(self.M, self.mu, self.a, self.p0, self.e0, 1.0)

        # get amplitudes along trajectory
        amp = RomanAmplitude()

        teuk_modes = amp(p, e)
        # get ylms
        ylm_gen = GetYlms(assume_positive_m=True, use_gpu=use_gpu)
        ylms = ylm_gen(amp.unique_l, amp.unique_m, self.theta, self.phi).copy()[amp.inverse_lm]
        modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]
        mode_selector = ModeSelector(amp.m0mask, use_gpu=use_gpu)

        (teuk_modes_in, ylms_in, ls, ms, ns) = mode_selector(teuk_modes, ylms, modeinds, eps=eps)
        return teuk_modes_in, ylms_in, ls, ms, ns

    def get_hphc_source(self, T_obs, dt, eps=1e-5, modes=None):
        """ Calculate the time domain waveform. TODO: use T_obs

        :param T_obs: the observation time in [year]
        :param dt: sampling time in [s]
        :param modes: (str or list or None)
            - If None, perform our base mode filtering with eps as the fractional accuracy on the total power.
            - If ‘all’, it will run all modes without filtering.
            - If a list of tuples (or lists) of mode indices (e.g. [(l1,m1,n1), (l2,m2,n2)]) is provided,
                it will return those modes combined into a single waveform.
        :param eps: Controls the fractional accuracy during mode filtering.
            Raising this parameter will remove modes.
            Lowering this parameter will add modes.
            Default that gives a good overlap is 1e-5.
        :return: (hp, hc) (numpy ndarray, numpy ndarray)
        """
        para_list = (self.M, self.mu, self.a, self.p0, self.e0, self.x0, self.dist,
                     self.qS, self.phiS, self.qK, self.phiK, self.Phi_phi0, self.Phi_theta0, self.Phi_r0)
        h = self.wave_func(*para_list, T=T_obs, dt=dt, eps=eps, mode_selection=modes)
        return h.real, h.imag

    def get_hphc(self, tf, eps=1e-5, modes=None):
        Tobs = tf[-1]/YRSID_SI
        dt = tf[1]-tf[0]
        # T = Tobs - int(Tobs * YRSID_SI/dt - tf.shape[0]) * dt/YRSID_SI
        # print("the total observ time is ", Tobs)
        hpS, hcS = self.get_hphc_source(Tobs, dt, eps, modes)

        tf_size = tf.shape[0]
        h_size = hpS.shape[0]
        if tf_size > h_size:
            hp = np.zeros_like(tf)
            hc = np.zeros_like(tf)
            hp[:h_size] = hpS
            hc[:h_size] = hcS
        elif tf_size < h_size:
            hp = hpS[-tf_size:]
            hc = hcS[-tf_size:]
        else:
            hp = hpS
            hc = hcS
        return hp, hc


waveforms = {'burst': BurstWaveform,
             'bhb_PhenomD': BHBWaveform,
             'bhb_EccFD': BHBWaveformEcc,
             'gcb': GCBWaveform,
             'emri': EMRIWaveform,
             }  # all available waveforms
