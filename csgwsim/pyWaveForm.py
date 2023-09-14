#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyWaveForm.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 12:32:36
# ==================================

# import os
# import sys
import numpy as np
from numpy import sin, cos, sqrt

from utils import to_m1m2
from Constants import MSUN_SI, MSUN_unit, MPC_SI, YRSID_SI, PI, C_SI, G_SI
from Waveforms.PyIMRPhenomD import IMRPhenomD as pyIMRD
from Waveforms.PyIMRPhenomD import IMRPhenomD_const as PyIMRC
from Waveforms.FastEMRIWaveforms.FastEMRI import *

# try:
#     import bbh
# except:
#     bbh_path = './Waveforms/bbh'
#     abs_bbh_path = os.path.abspath(bbh_path)
#
#     if abs_bbh_path not in sys.path:
#         sys.path.append(abs_bbh_path)
#
# try:
#     import few
# except:
#     few_path = './Waveforms/FastEMRIWaveforms'
#     abs_few_path = os.path.abspath(few_path)
#
#     if abs_few_path not in sys.path:
#         sys.path.append(abs_few_path)
# try:
#     from FastEMRI import *
# except:
#     few_path = './Waveforms/FastEMRIWaveforms'
#     abs_few_path = os.path.abspath(few_path)
#
#     if abs_few_path not in sys.path:
#         sys.path.append(abs_few_path)
#
#     from FastEMRI import *


# Note1: one can use __slots__=('mass1', 'mass2', 'etc') to fix the attributes
#        then the class will not have __dict__ anymore, and attributes in __slots__ are read-only.
# Note2: One can use @DynamicAttrs to avoid warnings of 'no attribute'.
class BasicWaveform(object):
    """
    Class for waveform
    -------------------------------
    Parameters:
    - pars: dict of parameters for different sources
    such as:
        - type: GCB; BHB; EMRI; SGWB for different sources
        - lambda: longitude of the source in ecliptic coordinates
        - beta: latitude of the source in ecliptic coordinates
        - psi: polarization angle
        - iota: inclination angle
        - Mc: chirp mass
        - DL: luminosity distance
        - etc
    """
    __slots__ = ('DL', 'mass1', 'mass2', 'Lambda', 'Beta', 'phi_c',
                 'T_obs', 'tc', 'iota', 'var_phi', 'psi', 'add_para')

    def __init__(self, mass1, mass2, T_obs=None, DL=1., Lambda=None, Beta=None,
                 phi_c=0., tc=0., iota=0., var_phi=0., psi=0, **kwargs):
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

        if (self.mass1 is None) or (self.mass2 is None) or (self.T_obs is None):
            raise ValueError('mass1, mass2 and T_obs should NOT be None')

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

    def _p0(self):
        """See "LDC-manual-002.pdf" (Eq. 12, 13) & Marsat et al. (Eq. 14)"""
        sib, csb = sin(self.Beta), cos(self.Beta)
        sil, csl = sin(self.Lambda), cos(self.Lambda)
        sil2, csl2 = sin(2*self.Lambda), cos(2*self.Lambda)

        p0_plus = np.array([-sib**2 * csl**2 + sil**2, (sib**2+1)*(-sil*csl),  sib*csb*csl,
                            (sib**2+1)*(-sil*csl),     -sib**2*sil**2+csl**2,  sib*csb*sil,
                            sib*csb*csl,                sib*csb*sil,          -csb**2]).reshape(3, 3)
        p0_cross = np.array([-sib*sil2, sib*csl2,  csb*sil,
                             sib*csl2,  sib*sil2, -csb*csl,
                             csb*sil,  -csb*csl,   0]).reshape(3, 3)
        return p0_plus, p0_cross

    def polarization(self):
        """See "LDC-manual-002.pdf" (Eq. 19)"""
        p0_plus, p0_cross = self._p0()
        p_plus = p0_plus*cos(2*self.psi) + p0_cross*sin(2*self.psi)
        p_cross = - p0_plus*sin(2*self.psi) + p0_cross*cos(2*self.psi)
        return p_plus, p_cross


class BurstWaveform(object):
    """
    A sin-Gaussian waveforms for poorly modelled burst source
    --------------------------------------------------------
    """

    def __init__(self, amp, tau, fc, tc=0):
        self.amp = amp
        self.tau = tau
        self.fc = fc
        self.tc = tc

    def __call__(self, tf):
        t = tf-self.tc
        h = (2/np.pi)**0.25 * self.tau**(-0.5) * self.amp
        h *= np.exp(- (t/self.tau)**2)*np.exp(2j*np.pi*self.fc*t)
        return h.real, h.imag


class BHBWaveform(BasicWaveform):
    """
    This is Waveform for BHB
    ------------------------
    Parameters:
    - m1, m2: mass of black holes
    - chi1, chi2: spin of the two black holes
    - DL: in MPC
    """
    __slots__ = ('chi1', 'chi2', 'ra', 'dec', 'h22')
    # _true_para_key = ('DL', 'Mc', 'eta', 'chi1', 'chi2', 'phi_c', 'iota', 'tc', 'var_phi', 'psi', 'Lambda', 'Beta')
    # fisher_key = ('Mc', 'eta', 'chi1', 'chi2', 'DL', 'phi_c', 'iota', 'tc', 'Lambda', 'Beta', 'psi')

    def __init__(self, mass1, mass2, DL=1., Lambda=None, Beta=None,
                 phi_c=0., T_obs=None, tc=0., iota=0., var_phi=0., psi=0., chi1=0., chi2=0.,
                 ra=None, dec=None, **kwargs):

        BasicWaveform.__init__(self, mass1, mass2, DL, Lambda, Beta,
                               phi_c, T_obs, tc, iota, var_phi, psi, **kwargs)
        self.chi1 = chi1
        self.chi2 = chi2
        self.ra = ra
        self.dec = dec
        # self.MfRef_in = MfRef_in
        # self.fRef = fRef  # 0.

        if (self.Lambda is not None) and (self.Beta is not None):
            # print(f'[WaveParas]<Sky location is given by Ecliptic frame:'
            #       f'\n(lon, lat) in rad:({self.Lambda:.3f}, {self.Beta:.3f})>')
            pass
        else:
            try:
                from utils import icrs_to_ecliptic
                self.Lambda, self.Beta = icrs_to_ecliptic(ra, dec)
                print(f'[WaveParas]<Sky location is given by ICRS frame:'
                      f'\n(ra, dec) in rad:({ra:.3f}, {dec:.3f})>')
            except AttributeError:
                raise ValueError('ParameterClass without *valid* sky location parameters!') from None

        # if not det_frame_para:
        #     self.raw_source_masses = {'mass1': self.mass1, 'mass2': self.mass2,
        #                               'M': self.M, 'Mc': self.Mc}
        #     self.mass1 *= 1+self.z
        #     self.mass2 *= 1+self.z

    # def __eq__(self, other):
    #     return all([getattr(self, key) == getattr(other, key) for key in self._true_para_key])

    @property
    def f_min(self):
        return 5**(3/8)/(8*np.pi) * (MSUN_unit*self.Mc)**(-5/8) * self.T_obs**(-3/8)

    def _y22(self):
        """See "LDC-manual-002.pdf" (Eq. 31)"""
        y22_o = sqrt(5/4/PI) * cos(self.iota/2)**4 * np.exp(2j*self.var_phi)
        y2_2_conj = sqrt(5/4/PI) * sin(self.iota/2)**4 * np.exp(2j*self.var_phi)
        return y22_o, y2_2_conj

    # p_lm(self, l=2, m=2):
    #     y_lm_o = spin_weighted_spherical_harmonic(-2, l, m, self.iota, self.var_phi)
    #     y_l_m_conj = spin_weighted_spherical_harmonic(-2, l, -m, self.iota, self.var_phi).conjugate()
    @property
    def p22(self):
        """See Marsat et al. (Eq. 16) https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011"""
        y22_o, y2_2_conj = self._y22()
        p0_plus, p0_cross = self._p0()
        
        return (1/2 * y22_o * np.exp(-2j*self.psi) * (p0_plus + 1j*p0_cross) +
                1/2 * y2_2_conj * np.exp(2j*self.psi) * (p0_plus - 1j*p0_cross))

    def h22_FD(self, freq, fRef=0., t0=0.):
        NF = freq.shape[0]

        amp_imr = np.zeros(NF)
        phase_imr = np.zeros(NF)
        if PyIMRC.findT:
            time_imr = np.zeros(NF)
            timep_imr = np.zeros(NF)
        else:
            time_imr = np.zeros(0)
            timep_imr = np.zeros(0)

        # Create structure for Amp/phase/time FD waveform
        self.h22 = pyIMRD.AmpPhaseFDWaveform(NF, freq, amp_imr, phase_imr, time_imr, timep_imr, fRef, t0)

        # Generate h22 FD amplitude and phase on a given set of frequencies
        self.h22 = pyIMRD.IMRPhenomDGenerateh22FDAmpPhase(
            self.h22, freq,
            self.phi_c, self.MfRef_in,
            self.mass1*MSUN_SI, self.mass2*MSUN_SI,
            self.chi1, self.chi2,
            self.DL*MPC_SI)

        return self.h22

    def amp_phase(self, freq, mode):
        """
        Generate the amp and phase in frequency domain
        ----------------------------------------------
        Parameters:
        -----------
        - freq: frequency list
        - mode: mode of GW
        # FIXME: Default argument value is mutable if mode=[(2, 2)], use tuple instead, btw it is unused

        Return:
        -------
        - amp:
        - phase:
        - tf: time of freq
        - tfp: dt/df
        """
        h22 = self.h22_FD(freq, self.fRef, self.tc)

        amp = {(2, 2): h22.amp}
        phase = {(2, 2): h22.phase}
        tf = {(2, 2): h22.time}
        tfp = {(2, 2): h22.timep}

        return amp, phase, tf, tfp

    # def gen_ori_waveform(self, delta_f=None, f_min=None, f_max=1.):
    #     """Generate f-domain TDI waveform(IMRPhenomD, h22 mode)"""
    #     from pyIMRPhenomD import IMRPhenomDh22AmpPhase
    #     if delta_f is None:
    #         delta_f = 1/self.T_obs
    #     freq_phd = np.arange(np.ceil(f_min/delta_f)*delta_f, f_max, delta_f)
    #     # freq_phd = np.arange(f_min, f_max, delta_f)  # TODO
    #     wf_phd_class = IMRPhenomDh22AmpPhase(freq_phd, *self.wave_para_phenomd())
    #     freq, amp, phase = wf_phd_class.GetWaveform()  # freq, amp, phase
    #     # these are actually cython pointers, we should also use .copy() to acquire ownership
    #     wf = (freq.copy(), amp.copy(), phase.copy())
    #
    #     return wf


class BHBWaveformEcc(BHBWaveform):
    pass


class GCBWaveform(BasicWaveform):
    """
    This is Waveform for GCB.
    ------------------------
    Parameters:
    - Mc: chirp mass
    - DL: luminosity distance
    - phi0: initial phase at t = 0
    - f0: frequency of the source
    - fdot: derivative of frequency: df/dt
        - default: None, calculated physically
    - fddot: double derivative of frequency: d^2f/dt^2
        - default: None, calculated physically
    --------------------------
    How to call it:
    ```python
    ```
    tf = np.arange(0,Tobs, delta_T)
    GCB = GCBWaveform(Mc=0.5, DL=0.3, phi0=0, f0=0.001)
    hpS, hcS = GCB(tf)
    """

    def __init__(self, Mc, DL, phi0, f0, fdot=None, fddot=None, **kwargs):
        eta = 0.25  # FIXME
        m1, m2 = to_m1m2(Mc, eta)
        BasicWaveform.__init__(self, m1, m2, DL, **kwargs)
        self.phi0 = phi0
        self.f0 = f0
        # self.fdot = fdot
        if fdot is None:
            self.fdot = (96/5*PI**(8/3) *
                         (G_SI*Mc*MSUN_SI/C_SI**3)**(5/3)
                         * f0**(11/3))
        else:
            self.fdot = fdot
        if fddot is None:
            self.fddot = 11/3*self.fdot**2/f0
        else:
            self.fddot = fddot
        self.amp = 2*(G_SI*Mc*MSUN_SI)**(5/3)
        self.amp = self.amp/C_SI**4/(DL*MPC_SI)
        self.amp = self.amp*(PI*f0)**(2/3)

    def get_hphc(self, t): # FIXME // name of the fucntion
        phase = 2*PI*(self.f0+0.5*self.fdot*t +
                      1/6*self.fddot*t*t)*t+self.phi0
        hp = self.amp*cos(phase)
        hc = self.amp*sin(phase)

        # TODO: What do we really want from __call__? Original wf or SSB wf?
        #  Could we not use __call__ but add two normal methods in class?
        cs2p = cos(2*self.psi)
        sn2p = sin(2*self.psi)
        csi = cos(self.iota)

        hp_SSB = -(1+csi*csi)*hp*cs2p+2*csi*hc*sn2p
        hc_SSB = -(1+csi*csi)*hp*sn2p-2*csi*hc*cs2p

        return hp_SSB, hc_SSB


class FastGB(GCBWaveform):
    """
    Calculate the GCB waveform using fast/slow TODO
    """

    def __init__(self, N, Mc, DL, phi0, f0, **kwargs):
        super().__init__(Mc, DL, phi0, f0, **kwargs)
        self.N = N


class EMRIWaveform(object):
    """
    This is waveform for EMRI
    --------------------------
    Parameters:
    - M (double): Mass of larger black hole in solar masses.
    - mu (double): Mass of compact object in solar masses.
    - a (double): Dimensionless spin of massive black hole.
    - p0 (double): Initial semilatus rectum (Must be greater than
        the separatrix at the given e0 and x0).
        See documentation for more information on :math:`p_0<10`.
    - e0 (double): Initial eccentricity.
    - x0 (double): Initial cosine of the inclination angle.
        (:math:`x_I=\cos{I}`). This differs from :math:`Y=\cos{\iota}\equiv L_z/\sqrt{L_z^2 + Q}`
        used in the semi-relativistic formulation. When running kludge waveforms,
        :math:`x_{I,0}` will be converted to :math:`Y_0`.
    - dist (double): Luminosity distance in Gpc.
    - qS (double): Sky location polar angle in ecliptic
        coordinates.
    - phiS (double): Sky location azimuthal angle in
        ecliptic coordinates.
    - qK (double): Initial BH spin polar angle in ecliptic
        coordinates.
    - phiK (double): Initial BH spin azimuthal angle in
        ecliptic coordinates.
    - Phi_phi0 (double, optional): Initial phase for :math:`\Phi_\phi`.
        Default is 0.0.
    - Phi_theta0 (double, optional): Initial phase for :math:`\Phi_\Theta`.
        Default is 0.0.
    - Phi_r0 (double, optional): Initial phase for :math:`\Phi_r`.
        Default is 0.0.
    - *args (tuple, optional): Tuple of any extra parameters that go into the model.
    - **kwargs (dict, optional): Dictionary with kwargs for online waveform
        generation.
    """

    def __init__(self, M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK,
                 Phi_phi0=0, Phi_theta0=0, Phi_r0=0,
                 model="FastSchwarzschildEccentricFlux",
                 model_insp="SchwarzEccFlux",
                 inspiral_kwargs=inspiral_kwargs,
                 amplitude_kwargs=amplitude_kwargs,
                 Ylm_kwargs=Ylm_kwargs,
                 sum_kwargs=sum_kwargs,
                 use_gpu=use_gpu):  # TODO: make it a subclass of BasicWaveform
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

        self.gen_wave = GenerateEMRIWaveform(
            model,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
        )
        self.theta, self.phi = self.gen_wave._get_viewing_angles(qS, phiS, qK, phiK)  # get view angle

        # first, lets get amplitudes for a trajectory
        self.traj = EMRIInspiral(func=model_insp)
        self.ylm_gen = GetYlms(assume_positive_m=True, use_gpu=use_gpu)

    def get_harmonic_mode(self, eps=1e-5):
        """
        To calculate how many harmonic mode
        -----------------------------------
        Parameters:
        - eps: tolerance on mode contribution to total power
        """
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = self.traj(self.M, self.mu, self.a, self.p0, self.e0, 1.0)

        # get amplitudes along trajectory
        amp = RomanAmplitude()

        teuk_modes = amp(p, e)

        theta, phi = self.gen_wave._get_viewing_angles(qS, phiS, qK, phiK)  # get view angle
        # get ylms
        ylms = self.ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

        mode_selector = ModeSelector(amp.m0mask, use_gpu=False)

        modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]

        (teuk_modes_in, ylms_in, ls, ms, ns) = mode_selector(teuk_modes, ylms, modeinds, eps=eps)
        return teuk_modes_in, ylms_in, ls, ms, ns

    def get_hphc(self, Tobs, dt, eps=1e-5, modes=None): # FIXME // change name
        """
        Calculate the time domain waveforms
        -----------------------------------
        Return:
        - hp, hc

        Parameters:
        - Tobs: the observation time in [year]
        - dt: sampling time in [s]
        - modes: (str or list or None)
            - If None, perform our base mode filtering with eps as the fractional accuracy on the total power.
            - If ‘all’, it will run all modes without filtering.
            - If a list of tuples (or lists) of mode indices (e.g. [(l1,m1,n1), (l2,m2,n2)]) is provided,
                it will return those modes combined into a single waveform.
        - eps: Controls the fractional accuracy during mode filtering.
            Raising this parameter will remove modes.
            Lowering this parameter will add modes.
            Default that gives a good overlap is 1e-5.
        """
        h = self.gen_wave(
            self.M,
            self.mu,
            self.a,
            self.p0,
            self.e0,
            self.x0,
            self.dist,
            self.qS,
            self.phiS,
            self.qK,
            self.phiK,
            self.Phi_phi0,
            self.Phi_theta0,
            self.Phi_r0,
            T=Tobs,
            dt=dt,
            eps=eps,
            mode_selection=modes,
        )

        return h.real, h.imag

    def get_ssb_wf(self, tf, eps=1e-5, modes=None):
        # TODO: What do we really want from __call__? Original wf or SSB wf?
        #  Could we not use __call__ but add two normal methods in class?
        Tobs = tf[-1]/YRSID_SI
        dt = tf[1]-tf[0]
        # T = Tobs - int(Tobs * YRSID_SI/dt - tf.shape[0]) * dt/YRSID_SI
        # print("the total observ time is ", Tobs)
        hpS, hcS = self(Tobs, dt, eps, modes)

        cs2p = cos(2*self.psi)
        sn2p = sin(2*self.psi)
        csi = cos(self.iota)

        hp_SSB = -(1+csi*csi)*hpS*cs2p+2*csi*hcS*sn2p
        hc_SSB = -(1+csi*csi)*hpS*sn2p-2*csi*hcS*cs2p

        return hp_SSB, hc_SSB


waveforms = {'burst': BurstWaveform,
             'bhb_PhenomD': BHBWaveform,
             'bhb_EccFD': BHBWaveformEcc,
             'gcb': GCBWaveform,
             'gcb_fast': FastGB,
             'emri': EMRIWaveform,
             }  # all available waveforms
