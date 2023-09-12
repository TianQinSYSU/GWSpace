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

from utils import to_m1m2
from Constants import MSUN_SI, MPC_SI, YRSID_SI, PI, C_SI, G_SI
from Waveforms.PyIMRPhenomD import IMRPhenomD as pyIMRD
from Waveforms.PyIMRPhenomD import IMRPhenomD_const as PyIMRC
from ..Waveforms.FastEMRIWaveforms.FastEMRI import *

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


class WaveForm(object):
    """
    Class for waveform
    -------------------------------
    Parameters:
    - pars: dict of parameters for different sources
    such as:
        - pars['type']: GCB; BHB; EMRI; SGWB for different sources
        - pars['lambda']: longitude of the source in ecliptic coordinates
        - pars['beta']: latitude of the source in ecliptic coordinates
        - pars['psi']: polarization angle
        - pars['iota']: inclination angle
        - pars['Mc']: chirp mass
        - pars['DL']: luminosity distance
        - etc
    """

    def __init__(self, pars):
        self.pars = pars

        if pars['type'] == 'GCB':
            Mc = pars["Mc"]
            DL = pars["DL"]
            phi0 = pars["phi0"]
            f0 = pars["f0"]
            if "fdot" in pars.keys():
                fdot = pars['fdot']
            else:
                fdot = None
            if "fddot" in pars.keys():
                fddot = pars['fddot']
            else:
                fddot = None
            self.gw = GCBWaveform(Mc, DL, phi0, f0, fdot, fddot)
        elif pars['type'] == 'BHB':
            if 'Mc' in pars.keys():
                Mc = pars['Mc']
                eta = pars['eta']
                m1, m2 = to_m1m2(Mc, eta)
            elif 'm1' in pars.keys():
                m1 = pars["m1"]
                m2 = pars["m2"]

            chi1 = pars['chi1']
            chi2 = pars['chi2']
            DL = pars['DL']
            phic = pars['phic']
            MfRef_in = pars['MfRef_in']
            self.tc = pars['tc']
            self.gw = BHBWaveform(m1, m2, chi1, chi2, DL, phic, MfRef_in)

            try:
                self.fRef = pars['fRef']
            except:
                self.fRef = 0.
            try:
                self.varphi = pars['varphi']
            except:
                self.varphi = 0.
        elif pars['type'] == 'EMRI':
            self.M = pars['M']
            self.mu = pars['mu']
            self.a = pars['a']
            self.p0 = pars['p0']
            self.e0 = pars['e0']
            self.x0 = pars['x0']
            self.dist = pars['dist']

            self.qS = pars['qS']
            self.phiS = pars['phiS']
            self.qK = pars['qK']
            self.phiK = pars['phiK']
            self.Phi_phi0 = pars['Phi_phi0']
            self.Phi_theta0 = pars['Phi_theta0']
            self.Phi_r0 = pars['Phi_r0']

            self.gw = EMRIWaveform(
                self.M, self.mu, self.a, self.p0, self.e0,
                self.x0, self.dist, self.qS, self.phiS,
                self.qK, self.phiK, self.Phi_phi0,
                self.Phi_theta0, self.Phi_r0)
            # self.gw.theta
        elif pars['type'] == 'Burst':
            self.amp = pars['amp']
            self.tau = pars['tau']
            self.fc = pars['fc']
            self.tc = pars['tc']

            self.gw = BurstWaveform(
                self.amp, self.tau, self.fc, self.tc)
        try:
            lambd, beta = pars['lambda'], pars['beta']
        except:
            lambd = np.pi/2-self.gw.theta
            beta = self.gw.phi

        self.u, self.v, self.k = self.refFrame(lambd, beta)

        self.psi = pars['psi']
        self.iota = pars['iota']

    def amp_phase(self, freq, mode=[(2, 2)]):
        """
        Generate the amp and phase in frequency domain
        ----------------------------------------------
        Parameters:
        -----------
        - freq: frequency list
        - mode: mode of GW

        Return:
        -------
        - amp:
        - phase:
        - tf: time of freq
        - tfp: dt/df
        """
        h22 = self.gw.h22_FD(freq, self.fRef, self.tc)

        amp = {}
        phase = {}
        tf = {}
        tfp = {}
        amp[(2, 2)] = h22.amp
        phase[(2, 2)] = h22.phase
        tf[(2, 2)] = h22.time
        tfp[(2, 2)] = h22.timep

        return amp, phase, tf, tfp

    def __call__(self, tf, eps=1e-5, modes=None):
        if self.pars['type'] == 'GCB':
            hpS, hcS = self.gw(tf)
        elif self.pars['type'] == 'EMRI':
            Tobs = tf[-1]/YRSID_SI
            dt = tf[1]-tf[0]
            # T = Tobs - int(Tobs * YRSID_SI/dt - tf.shape[0]) * dt/YRSID_SI
            # print("the total observ time is ", Tobs)
            hpS, hcS = self.gw(Tobs, dt, eps, modes)

        cs2p = np.cos(2*self.psi)
        sn2p = np.sin(2*self.psi)
        csi = np.cos(self.iota)

        hp_SSB = -(1+csi*csi)*hpS*cs2p+2*csi*hcS*sn2p
        hc_SSB = -(1+csi*csi)*hpS*sn2p-2*csi*hcS*cs2p

        return hp_SSB, hc_SSB

    @staticmethod
    def refFrame(lambd, beta):
        csl = np.cos(lambd)
        snl = np.sin(lambd)
        csb = np.cos(beta)
        snb = np.sin(beta)

        u = np.array([snl, -csl, 0])
        v = np.array([-snb*csl, -snb*snl, csb])
        k = np.array([-csb*csl, -csb*snl, -snb])

        return u, v, k


class BurstWaveform(object):
    """
    A sin-Gaussian waveforms for pooly modelled burst source
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


class BHBWaveform(object):
    """
    This is Waveform for BHB
    ------------------------
    Parameters:
    - m1, m2: mass of black holes
    - chi1, chi2: spin of the two black holes
    - DL: in MPC
    """

    def __init__(self, m1, m2, chi1=0., chi2=0., DL=1.0, phic=0, MfRef_in=0):

        # set parameters for a black hole binary system
        self.chi1 = chi1
        self.chi2 = chi2
        self.m1_SI = m1*MSUN_SI
        self.m2_SI = m2*MSUN_SI
        self.distance = DL*MPC_SI
        self.phic = phic
        self.MfRef_in = MfRef_in

    def h22_FD(self, freq, fRef=0, t0=0):
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

        # Generate h22 FD amplitude and phse on a given set of frequencies
        self.h22 = pyIMRD.IMRPhenomDGenerateh22FDAmpPhase(
            self.h22, freq,
            self.phic, self.MfRef_in,
            self.m1_SI, self.m2_SI,
            self.chi1, self.chi2,
            self.distance)

        return self.h22


class GCBWaveform(object):
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

    def __init__(self, Mc, DL, phi0, f0, fdot=None, fddot=None):
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
        self.phi0 = phi0

    def __call__(self, t):
        phase = 2*PI*(self.f0+0.5*self.fdot*t +
                      1/6*self.fddot*t*t)*t+self.phi0
        hp = self.amp*np.cos(phase)
        hc = self.amp*np.sin(phase)
        return hp, hc


class FastGB(object):
    """
    Calculate the GCB waveform using fast/slow
    """

    def __init__(self, pars, N):
        self.N = N
        pass


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
                 use_gpu=use_gpu):
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

    def __call__(self, Tobs, dt, eps=1e-5, modes=None):
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
