#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: FastEMRI.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-28 15:51:20
# ==================================

import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.waveform import GenerateEMRIWaveform
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector

from .utils import get_uvk
from .Constants import YRSID_SI

use_gpu = False

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}


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
                 Phi_phi0=0, Phi_theta0=0, Phi_r0=0, psi=0,
                 model="FastSchwarzschildEccentricFlux",
                 model_insp="SchwarzEccFlux", **kwargs):  # TODO: make it a subclass of BasicWaveform
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

        self.gen_wave = GenerateEMRIWaveform(model,
                                             inspiral_kwargs=inspiral_kwargs,
                                             amplitude_kwargs=amplitude_kwargs,
                                             Ylm_kwargs=Ylm_kwargs,
                                             sum_kwargs=sum_kwargs,
                                             use_gpu=use_gpu)
        self.theta, self.phi = self.gen_wave._get_viewing_angles(qS, phiS, qK, phiK)  # get view angle
        Lambda = self.phi
        Beta = np.pi/2-self.theta
        self.vec_u, self.vec_v, self.vec_k = get_uvk(Lambda, Beta)
        self.psi = psi

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

        # get ylms
        ylms = self.ylm_gen(amp.unique_l, amp.unique_m, self.theta, self.phi).copy()[amp.inverse_lm]

        mode_selector = ModeSelector(amp.m0mask, use_gpu=False)

        modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]

        (teuk_modes_in, ylms_in, ls, ms, ns) = mode_selector(teuk_modes, ylms, modeinds, eps=eps)
        return teuk_modes_in, ylms_in, ls, ms, ns

    def get_hphc_source(self, Tobs, dt, eps=1e-5, modes=None):  # FIXME // change name
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

    def get_hphc(self, tf, eps=1e-5, modes=None):
        Tobs = tf[-1]/YRSID_SI
        dt = tf[1]-tf[0]
        # T = Tobs - int(Tobs * YRSID_SI/dt - tf.shape[0]) * dt/YRSID_SI
        # print("the total observ time is ", Tobs)
        hpS, hcS = self.get_hphc_source(Tobs, dt, eps, modes)

        cs2p = np.cos(2*self.psi)
        sn2p = np.sin(2*self.psi)
        # csi = cos(self.iota)

        # hp_SSB = -(1+csi*csi)*hpS*cs2p+2*csi*hcS*sn2p
        # hc_SSB = -(1+csi*csi)*hpS*sn2p-2*csi*hcS*cs2p
        hp_SSB = hpS*cs2p-hcS*sn2p
        hc_SSB = hpS*sn2p+hcS*cs2p
        return hp_SSB, hc_SSB
