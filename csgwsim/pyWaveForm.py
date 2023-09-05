#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyWaveForm.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 12:32:36
# ==================================

import os
import sys
# import numpy as np

try:
    import bbh
except:
    bbh_path = './Waveforms/bbh'
    abs_bbh_path = os.path.abspath(bbh_path)

    if abs_bbh_path not in sys.path:
        sys.path.append(abs_bbh_path)

try:
    import few
except:
    few_path = './Waveforms/FastEMRIWaveforms'
    abs_few_path = os.path.abspath(few_path)

    if abs_few_path not in sys.path:
        sys.path.append(abs_few_path)

from utils import to_m1m2
from GCB import GCBWaveform
from BHB import *
from EMRI import EMRIWaveform
from Burst import BurstWaveform


class WaveForm:
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



