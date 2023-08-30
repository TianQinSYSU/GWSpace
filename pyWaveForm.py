#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: pyWaveForm.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 12:32:36
#==================================

import numpy as np
from utils import to_m1m2, sYlm
from GCB import GCBWaveform
from BHB import BHBWaveform


class WaveForm:
    '''
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
        - pars['DL']: luminsity distance
        - etc
    '''
    
    def __init__(self, pars):
        self.pars = pars
        lambd, beta = pars['lambda'], pars['beta']
        
        self.u, self.v, self.k = self.refFrame(lambd, beta)
        
        self.psi = pars['psi']
        self.iota = pars['iota']
        
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
            self.gw = BHBWaveform(m1,m2,chi1,chi2,DL,phic,MfRef_in)

            try:
                self.fRef = pars['fRef']
            except:
                self.fRef = 0.
            try:
                self.varphi = pars['varphi']
            except:
                self.varphi = 0.

    def amp_phase(self, freq, mode=[(2,2)]):
        '''
        Generate the amp and phase in frequency domain
        ----------------------------------------------
        Return:
        - amp:
        - phase:
        - time: tf
        - timep: dt/df
        '''
        h22 = self.gw.h22_FD(freq, self.fRef, self.tc)
        
        amp = {}; phase = {}; tf = {}; tfp = {}
        amp[(2,2)] = h22.amp
        phase[(2,2)] = h22.phase
        tf[(2,2)] = h22.time
        tfp[(2,2)] = h22.timep

        return (amp, phase, tf, tfp)

        
    def __call__(self, time):
        if self.pars['type'] == 'GCB':
            hpS, hcS = self.gw(time)

        cs2p = np.cos(2 * self.psi)
        sn2p = np.sin(2 * self.psi)
        csi = np.cos(self.iota)
        
        hp_SSB = -(1+csi*csi) * hpS *cs2p + 2*csi * hcS * sn2p
        hc_SSB = -(1+csi*csi) * hpS *sn2p - 2*csi * hcS * cs2p
        return (hp_SSB, hc_SSB)
        
    def refFrame(self, lambd, beta):
        csl = np.cos(lambd)
        snl = np.sin(lambd)
        csb = np.cos(beta)
        snb = np.sin(beta)
        
        u = np.array([snl, -csl, 0])
        v = np.array([-snb * csl, -snb * snl, csb])
        k = np.array([-csb * csl, -csb * snl, -snb])
        
        return (u,v,k)


if __name__ == '__main__':
    print("This is waveform generation code")

    Tobs = 10000 #const.YRSID_SI / 4
    delta_f = 1/Tobs    
    delta_T = 1
    f_max = 1/(2*delta_T)
    
    tf = np.arange(0,Tobs, delta_T)

    print("Testing of GCB waveform")
    GCBpars = {"type": "GCB",
           "Mc": 0.5,
           "DL": 0.3,
           "phi0": 0.0,
           "f0": 0.001,
           "psi": 0.2,
           "iota": 0.3,
           "lambda": 0.4,
           "beta": 1.2,
          }

    print("Mc" in GCBpars.keys())
    
    GCBwf = WaveForm(GCBpars)
    hpssb, hcssb = GCBwf(tf)

    import matplotlib.pyplot as plt

    plt.plot(tf, hpssb, 'r-', label=r'$h_+^{\rm SSB}$')
    plt.plot(tf, hcssb, 'b--', label=r'$h_\times^{\rm SSB}$')

    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")

    plt.legend(loc="best")
    

    print("Testing of BHB waveform")
    BHBpars = {"type": "BHB",
           "m1": 3.5e6,
           "m2": 2.1e5,
           "chi1": 0.2,
           "chi2": 0.1,
           "DL": 1e3,
           "phic": 0.0,
           "MfRef_in": 0,
           "psi": 0.2,
           "iota": 0.3,
           "lambda": 0.4,
           "beta": 1.2,
           "tc": 0,
          }

    BHBwf = WaveForm(BHBpars)

    NF = 1024
    freq = 10 ** np.linspace(-4,0,NF)

    amp, phase, time, timep = BHBwf.amp_phase(freq)

    plt.figure()
    plt.loglog(freq, amp[(2,2)])
    plt.ylabel('amplitude')
    plt.xlabel('freq')
    
    plt.figure()
    plt.loglog(freq, np.abs(phase[(2,2)]))
    plt.ylabel('Phase')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(time[(2,2)]))
    plt.ylabel('time')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(timep[(2,2)]))
    plt.ylabel('dt/df')
    plt.xlabel('freq')
    
    plt.show()

