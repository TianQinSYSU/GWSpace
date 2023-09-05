#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyNoise.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2022-11-04 11:03:09
# ==================================

import numpy as np
from Constants import C_SI, PI,
import matplotlib.pyplot as plt


class TianQinNoise(object):

    def __init__(self, Na=1e-30, Np=1e-24, armL=1.7e8):
        self.Na = Na
        self.Np = Np
        self.armL = armL
        self.LT = self.armL/C_SI

    def noises(self, freq, unit="relativeFrequency"):
        """
        Acceleration noise & Optical Metrology System
        Sp = self.Np / (2 * self.armL)**2 * np.ones_like(freq)
        Sa = self.Na *(1+1e-4/freq) / (2 * PI * freq)**4 / (2 * self.armL)**2
        """
        omega = 2*PI*freq
        # In acceleration
        Sa_a = self.Na  # * (1. + 0.1e-3/freq ) # without the tail of freq

        # In displacement
        Sa_d = Sa_a/omega**4
        Soms_d = self.Np*np.ones_like(freq)

        # In Relative frequency unit
        Sa_nu = Sa_d*(omega/C_SI)**2
        Soms_nu = Soms_d*(omega/C_SI)**2

        if unit == "displacement":
            return Sa_d, Soms_d
        elif unit == "relativeFrequency":
            return Sa_nu, Soms_nu  # Spm, Sop
        else:
            print(f"No such unit of {self.unit}")

    def sensitivity(self, freq):
        Sa, Sp = self.noises(freq, unit="displacement")
        f_star = C_SI/(2*PI*self.armL)
        sens = (2*(1+np.cos(freq/f_star))*Sa*(1+1e-4/freq)+Sp)
        tmp = (1+(freq/0.41/C_SI*2*self.armL)**2)
        return 10./3/self.armL**2*sens*tmp


class LISANoise(object):

    def __init__(self, Na=3e-15**2, Np=15.0e-12**2,
                 armL=2.5e9, model="SciRDv1"):
        self.Na = Na
        self.Np = Np
        self.armL = armL
        self.LT = self.armL/C_SI

    def noises(self, freq, unit="relativeFrequency"):
        """
        Acceleration noise & Optical Metrology System
        """
        # In acceleration
        Sa_a = self.Na*(1.+(0.4e-3/freq)**2)*(1+(freq/8e-3)**4)

        # In displacement
        Sa_d = Sa_a/(2*PI*freq)**4
        Soms_d = self.Np*(1+(2e-3/freq)**4)

        # In Relative frequency unit
        Sa_nu = Sa_d*(2*PI*freq/C_SI)**2
        Soms_nu = Soms_d*(2*PI*freq/C_SI)**2

        if unit == "displacement":
            return Sa_d, Soms_d
        elif unit == "relativeFrequency":
            return Sa_nu, Soms_nu  # Spm, Sop
        else:
            print(f"No such unit of {self.unit}")

    def sensitivity(self, freq):
        Sa, Sp = self.noises(freq, unit="displacement")
        All_m = np.sqrt(4*Sa+Sp)

        ## Average the antenna response
        AvResp = np.sqrt(5)

        ## projection effect
        Proj = 2./np.sqrt(3)

        ## Approximative transfer function
        f0 = 1./(2.*self.LT)
        a = 0.41
        T = np.sqrt(1+(freq/(a*f0))**2)
        sens = (AvResp*Proj*T*All_m/self.armL)**2
        return sens


def noise_XYZ(freq, Sa, Sp, armL):
    u = freq*(2*PI*armL/C_SI)
    cu = np.cos(u)
    su = np.sin(u)
    su2 = su*su
    sx = 16*su2*(2*(1+cu*cu)*Sa+Sp)
    sxy = -8*su2*cu*(Sp+4*Sa)
    return sx, sxy


def noise_AET(freq, Sa, Sp, armL):
    u = freq*(2*PI*armL/C_SI)
    cu = np.cos(u)
    su = np.sin(u)
    su2 = su*su
    ae = 8*su2*((2+cu)*Sp+4*(1+cu+cu**2)*Sa)
    tt = 16*su2*(1-cu)*(Sp+2*(1-cu)*Sa)
    return ae, ae, tt



