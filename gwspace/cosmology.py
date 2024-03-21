#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: cosmology.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2024-03-21 15:42:52
#==================================

import numpy as np
from scipy.integrate import quad
from scipy import optimize
from gwspace.constants import C_SI, H0, Omega_m_Planck2018, H0_SI, YEAR
from gwspace.utils import GeneralPars

class cosmology:
    '''
    General cosmology model
    -----------------------
    the distance etc can be found at astro-ph/9905116
    '''
    H0_planck = H0
    Omm_planck = Omega_m_Planck2018

    def __init__(self, H0=H0_planck, Omm=Omm_planck, Omk=0, w0=-1, wa=0):
        self.H0 = H0
        self.Omm = Omm
        self.Omk = Omk
        self.Omde = 1-Omm-Omk
        self.w0 = w0
        self.wa = wa

        self.tH = GeneralPars(
                1./H0_SI/YEAR/1e9, 
                "Gyr", 
                "The Hubble time")
        self.DH = GeneralPars(
                C_SI/1000/H0,
                "Mpc",
                "The Hubble distance")

    def w_de(self, z):
        return self.w0 + self.wa *z/(1+z)

    def Ez2(self, z):
        Ommz = self.Omm*(1+z)**3
        Omkz = self.Omk*(1+z)**2
        Omdez = self.Omde*(1+z)**(-3*(1+self.w0+self.wa)) * np.exp(-3*self.wa*z/(1+z))
        return Ommz + Omkz + Omdez
    
    def Hofz(self, z):
        '''
        Hubble parameters in unit of km/s/Mpc
        '''
        Ez2 = self.Ez2(z)
        return self.H0 * np.sqrt(Ez2)

    def __oEz(self, z):
        return 1/np.sqrt(self.Ez2(z))

    def dc(self, z):
        '''
        comoving distance without unit
        '''
        d, err = quad(self.__oEz, 0, z)
        return d

    def DC(self, z):
        '''
        Comoving distance in unit of "Mpc"
        '''
        dc = self.dc(z)
        if self.Omk > 1e-8:
            sk = np.sqrt(self.Omk)
            d = 1/sk * np.sinh(sk * dc)
        elif self.Omk < -1e-8:
            sk = np.sqrt(- self.Omk)
            d = 1/sk * np.sin(sk * dc)
        else:
            d = dc

        return GeneralPars(self.DH.value*d, 
                "Mpc",
                "Comoving distance at z = %s"%z)
        
    def DL(self, z):
        '''
        Luminosity distance in unit of Mpc
        '''
        d = self.DC(z).value * (1+z)
        return GeneralPars(d, "Mpc",
                "Luminosity distance at z = %s"%z)

    def DA(self, z):
        return GeneralPars(self.DC(z)/(1+z),
                "Mpc",
                "Angular diameter distance at z = %s")

    def cal_z(self, dl):
        '''
        calculate the redshift at the Luminosity distance of dl
        '''
        z = optimize.root(lambda x: self.DL(x).value - dl, 0).x[0]
        return z

    def lookback_time(self, z, unit="Gyr"):
        '''
        $$
        H(z) = \frac{da}{adt} = (1+z)\frac{d1/(1+z)}{dt} = - \frac{1}{1+z} \frac{dz}{dt}
        $$
        
        $$
        - \int_0^z \frac{1}{(1+z) H(z)} dz = \int_{t}^{t-\Delta t}dt = -\Delta t
        $$
        '''
        _zEz = lambda z: self.__oEz(z)/(1+z)
        delta_t, err = quad(_zEz, 0, z)
        dt = delta_t * self.tH.value
        return GeneralPars(dt, "Gyr",
                "The lookback time from now")

    def differential_comoving_volume(self, z):
        '''
        dVc = DH (1+z)^2 DA^2/E(z) d\Omega dz
        '''
        dVc = self.DH.value * self.DC(z).value**2 * self.__oEz(z)
        return GeneralPars(dVc, "Mpc^3/sr",
                "the comoving volume element dVc/dOmega/dz")

    def comoving_volume(self, z):
        dc = self.dc(z)
        if self.Omk > 1e-8:
            sk = np.sqrt(self.Omk)
            vc = 3/2./self.Omk * (dc*np.sqrt(1+self.Omk * dc**2)
                    -1/sk * np.arcsinh(sk * dc))
        elif self.Omk < -1e-8:
            sk = np.sqrt(-self.Omk)
            vc = 3/2./self.Omk * (dc*np.sqrt(1+self.Omk * dc**2)
                    -1/sk * np.arcsin(sk * dc))
        else:
            vc = dc**3

        Vc = 4*np.pi/3*self.DH.value**3 * vc
        return GeneralPars(Vc, "Mpc^3",
                "Comoving volume")

    def dPdz(self, z, numden, cross):
        '''
        Probability of intersecting objects
        '''
        return numden(z) * cross(z)*self.DH.value * (1+z)**2*self.__oEz(z)


