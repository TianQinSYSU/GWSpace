#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: .py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 10:23:36
#==================================

class constant:
    '''
    Here is some constants
    '''
    
    def __init__(self):
        from scipy import constants
        
        # nature constant
        self.C_SI = constants.c
        self.G_SI = constants.G
        self.PC = constants.parsec
        self.MPC = 1e6 * constants.parsec
        self.MPC_T = self.MPC / constants.c
        
        # solar system
        self.YR = constants.year
        self.DAY = constants.day
        self.MONTH = 30 * constants.day
        self.YRSID_SI = 3.15581497635e7
        self.AU_SI = constants.au
        self.AU_T = constants.au / constants.c
        self.Omega_earth = 2*constants.pi / self.YRSID_SI
        # Sep. equinox (09-22/23)
        # Perihelion (01-03/04)
        self.Beta_perihelion = (31+30+31+11)/self.YRSID_SI * constants.pi *2
        # M_sun = 1.988475e30  # solar mass in kg
        self.MSUN_SI = 1.988546954961461467461011951140572744e30
        self.MSUN_unit = self.G_SI * self.MSUN_SI / self.C_SI
        self.ecc_earth = 0.0167
        
        # mathmatica
        self.PI = constants.pi
        self.twoPI = 2 * constants.pi
        self.trigPI = 3 * constants.pi
        self.PIo2 = constants.pi / 2.
        self.PIo3 = constants.pi / 3.


#const = constant()


## ==================================================
## General physical constants
## --------------------------------------------------
c = 299792458.0             # Speed of light in vacuo, [m s^-1]
G = 6.67259e-11             # Gravitational constant, [N m^2 kg^-2]
pc = 3.0856775807e16        # Parsec, [m]

## ==================================================
## General math constants
## --------------------------------------------------
pi    = 3.1415926535897932384626433832795029
twopi = 6.2831853071795864769252867665590058
pio2  = 1.5707963267948965579989817342720926
pio3  = 1.0471975511965976313177861811709590
pio4  = 0.7853981633974482789994908671360463
sqrt2   = 1.4142135623730950488016887242096981
sqrt2o2 = 0.7071067811865475244008443621048490
sqrt3   = 1.7320508075688771931766041234368458
sqrt3o2 = 0.8660254037844385965883020617184229

## ==================================================
## Unit time constants
## --------------------------------------------------
day = 86400 # s 
month = 2592000 # s : 30 day
year = 31536000 # s: 365 day
daysid_SI =   86164.09053      # Mean sidereal day,
yrsid_SI = 31558149.763545600  # siderial year [sec] (http://hpiers.obspm.fr/eop-pc/models/constants.html)
yrtrop_SI =   31556925.2       # Tropical year (1994), s

## ==================================================
## Unit distance constants
## --------------------------------------------------
kpc = 1e3*pc
Mpc = 1e6*pc
Gpc = 1e9*pc
pct = pc/c  # parsec in seconds
AU  = 1.4959787066e+11 # Astronomical unit, [m]

## ==================================================
## General constant about earth in Solar system
## --------------------------------------------------
eearth = 0.0167                                # the eccentricity of the geocenter orbit around the Sun.
fm = 1.0/yrsid_SI                              # 1/(one sidereal year) = 3.14 × 10−8 Hz is the modulation frequency

GMsun = 1.32712442099e+20   # GMsun in SI (http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf)
MsunKG = GMsun/G            # Solar mass [kg] (Stas ?)
# MsunKG = 1.988475e30         # Solar mass [kg] (Luzum et al., The IAU 2009 system of astronomical constants: 
#                             the report of the IAU working group on numerical standards for Fundamental Astronomy,
#                             Celestial Mechanics and Dynam- ical Astronomy, 2011)
ua = 149597870700.           # Astronomical unit [m](resolution B2 of IAU2012)

## ==================================================
## Convert distance/mass in time
## --------------------------------------------------
kg2s = G/(c*c*c)        # Kilogram to seconds
MTsun = GMsun/c**3      # Solar mass in seconds

if __name__ == '__main__':
    print('This file contains different constants.')

    const = constant()
    print("Here is some constants,", end=" ")
    print("such as the spped of light:", end=" ")
    print(f"C_SI = {const.C_SI}")

