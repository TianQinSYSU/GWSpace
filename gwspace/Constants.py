#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Constants.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-08-14 23:09:12
# ==================================

# /****** Boolean conventions for loading files ******/

SUCCESS = 0
FAILURE = 1
NONE = -1

# /* Mathematica 11.3.0.0 */

PI = 3.141592653589793238462643383279502884
PI_2 = 1.570796326794896619231321691639751442
PI_3 = 1.047197551196597746154214461093167628
PI_4 = 0.785398163397448309615660845819875721
SQRTPI = 1.772453850905516027298167483341145183
SQRTTWOPI = 2.506628274631000502415765284811045253
INVSQRTPI = 0.564189583547756286948079451560772585
INVSQRTTWOPI = 0.398942280401432677939946059934381868
GAMMA = 0.577215664901532860606512090082402431
SQRT2 = 1.414213562373095048801688724209698079
SQRT3 = 1.732050807568877293527446341505872367
SQRT6 = 2.449489742783178098197284074705891392
INVSQRT2 = 0.707106781186547524400844362104849039
INVSQRT3 = 0.577350269189625764509148780501957455
INVSQRT6 = 0.408248290463863016366214012450981898

# GAMMA = 0.577215664901532860606512090082402431

# Physical constants in SI units

C_SI = 299792458.
G_SI = 6.67259e-11
MSUN_SI = 1.988546954961461467461011951140572744e30
MTSUN_SI = 4.9254923218988636432342917247829673e-6
MSUN_unit = G_SI*MSUN_SI/C_SI**3  # 4.926e-6  # the factor of mass(in solar mass) from the nature unit to SI
PC_SI = 3.085677581491367278913937957796471611e16
KPC_SI = 3.085677581491367278913937957796471611e19
MPC_SI = 3.085677581491367278913937957796471611e22
MPC_T = MPC_SI/C_SI
AU_SI = 1.4959787066e11
AU_T = AU_SI/C_SI
YRSID_SI = 3.15581497635e7  # Sidereal year as found on http://hpiers.obspm.fr/eop-pc/models/constants.html
DAY = 86400
MONTH = 2592000
YEAR = 31536000

DAYSID_SI = 86164.09053  # Mean sidereal day

# Orbital pulsation: 2pi/year - use sidereal year as found on http://hpiers.obspm.fr/eop-pc/models/constants.html
# EarthOrbitOmega_SI = 1.99098659277e-7
EarthOrbitOmega_SI = 2*PI/YRSID_SI
EarthOrbitFreq_SI = 1/YRSID_SI

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
EarthMass = 5.9722e24  # [kg]
EarthEcc = 0.01671022
# angle measured from the vernal equinox to the perihelion i.e. **Argument of perihelion**
# 102.94719/180 * PI
Perihelion_Ang = 1.7967674211761813
# # vernal equinox is September equinox (09-22/23)
# # perihelion is at about 01-03/04
# # (31 + 30 + 31 + 11)/365.2425 * 2 * PI

# ecliptic longitude and latitude of J0806.3+1527
J0806_phi = 2.103121748653167  # 120.5
J0806_theta = 1.65282680163863  # -4.7 = 90 + 4.7

# Constants used to relate time scales
EPOCH_J2000_0_TAI_UTC = 32  # Leap seconds (TAI-UTC) on the J2000.0 epoch (2000 JAN 1 12h UTC)
EPOCH_J2000_0_GPS = 630763213  # GPS seconds of the J2000.0 epoch (2000 JAN 1 12h UTC)

# cosmological constants
H0 = 67.4  # in km/s/Mpc: Planck 2018 1807.06209
H0_SI = H0*1.e3/MPC_SI  # in /s
Omega_m_Planck2018 = 0.315  # Planck 2018 1807.06209
