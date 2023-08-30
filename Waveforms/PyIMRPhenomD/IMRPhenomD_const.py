"""constants for IMRPhenomD by Matthew Digman copyright 2021"""
#/*
# * Copyright (C) 2015 Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
# *
# *  This program is free software; you can redistribute it and/or modify
# *  it under the terms of the GNU General Public License as published by
# *  the Free Software Foundation; either version 2 of the License, or
# *  (at your option) any later version.
# *
# *  This program is distributed in the hope that it will be useful,
# *  but WITHOUT ANY WARRANTY; without even the implied warranty of
# *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# *  GNU General Public License for more details.
# *
# *  You should have received a copy of the GNU General Public License
# *  along with with program; see the file COPYING. If not, write to the
# *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# *  MA  02111-1307  USA
# */

GAMMA = 0.577215664901532860606512090082402431
#Dimensionless frequency (Mf) at which the inspiral phase
#switches to the intermediate phase
PHI_fJoin_INS = 0.018
#Dimensionless frequency (Mf) at which the inspiral amplitude
#switches to the intermediate amplitude
AMP_fJoin_INS = 0.014
#Minimal final spin value below which the waveform might behave pathological
#because the ISCO frequency is too low. For more details, see the review wiki
#page https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/PhenD_LargeNegativeSpins
MIN_FINAL_SPIN = -0.717
f_CUT = 0.2 # Dimensionless frequency (Mf) at which define the end of the waveform
MTSUN_SI = 4.925491025543575903411922162094833998e-6 #Geometrized solar mass, s
MSUN_SI = 1.988546954961461467461011951140572744e30 #Solar mass, kg
MRSUN_SI = 1.476625061404649406193430731479084713e3 #Geometrized solar mass, m
findT = True #set to True to get the time (and timep) array, False otherwise

CLIGHT = 2.99792458e8     # Speed of light in m/s
PC_SI = 3.085677581491367278913937957796471611e16 #parsec m
include3PNSS = False #whether to include 3pn spin in inspiral
