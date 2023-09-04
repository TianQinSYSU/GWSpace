""" A Python JIT implementation of TianQin orbit by Hong-Yu Chen 2023"""

import numpy as np
from numba import njit

pi = np.pi
year = 31558149.763545600 # siderial year [sec] (http://hpiers.obspm.fr/eop-pc/models/constants.html)
f_m = 1 / year #  Earth  orbiting frequency ( 1 / 365  days )
fsc = 1/314496 # TianQin orbiting frequency ( 1 / 3.64 days )

Rearth = 149597870700 # [m]
eccEarth = 0.0167
RTianqin = 1e8 # [m] semi-major axis of the spacecraft orbit around the Earth
eccTianqin = 0 # should be lower than 0.05. Temporarily set it to zero

# (θs = −4.7◦, φs = 120.5◦) in the ecliptic coordinates.
thetas = -4.7 /180*pi # degree -> rad
phis = 120.5/180*pi   # degree -> rad

# Sin(phis) ,Cos(phis), Sin(thetas), Cos(thetas) of J0806.3+1527  ####   
snps = np.sin(phis)
csps = np.cos(phis)
snts = np.sin(thetas)
csts = np.cos(thetas)

kappa0 = 0 # the mean ecliptic longitude measured from the vernal equinox (or September equinox) at t=0
kappa1 = 0
kappa2 = 2/3*pi
kappa3 = 4/3*pi

@njit()
def alp_t(time_alpha, kai=0):
    alp_t1 = 2*pi * fsc * time_alpha + kappa0 # Orbit frequency of satellite1
    alp_t2 = alp_t1 + 2/3*pi                  # Orbit frequency of satellite2
    alp_t3 = alp_t1 + 4/3*pi                  # Orbit frequency of satellite3

    alp_t_earth = 2*pi * f_m  * time_alpha + kai # Orbit frequency of Earth

    return alp_t1, alp_t2, alp_t3, alp_t_earth

############### The Earth orbit ############################
@njit()
def Earth(csa,csa2,sia,sia2):
    x = Rearth*(csa+0.5*eccEarth*(csa2-3)-3/2*eccEarth**2*csa*sia**2)
    y = Rearth*(sia+0.5*eccEarth*sia2+1/4*eccEarth**2*sia*(3*csa2-1))
    z = np.zeros(len(csa)) 
    return x,y,z

########## The coordinate of TQ (main part without eccentricity) SSB ############
@njit()
def xyz1_TQ_withoutEccentricity(sinalp_t1,cosalp_t1,cosalp_2_t1,sinalp_earth,cosalp_earth):
    x =  RTianqin*(csps*snts*sinalp_t1+cosalp_t1*snps) + Rearth*cosalp_earth   
    y =  RTianqin*(snps*snts*sinalp_t1-cosalp_t1*csps) + Rearth*sinalp_earth
    z = -RTianqin*sinalp_t1*csts
    return x,y,z

@njit()
def xyz2_TQ_withoutEccentricity(sinalp_t2,cosalp_t2,cosalp_2_t2,sinalp_earth,cosalp_earth):
    x =  RTianqin*(csps*snts*sinalp_t2+cosalp_t2*snps) + Rearth*cosalp_earth
    y =  RTianqin*(snps*snts*sinalp_t2-cosalp_t2*csps) + Rearth*sinalp_earth
    z = -RTianqin*sinalp_t2*csts
    return x,y,z

@njit()
def xyz3_TQ_withoutEccentricity(sinalp_t3,cosalp_t3,cosalp_2_t3,sinalp_earth,cosalp_earth):
    x =  RTianqin*(csps*snts*sinalp_t3+cosalp_t3*snps) + Rearth*cosalp_earth
    y =  RTianqin*(snps*snts*sinalp_t3-cosalp_t3*csps) + Rearth*sinalp_earth
    z = -RTianqin*sinalp_t3*csts
    return x,y,z

############################## TianQin Eccentricity ##############################
@njit()
def xyz1_TQ_TianqinEccentricity(sinalp_t1,cosalp_t1,cosalp_2_t1):
    x =  RTianqin*eccTianqin*(0.5*(cosalp_2_t1-3)*snps+cosalp_t1*csps*snts*sinalp_t1) \
        +RTianqin*eccTianqin*eccTianqin/4*sinalp_t1*((3*cosalp_2_t1-1)*csps*snts-6*cosalp_t1*sinalp_t1*snps)

    y = -RTianqin*eccTianqin*(0.5*(cosalp_2_t1-3)*csps-cosalp_t1*snps*snts*sinalp_t1) \
        +RTianqin*eccTianqin*eccTianqin/4*sinalp_t1*((3*cosalp_2_t1-1)*snps*snts+6*cosalp_t1*sinalp_t1*csps)

    z = -RTianqin*eccTianqin*cosalp_t1*sinalp_t1*csts \
        -eccTianqin**2*RTianqin/4*(3*cosalp_2_t1-1)*sinalp_t1*csts
    return x,y,z

@njit()
def xyz2_TQ_TianqinEccentricity(sinalp_t2,cosalp_t2,cosalp_2_t2):
    x =  RTianqin*eccTianqin*(0.5*(cosalp_2_t2-3)*snps+cosalp_t2*csps*snts*sinalp_t2) \
        +RTianqin*eccTianqin*eccTianqin/4*sinalp_t2*((3*cosalp_2_t2-1)*csps*snts-6*cosalp_t2*sinalp_t2*snps)
              
    y = -RTianqin*eccTianqin*(0.5*(cosalp_2_t2-3)*csps-cosalp_t2*snps*snts*sinalp_t2) \
        +RTianqin*eccTianqin*eccTianqin/4*sinalp_t2*((3*cosalp_2_t2-1)*snps*snts+6*cosalp_t2*sinalp_t2*csps)
        
    z = -RTianqin*eccTianqin*cosalp_t2*sinalp_t2*csts \
        -eccTianqin**2*RTianqin*(3*cosalp_2_t2-1)*sinalp_t2*csts
    return x,y,z

@njit()
def xyz3_TQ_TianqinEccentricity(sinalp_t3,cosalp_t3,cosalp_2_t3):
    x =  RTianqin*eccTianqin*(0.5*(cosalp_2_t3-3)*snps+cosalp_t3*csps*snts*sinalp_t3) \
        +RTianqin*eccTianqin*eccTianqin/4*sinalp_t3*((3*cosalp_2_t3-1)*csps*snts-6*cosalp_t3*sinalp_t3*snps)
              
    y = -RTianqin*eccTianqin*(0.5*(cosalp_2_t3-3)*csps-cosalp_t3*snps*snts*sinalp_t3) \
        +RTianqin*eccTianqin*eccTianqin/4*sinalp_t3*((3*cosalp_2_t3-1)*snps*snts+6*cosalp_t3*sinalp_t3*csps)
        
    z = -RTianqin*eccTianqin*cosalp_t3*sinalp_t3*csts \
        -eccTianqin**2*RTianqin*(3*cosalp_2_t3-1)*sinalp_t3 * csts
    return x,y,z
    
############################## Earth Eccentricity ##############################
@njit()
def xyz_TQ_EarthEccentricity(csa,csa2,sia,sia2):
    x = Rearth*eccEarth/2*(csa2-3) - 3/2*Rearth*eccEarth**2*csa*sia**2
    y = Rearth*eccEarth/2*sia2 + Rearth*eccEarth**2/4*(3*csa2-1)*sia 
    z = np.zeros_like(x)
    return x,y,z

@njit()
def addEccentricity(x,y,z,x_Eccentricity,y_Eccentricity,z_Eccentricity):
    x += x_Eccentricity
    y += y_Eccentricity
    z += z_Eccentricity
    return x,y,z

@njit()
def TQ_orbit(time_alpha, kai=0, eccTianqin=0, eccEarth=0.0167):
    alp_t1, alp_t2, alp_t3, alp_t_earth = alp_t(time_alpha, kai)
    
    sinalp_t1 = np.sin(alp_t1)    # Calculate the sin(alpha1(time)) at first
    cosalp_t1 = np.cos(alp_t1)    # Calculate the cos(alpha1(time)) at first 
    cosalp_2_t1 = np.cos(2*alp_t1)# Calculate the cos(2*alpha1(time)) at first

    sinalp_t2 = np.sin(alp_t2)    # Calculate the sin(alpha2(time)) at first
    cosalp_t2 = np.cos(alp_t2)    # Calculate the cos(alpha2(time)) at first
    cosalp_2_t2 = np.cos(2*alp_t2)# Calculate the cos(2*alpha1(time)) at first

    sinalp_t3 = np.sin(alp_t3)    # Calculate the sin(alpha3(time)) at first
    cosalp_t3 = np.cos(alp_t3)    # Calculate the cos(alpha3(time)) at first
    cosalp_2_t3 = np.cos(2*alp_t3)# Calculate the cos(2*alpha3(time)) at first
        
    ################ Alpha_earth() α #############
    # β is the angle measured from the vernal equinox to the perihelion
    csa = np.cos(alp_t_earth)    # β = 0 
    sia = np.sin(alp_t_earth)    # β = 0 
    csa2 = np.cos(2*alp_t_earth) # β = 0 
    sia2 = np.sin(2*alp_t_earth) # β = 0         

    sinalp_earth = np.sin(alp_t_earth-pi/4)       # β =  π /4   
    cosalp_earth = np.cos(alp_t_earth-pi/4)       # β =  π /4     
    # sinalp_2_earth = np.sin(2*(alp_t_earth-pi/4)) # β =  π /4  
    # cosalp_2_earth = np.cos(2*(alp_t_earth-pi/4)) # β =  π /4

    x1, y1, z1 = xyz1_TQ_withoutEccentricity(sinalp_t1,cosalp_t1,cosalp_2_t1,sinalp_earth,cosalp_earth)
    x2, y2, z2 = xyz2_TQ_withoutEccentricity(sinalp_t2,cosalp_t2,cosalp_2_t2,sinalp_earth,cosalp_earth)
    x3, y3, z3 = xyz3_TQ_withoutEccentricity(sinalp_t3,cosalp_t3,cosalp_2_t3,sinalp_earth,cosalp_earth)
    
    if (eccTianqin != 0):
        x1_TE, y1_TE, z1_TE = xyz1_TQ_TianqinEccentricity(sinalp_t1,cosalp_t1,cosalp_2_t1)
        x2_TE, y2_TE, z2_TE = xyz2_TQ_TianqinEccentricity(sinalp_t2,cosalp_t2,cosalp_2_t2)
        x3_TE, y3_TE, z3_TE = xyz3_TQ_TianqinEccentricity(sinalp_t3,cosalp_t3,cosalp_2_t3)

        x1, y1, z1 = addEccentricity(x1,y1,z1,x1_TE,y1_TE,z1_TE)
        x2, y2, z2 = addEccentricity(x2,y2,z2,x2_TE,y2_TE,z2_TE)
        x3, y3, z3 = addEccentricity(x3,y3,z3,x3_TE,y3_TE,z3_TE)
    # else:
    #     print('without Tianqin Eccentricity')

    # The eccentricity of Earth should NOT be zero
    if (eccEarth != 0):
        x_EE, y_EE, z_EE = xyz_TQ_EarthEccentricity(csa,csa2,sia,sia2)

        x1, y1, z1 = addEccentricity(x1,y1,z1,x_EE,y_EE,z_EE)
        x2, y2, z2 = addEccentricity(x2,y2,z2,x_EE,y_EE,z_EE)
        x3, y3, z3 = addEccentricity(x3,y3,z3,x_EE,y_EE,z_EE)
    # else:
    #     print('without Earth Eccentricity')

    return x1,y1,z1, x2,y2,z2, x3,y3,z3
