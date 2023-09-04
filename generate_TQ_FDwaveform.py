""" A Python JIT implementation of TianQin TDI-1.0 responsed IMRPhenomD waveform by Hong-Yu Chen 2023"""
# This file need to import the 'pyIMRPhenomD' in [https://github.com/XGI-MSU/PyIMRPhenomD], 
#                      and the 'TianqinOrbit' which also written by Hong-Yu Chen.
# Put the related .py files in the same folder.

# TO DO: Using 'jax'[https://github.com/google/jax] to implement Python JIT, rather than 'numba'.

import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from IMRPhenomD import IMRPhenomDGenerateFD
from TianqinOrbit import TQ_orbit

### Mathematics constants
pi = np.pi

### Physical constants
c = 299792458.0     # Speed of light in vacuum [m.s^-1] (CODATA 2014)
G = 6.67408e-11     # Newtonian constant of graviation [m^3.kg^-1.s^-2] (CODATA 2014)
h = 6.626070040e-34 # Planck constant [J.s] (CODATA 2014)

### Derived constants
kg2s = G/(c**3) # Kilogram to seconds

### Astronomical units
pc = 3.08567758149136727e+16  # Parsec [m] (XXIX General Assembly of the International Astronomical Union, RESOLUTION B2 on recommended zero points for the absolute and apparent bolometric magnitude scales, 2015)
GMsun = 1.32712442099e+20     # GMsun in SI (http://asa.usno.navy.mil/static/files/2016/Astronomical_Constants_2016.pdf)
MsunKG = 1.98854695496e+30    # Solar mass [kg]
# MsunKG = GMsun/G
# MsunKG = 1.988475e30        # Solar mass [kg] (Luzum et al., The IAU 2009 system of astronomical constants: the report of the IAU working group on numerical standards for Fundamental Astronomy, Celestial Mechanics and Dynam- ical Astronomy, 2011)
ua = 149597870700.            # Astronomical unit [m](resolution B2 of IAU2012)
YRSID_SI = 31558149.763545600 # siderial year [sec] (http://hpiers.obspm.fr/eop-pc/models/constants.html)
MTsun = 4.92549102554e-06     # Solar mass in seconds
# MTsun = GMsun/(c**3)

### Cut-off frequency of the signal
f_CUT = 0.2 # 0.2[sec] for 2-2 mode template; 0.6[sec] for high order mode template

### TianQin arm length
armL = np.sqrt(3)*1e8
Lc = armL / c

@njit()
def Freq_Max(Mc, eta):
    Mt_T = Mc * eta**(-3/5) * MTsun # total mass
    return f_CUT/Mt_T

@njit()
def Freq_Ins(Tobs,t0,Mc):
    if (Tobs >= t0):
        raise ValueError('This function (Freq_Ins) only can be used in inspiral signal!')
    Mc_T = Mc * MTsun
    return 1/(8*pi) * ((t0-Tobs)/5)**(-3/8) * Mc_T**(-5/8)

@njit()
def freq_bound(Tobs,t0,Mc,eta):
    if (Tobs >= t0):
        f_max = Freq_Max(Mc,eta)
    else:
        f_max = Freq_Ins(Tobs,t0,Mc)
    if (f_max > 1.0):
        f_max = 1.0

    f_min = Freq_Ins(0,t0,Mc)
    if (f_min < 1e-4):
        f_min = 1e-4
    
    return f_max, f_min

def tf(freq, phase, t0):
    # This function can't directly / don't need to use 'numba.njit' to implement Python JIT.
    # The 'scipy' is a HPC package, and is difficult to implement Python JIT by 'numba'. 

    # The 'scipy.interpolate.InterpolatedUnivariateSpline.derivative' can only use in (size > 3).
    # When (3 >= size > 1), I use the 'numpy.gradient' to calculcate the derivation. 
    # Maybe it's not necessary, because I don't expect a high likelihood/SNR ratio in this case.
    if (phase.size > 3):
        tfspline = spline(freq, 1/(2*pi)*(phase - phase[0])).derivative() # get rid of possibly huge constant in the phase
        tf_vec = tfspline(freq) + t0
    elif (phase.size > 1):
        tf_vec = np.gradient(1/(2*pi)*(phase - phase[0])) / np.gradient(freq) + t0
    elif (phase.size == 1):
        tf_vec = t0
    else:
        raise ValueError('No phase input in the function (tf)!')
    return tf_vec

@njit()
def vectorK(lambd,beta):
    return np.array([-np.cos(beta)*np.cos(lambd),-np.cos(beta)*np.sin(lambd),-np.sin(beta)])

@njit()
def Y22(iota,varphi=0):
    Y22_o = np.sqrt(5/4/pi) * np.cos(iota/2)**4 * np.exp(2*1j*varphi)
    Y22_c = np.sqrt(5/4/pi) * np.sin(iota/2)**4 * np.exp(2*1j*varphi)
    return Y22_o, Y22_c

@njit()
def P0(lambd,beta):
    sib = np.sin(beta)
    csb = np.cos(beta)
    sil = np.sin(lambd)
    csl = np.cos(lambd)
    sil2 = np.sin(2*lambd)
    csl2 = np.cos(2*lambd)
        
    P0_plus  = np.array([-sib**2*csl**2 + sil**2, (sib**2+1)*(-sil*csl),   sib*csb*csl,
                         (sib**2+1)*(-sil*csl),   -sib**2*sil**2 + csl**2, sib*csb*sil,
                         sib*csb*csl,             sib*csb*sil,             -csb**2    ]).reshape(3,3)
        
    P0_cross = np.array([-sib*sil2, sib*csl2, csb*sil, 
                         sib*csl2,  sib*sil2, -csb*csl,
                         csb*sil,   -csb*csl, 0       ]).reshape(3,3) 
        
    return P0_plus, P0_cross

@njit()
def P22(lambd,beta,psi,iota,varphi=0):
    Y22_o, Y22_c = Y22(iota,varphi)
    P0_plus,P0_cross = P0(lambd,beta)
        
    return (1/2 * Y22_o * np.exp(-2j*psi) * (P0_plus + 1j*P0_cross) + 
            1/2 * Y22_c * np.exp( 2j*psi) * (P0_plus - 1j*P0_cross) )

@njit()
def matrix_res_pro(n,p,m):
    # array * matrix * array
    nPn = (n[0]*p[0,0]*m[0] + n[0]*p[0,1]*m[1] + n[0]*p[0,2]*m[2] \
          +n[1]*p[1,0]*m[0] + n[1]*p[1,1]*m[1] + n[1]*p[1,2]*m[2] \
          +n[2]*p[2,0]*m[0] + n[2]*p[2,1]*m[1] + n[2]*p[2,2]*m[2] )
    return nPn 

@njit()
def array_matrix(n,l):
    # array * matrix
    am = n[0]*l[0,0] + n[1]*l[1,0] + n[2]*l[2,0]
    return am  

def TQ_orbit_func(tf_vec):
    x1,y1,z1, x2,y2,z2, x3,y3,z3 = TQ_orbit(tf_vec)
    x0,y0,z0 = (x1+x2+x3)/3, (y1+y2+y3)/3, (z1+z2+z3)/3
    p0_earth = np.array([x0, y0, z0])

    U12 = np.array([x2-x1, y2-y1, z2-z1]) / armL # unit vector nl TQ1->TQ2
    U23 = np.array([x3-x2, y3-y2, z3-z2]) / armL #                TQ2->TQ3
    U31 = np.array([x1-x3, y1-y3, z1-z3]) / armL #                TQ3->TQ1

    p12 = np.array([x1+x2, y1+y2, z1+z2]) # Vector of TQ1+TQ2  
    p23 = np.array([x2+x3, y2+y3, z2+z3]) #           TQ2+TQ3
    p31 = np.array([x3+x1, y3+y1, z3+z1]) #           TQ1+TQ3

    U_vec = np.array([U12, U23, U31])
    p_vec = np.array([p12, p23, p31])

    return p0_earth, U_vec, p_vec

@njit()
def TQ_FDresponse(freq, U_vec, p_vec, k_vec, P22_mat):
    U12, U23, U31 = U_vec
    p12, p23, p31 = p_vec
    
    com = 1j*pi * freq * Lc / 2
    z = np.exp(2j*pi * freq * armL / c)

    # 22 mode special which induced by P22()
    n12Pn12 = matrix_res_pro(U12,P22_mat,U12) 
    n31Pn31 = matrix_res_pro(U31,P22_mat,U31)  
    n23Pn23 = matrix_res_pro(U23,P22_mat,U23) 

    vk12 = np.dot(k_vec, U12) 
    vk23 = np.dot(k_vec, U23) 
    vk31 = np.dot(k_vec, U31)

    sin21 = np.sinc(freq * Lc * (1 + vk12))
    sin12 = np.sinc(freq * Lc * (1 - vk12))
    sin23 = np.sinc(freq * Lc * (1 - vk23))
    sin32 = np.sinc(freq * Lc * (1 + vk23))
    sin31 = np.sinc(freq * Lc * (1 - vk31))
    sin13 = np.sinc(freq * Lc * (1 + vk31))

    Exp12 = np.exp(1j*pi * freq * (Lc + array_matrix(k_vec, p12)/c))
    Exp23 = np.exp(1j*pi * freq * (Lc + array_matrix(k_vec, p23)/c))
    Exp31 = np.exp(1j*pi * freq * (Lc + array_matrix(k_vec, p31)/c))

    y12 = com * sin12 * Exp12 * n12Pn12 
    y21 = com * sin21 * Exp12 * n12Pn12
    y13 = com * sin13 * Exp31 * n31Pn31
    y31 = com * sin31 * Exp31 * n31Pn31
    y23 = com * sin23 * Exp23 * n23Pn23
    y32 = com * sin32 * Exp23 * n23Pn23

    res_a = ((1.+z)*(y31 + y13) - y23 - z*y32 - y21 - z*y12) 
    res_e = 1/np.sqrt(3) * ((1.-z)*(y13 - y31) + (2.+z)*(y12 - y32) + (1.+2*z)*(y21 - y23))
    res_t = np.sqrt(2/3) * (y21 - y12 + y32 - y23 + y13 - y31)

    return res_a, res_e, res_t

@njit()
def AET_response(wf, t0, res_vec, k_vec, p0_earth):
    freq, amp, phase = wf
    res_a, res_e, res_t = res_vec

    x = pi * freq * Lc
    vkp0 = np.dot(k_vec,p0_earth)
    phaseRdelay = 2*pi * freq * vkp0 / c
    comf_aet = 2*np.exp(-1j*phaseRdelay) # phaseRdelay Doppler

    # Came from (29a),(29b) in arXiv:2003.00357v1 from a,e,t to A,E,T
    factorAE = 1j*np.sqrt(2)*np.sin(2.*x)*np.exp(2j*x)
    factorT  = 2.*np.sqrt(2)*np.sin(2.*x)*np.sin(x)*np.exp(3j*x)

    # Still minusing phaseRdelay
    # if not rescaled: #  Some common factor of noise & signal . Default: False!
    resA = res_a * comf_aet * factorAE
    resE = res_e * comf_aet * factorAE
    resT = res_t * comf_aet * factorT 

    ampphasefactor = amp * np.exp(1j*(phase + phaseRdelay)) * np.exp(2j*pi*freq*t0)
    Achannel = ampphasefactor * resA
    Echannel = ampphasefactor * resE
    Tchannel = ampphasefactor * resT

    return Achannel, Echannel, Tchannel

@njit()
def generate_TQ_AmpPhase(para_list, Tobs):
    Mc = para_list[0]
    eta = para_list[1]

    m1 = Mc * eta**(-3/5) * (1+np.sqrt(1-4*eta))/2
    m2 = Mc * eta**(-3/5) * (1-np.sqrt(1-4*eta))/2
    m1_SI = m1 * MsunKG
    m2_SI = m2 * MsunKG

    chi1 = para_list[2]
    chi2 = para_list[3]

    DL = para_list[4] # Gpc
    distance = DL * 1e9 * pc

    t0 = para_list[5]
    phi0 = para_list[6]

    deltaF = 1 / Tobs
    f_max, f_min = freq_bound(Tobs,t0,Mc,eta)

    if (f_max < f_min):   # If (f_max < 1e-4 Hz) or (f_min > 1 Hz), this will happen.
        return [], [], [] # TianQin will not detect the signal in this case.

    fRef_in = 0
    wf = IMRPhenomDGenerateFD(phi0,fRef_in,deltaF,m1_SI,m2_SI,chi1,chi2,f_min,f_max,distance)

    return wf

def generate_TQ_response(para_list, Tobs, wf):
    freq, amp, phase = wf
    
    t0 = para_list[5]
    lambd = para_list[7]
    beta = para_list[8]
    psi = para_list[9]
    iota = para_list[10]

    tf_vec = tf(freq,phase,t0)
    k_vec = vectorK(lambd,beta)
    P22_mat = P22(lambd,beta,psi,iota)

    p0_earth, U_vec, p_vec = TQ_orbit_func(tf_vec)
    res_a, res_e, res_t = TQ_FDresponse(freq, U_vec, p_vec, k_vec, P22_mat)
    res_vec = np.array([res_a, res_e, res_t])

    Achannel, Echannel, Tchannel = AET_response(wf, t0, res_vec, k_vec, p0_earth)

    return Achannel, Echannel, Tchannel

def generate_TQ_TDI_FDwaveform(para, Tobs, frequency=None):
    """
    This function is used to generate the TDI-1.0 responsed BBH Frequency-Domain waveform in TianQin.
    @ para      : (numpy.array[float]) a 11-dimensional parameter array for BBH.
                [0] r$M_c$     : chirp mass [solar mass]; 
                [1] r$\eta$    : symmetrical mass ratio (0,0.25); 
                [2] r$\chi_1$  : dimensionless spin of the primary BH (-1,1); 
                [3] r$\chi_2$  : dimensionless spin of the second BH (-1,1); 
                [4] r$D_L$     : luminosity distance [Gpc]
                [5] r$t_0$     : merge time [second]; 
                [6] r$\phi_0$  : merge phase (0,2*pi) [rad]; 
                [7] r$\lambda$ : ecliptic longitude (0,2*pi) [rad];
                [8] r$\\beta$  : ecliptic latitude (-pi/2,pi/2) [rad];
                [9] r$\psi$    : polarization (0,pi) [rad];
                [10] r$\iota$  : inclination (0,pi) [rad].
    @ Tobs      : (float) the observation time [second].
    @ frequency : (numpy.array[float]) the frequency array of the signal/data. 
                If None, then use the Nyquist frequency array with f_max = 1 Hz, f_min = 1e-4 Hz.

    Output :
    All the output have the same size.
    @ frequency : (numpy.array[float]) the frequency array of the signal/data.
                If input the parameter 'frequency', the output will be the same.
    @ signalA   : (numpy.array[numpy.complex64]) A channel TDI responsed BBH signal in TianQin.
    @ signalE   : (numpy.array[numpy.complex64]) E channel TDI responsed BBH signal in TianQin.
    @ signalT   : (numpy.array[numpy.complex64]) T channel TDI responsed BBH signal in TianQin.
    @ idx       : (numpy.array[bool]) the index pointed to meaningful frequency points.
    """

    if frequency is None:
        df_Nyquist = 1 / Tobs
        frequency = np.arange(1e-4,1,df_Nyquist)

    Nf = len(frequency)
    signalA = np.zeros(Nf, dtype = np.complex64)
    signalE = np.zeros(Nf, dtype = np.complex64)
    signalT = np.zeros(Nf, dtype = np.complex64)

    freq, amp, phase = generate_TQ_AmpPhase(para, Tobs)

    if (len(freq) > 1):
        amp_spline   = interp1d(freq,amp,bounds_error=False,fill_value=0)
        phase_spline = interp1d(freq,phase,bounds_error=False,fill_value=0)
        
        amp_all = amp_spline(frequency)
        idx = (amp_all != 0)
        freq_idx = frequency[idx]
        amp_idx = amp_spline(freq_idx)
        phase_idx = phase_spline(freq_idx)
    else: 
        idx = np.full(Nf, False, dtype=bool)
        return frequency, signalA, signalE, signalT, idx

    wf = np.array([freq_idx, amp_idx, phase_idx])
    Achannel, Echannel, Tchannel = generate_TQ_response(para, Tobs, wf)

    signalA[idx] = Achannel.copy()
    signalE[idx] = Echannel.copy()
    signalT[idx] = Tchannel.copy()

    return frequency, signalA, signalE, signalT, idx
