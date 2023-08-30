"""Python implementation of IMRPhenomD behavior by Matthew Digman copyright 2021"""
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
#
## LAL independent code (C) 2017 Michael Puerrer
#
#/**

import numpy as np
from numba import njit
from Waveforms.PyIMRPhenomD.IMRPhenomD_fring_helper import fring_interp,fdamp_interp,EradRational0815
import Waveforms.PyIMRPhenomD.IMRPhenomD_const as imrc

################ Miscellaneous functions ###############

########################################/
class COMPLEX16FrequencySeries:
    """ SM: structure for downsampled FD 22-mode Amp/Phase waveforms"""
    def __init__(self,epoch,f0,deltaF,length):
        """create structure for downsampled FD 22-mode Amp/Phase waveforms"""
        self.epoch = epoch
        self.f0 = f0
        self.deltaF = deltaF

        self.length = length
        self.data = np.zeros(length,dtype=np.complex128)


class AmpPhaseFDWaveform:
    """structure to store an amp/phase/time FD waveform"""
    def __init__(self,length,freq,amp,phase,time,timep,fRef=0.,t0=0.):
        """create structure for Amp/phase/time FD waveforms"""
        self.length = length
        self.freq = freq
        self.amp = amp
        self.phase = phase
        self.time = time
        self.timep = timep
        self.fRef = fRef
        self.t0 = t0

########################################/
@njit()
def PNPhasingSeriesTaylorF2(eta,chis,chia):
    """ From LALSimInspiralPNCoefficients.c
    The phasing function for TaylorF2 frequency-domain waveform.
    This function is tested in ../test/PNCoefficients.c for consistency
    with the energy and flux in this file.
    m1,   Mass of body 1, in Msol
    m2,   Mass of body 2, in Msol
    chi1, Component of dimensionless spin 1 along Lhat
    chi2, Component of dimensionless spin 2 along Lhat"""

    if eta<0.25:
        delta = np.sqrt(1-4*eta)
    else:
        delta = 0.

    #Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
    #We write dSigmaL for their (\delta m/m) * \Sigma_\ell
    #There's a division by mtotal^2 in both the energy and flux terms
    #We just absorb the division by mtotal^2 into SL and dSigmaL/

    SL = chis*(1-2*eta) + chia*delta
    dSigmaL = -delta*(chis*delta + chia)

    pfaN = 3/(128*eta)

#    /* Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18 */
    v0 = 1.
    v1 = 0.
    v2 = 5/9*(743/84+11*eta)
    v3 = -16*np.pi
    v4 = 5/72*(3058673/7056+5429/7*eta+617*eta**2)
    v5 = 5/9*np.pi*(7729/84-13*eta)
    vlogv5 = 5/3*np.pi*(7729/84-13*eta)
    v6 = (11583231236531/4694215680-640/3*np.pi**2-6848/21*imrc.GAMMA) \
            +(-15737765635/3048192+2255/12*np.pi**2)*eta+76055/1728*eta**2-127825/1296*eta**3 \
            +-6848/21*np.log(4.)
    vlogv6 = -6848/21
    v7 = np.pi*(77096675/254016+378515/1512*eta-74045/756*eta**2)

#     Compute 2.0PN SS, QM, and self-spin */
#     See Eq. (6.24) in arXiv:0810.5336
#     9b,c,d in arXiv:astro-ph/0504538
    pn_sigma = 1/16*(chia**2*(81 - 320*eta) + chis**2*(81 - 4*eta) + 162*chis*chia*delta)

    if imrc.include3PNSS:
        pn_ss3 = 1/2016*(70*chia*chis*delta*(15103 - 13160*eta) \
                        + 5*chis**2*(105721 + 4*eta*(-46483 + 14056*eta)) \
                        - 5*chia**2*(-105721 + 8*eta*(52649 + 24192*eta)))
    else:
        pn_ss3 = 0.

#     Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16 */
    pn_gamma = (554345/1134+110/9*eta)*SL+(13915/84-10/3*eta)*dSigmaL

    v7 += (-8980424995/762048+6586595/756*eta-305/36*eta**2)*SL-(170978035/48384-2876425/672*eta-4735/144*eta**2)*dSigmaL
    v6 += np.pi/3*(3760*SL+1490*dSigmaL)+pn_ss3
    v5 += -pn_gamma
    vlogv5 += -3*pn_gamma
    v4 += -10*pn_sigma
    v3 += 188/3*SL+25*dSigmaL

#     At the very end, multiply everything in the series by pfaN */
    v = (pfaN*v0,pfaN*v1,pfaN*v2,pfaN*v3,pfaN*v4,pfaN*v5,pfaN*v6,pfaN*v7)
    vlogv = (0.,0.,0.,0.,0.,pfaN*vlogv5,pfaN*vlogv6,0.)

    return v,vlogv

#######################################/

#######################################/

@njit()
def chiPN(eta,chis,chia):
    """PN reduced spin parameter
    See Eq 5.9 in http:#arxiv.org/pdf/1107.1267v2.pdf
    Convention m1 >= m2 and chi1 is the spin on m1"""
    if eta<0.25:
        delta = np.sqrt(1-4*eta)
    else:
        delta = 0.
    return chis*(1-76/113*eta)+delta*chia

############ Final spin, final mass, fring, fdamp ############

# Final Spin and Radiated Energy formulas described in 1508.07250
@njit()
def FinalSpin0815(eta,chis,chia):
    """Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
    s defined around Equation 3.6."""
#   Convention m1 >= m2
    if eta<0.25:
        Seta = np.sqrt(1-4*eta)
    else:
        Seta = 0.
#    m1 = (1+Seta)/2
#    m2 = (1-Seta)/2
#   s defined around Equation 3.6 arXiv:1508.07250
#    s = (m1**2*chi1 + m2**2*chi2)
#    s = chis+chia*Seta
    s = chis*(1-2*eta)+chia*Seta
    return 3.4641016151377544*eta - 4.399247300629289*eta**2 +9.397292189321194*eta**3 - 13.180949901606242*eta**4 \
            +(1 - 0.0850917821418767*eta - 5.837029316602263*eta**2)*s \
            +(0.1014665242971878*eta - 2.0967746996832157*eta**2)*s**2\
            +(-1.3546806617824356*eta + 4.108962025369336*eta**2)*s**3\
            +(-0.8676969352555539*eta + 2.064046835273906*eta**2)*s**4


@njit()
def fringdown(eta,chis,chia,finspin):
    denom = (1-EradRational0815(eta, chis, chia))
    fRD = fring_interp(np.array([finspin]))[0]/denom
    fDM = fdamp_interp(np.array([finspin]))[0]/denom
    return fRD,fDM

#****************************** Amplitude functions *******************************/

@njit()
def amp0Func(eta):
    """amplitude scaling factor defined by eq. 17 in 1508.07253"""
    return np.sqrt(2/3)/np.pi**(1/6)*np.sqrt(eta)

##############/ Amplitude: Inspiral functions ############/

@njit()
def rho_funs(eta,chi):
    """Phenom coefficients rho1, ..., rho3 from direct fit
    AmpInsDFFitCoeffChiPNFunc[eta, chiPN]
    See corresponding row in Table 5 arXiv:1508.07253"""
    xi = -1 + chi
    rho1 = 3931.8979897196696 - 17395.758706812805*eta \
        + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta**2)*xi \
        + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta**2)*xi**2 \
        + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta**2)*xi**3
    rho2 = -40105.47653771657 + 112253.0169706701*eta \
        + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta**2)*xi \
        + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta**2)*xi**2 \
        + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta**2)*xi**3
    rho3 = 83208.35471266537 - 191237.7264145924*eta \
        + (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta**2)*xi \
        + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta**2)*xi**2 \
        + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta**2)*xi**3
    return (rho1,rho2,rho3)

@njit()
def AmpInsAnsatz(Mfs,eta,chis,chia,chi,amp_mult=1.):
    """The Newtonian term in LAL is fine and we should use exactly the same (either hardcoded or call).
    We just use the Mathematica expression for convenience.
    Inspiral amplitude plus rho phenom coefficents. rho coefficients computed in rho_funs function.
    Amplitude is a re-expansion. See 1508.07253 and Equation 29, 30 and Appendix B arXiv:1508.07253 for details"""
    rhos = rho_funs(eta,chi)
    amp_prefactors = AmpInsPrefactors(eta,chis,chia,rhos)

    fv = Mfs**(1/3)
    Amps = amp_mult*1/np.sqrt(fv)**7*( \
              1 \
            + fv**2*amp_prefactors[0] \
            + fv**3*amp_prefactors[1] \
            + fv**4*amp_prefactors[2] \
            + fv**5*amp_prefactors[3] \
            + fv**6*amp_prefactors[4] \
            + fv**7*amp_prefactors[5] \
            + fv**8*amp_prefactors[6] \
            + fv**9*amp_prefactors[7] \
            )
    return Amps

@njit()
def AmpInsPrefactors(eta,chis,chia,rhos):
    chi1 = chis+chia
    chi2 = chis-chia
    rho1 = rhos[0]
    rho2 = rhos[1]
    rho3 = rhos[2]
    if eta<0.25:
        Seta = np.sqrt(1-4*eta)
    else:
        Seta = 0.

    two_thirds = ((-969 + 1804*eta)*np.pi**(2/3))/672
    one = ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*np.pi)/48
    four_thirds = ((-27312085 - 10287648*chi2**2 - 10287648*chi1**2*(1 + Seta) + 10287648*chi2**2*Seta \
                    + 24*(-1975055 + 857304*chi1**2 - 994896*chi1*chi2 + 857304*chi2**2)*eta+ 35371056*eta**2) \
                    * np.pi**(4/3)) / 8128512
    five_thirds = (np.pi**(5/3) * (chi2*(-285197*(-1 + Seta) + 4*(-91902 + 1579*Seta)*eta - 35632*eta**2) \
                    + chi1*(285197*(1 + Seta) - 4*(91902 + 1579*Seta)*eta - 35632*eta**2) \
                    + 42840*(-1+4*eta)*np.pi)) / 32256
    two = - (np.pi**2*(-336*(-3248849057+2943675504*chi1**2 - 3339284256*chi1*chi2 + 2943675504*chi2**2)*eta**2 \
                    - 324322727232*eta**3- 7*(-177520268561 + 107414046432*chi2**2 + 107414046432*chi1**2*(1 + Seta) \
                    - 107414046432*chi2**2*Seta + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi) \
                    + 12*eta*(-545384828789 - 176491177632*chi1*chi2 + 202603761360*chi2**2 \
                    + 77616*chi1**2*(2610335 + 995766*Seta) - 77287373856*chi2**2*Seta \
                    + 5841690624*(chi1 + chi2)*np.pi + 21384760320*np.pi**2)))/60085960704
    seven_thirds = rho1
    eight_thirds = rho2
    three = rho3
    return (two_thirds,one,four_thirds,five_thirds,two,seven_thirds,eight_thirds,three)


@njit()
def DAmpInsAnsatz(Mf,eta,chis,chia,chi,amp_mult=1.):
    """Take the AmpInsAnsatz expression pull of the f^7/6 and compute the first derivative
    with respect to frequency to get the expression below."""
    chi1 = chis+chia
    chi2 = chis-chia
    rhos = rho_funs(eta,chi)
    rho1 = rhos[0]
    rho2 = rhos[1]
    rho3 = rhos[2]
    if eta<0.25:
        Seta = np.sqrt(1-4*eta)
    else:
        Seta = 0.

    return amp_mult*(((-969 + 1804*eta)*np.pi**(2/3))/(1008.*Mf**(1/3)) \
            + ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*np.pi)/48. \
            + ((-27312085 - 10287648*chi2**2 - 10287648*chi1**2*(1 + Seta) \
            + 10287648*chi2**2*Seta + 24*(-1975055 + 857304*chi1**2 - 994896*chi1*chi2 + 857304*chi2**2)*eta \
            + 35371056*eta**2)*Mf**(1/3)*np.pi**(4/3))/6.096384e6 \
            + (5*Mf**(2/3)*np.pi**(5/3)*(chi2*(-285197*(-1 + Seta) \
            + 4*(-91902 + 1579*Seta)*eta - 35632*eta**2) + chi1*(285197*(1 + Seta) \
            - 4*(91902 + 1579*Seta)*eta - 35632*eta**2) + 42840*(-1 + 4*eta)*np.pi))/96768. \
            - (Mf*np.pi**2*(-336*(-3248849057 + 2943675504*chi1**2 - 3339284256*chi1*chi2 + 2943675504*chi2**2)*eta**2 - 324322727232*eta**3 \
            - 7*(-177520268561 + 107414046432*chi2**2 + 107414046432*chi1**2*(1 + Seta) - 107414046432*chi2**2*Seta \
            + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*np.pi) \
            + 12*eta*(-545384828789 - 176491177632*chi1*chi2 + 202603761360*chi2**2 + 77616*chi1**2*(2610335 + 995766*Seta) \
            - 77287373856*chi2**2*Seta + 5841690624*(chi1 + chi2)*np.pi + 21384760320*np.pi**2)))/3.0042980352e10 \
            + (7/3)*Mf**(4/3)*rho1 + (8/3)*Mf**(5/3)*rho2 + 3*Mf**2*rho3)

#############/ Amplitude: Merger-Ringdown functions ###########/

@njit()
def gamma_funs(eta,chi):
    """Phenom coefficients gamma1, ..., gamma3
    AmpMRDAnsatzFunc[]
    See corresponding row in Table 5 arXiv:1508.07253"""
    xi = -1 + chi
    gamma1 = 0.006927402739328343 + 0.03020474290328911*eta \
            + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta**2)*xi \
            + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta**2)*xi**2 \
            + (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta**2)*xi**3
    gamma2 = 1.010344404799477 + 0.0008993122007234548*eta \
            + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta**2)*xi \
            + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta**2)*xi**2 \
            + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta**2)*xi**3
    gamma3 = 1.3081615607036106 - 0.005537729694807678*eta \
            + (-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta**2)*xi \
            + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta**2)*xi**2 \
            + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta**2)*xi**3
    return (gamma1,gamma2,gamma3)

@njit()
def AmpMRDAnsatz(Mfs,fRD,fDM,eta,chi,amp_mult=1.):
    """Ansatz for the merger-ringdown amplitude. Equation 19 arXiv:1508.07253"""
    gammas = gamma_funs(eta,chi)
    gamma1 = gammas[0]
    gamma2 = gammas[1]
    gamma3 = gammas[2]
    fDMgamma3 = fDM*gamma3
    fminfRD = Mfs - fRD
    #Amps = amp_mult*Mfs**(-7/6)*np.exp(-fminfRD*gamma2/fDMgamma3)*(fDMgamma3*gamma1)/(fminfRD**2+fDMgamma3**2)
    Amps = amp_mult*fDMgamma3*gamma1*1/(Mfs**(7/6)*(fminfRD**2+fDMgamma3**2))*np.exp(-(gamma2/fDMgamma3)*fminfRD)
    return Amps

@njit()
def DAmpMRDAnsatz(f,fRD,fDM,eta,chi,amp_mult=1.):
    """first frequency derivative of AmpMRDAnsatz*f(7/6)"""
    gammas = gamma_funs(eta,chi)
    gamma1 = gammas[0]
    gamma2 = gammas[1]
    gamma3 = gammas[2]

    fDMgamma3 = fDM*gamma3
    fminfRD = f - fRD
    expfactor = np.exp((fminfRD*gamma2)/fDMgamma3)
    pow2pluspow2 = fminfRD**2 + fDMgamma3**2

    return amp_mult*((-2*fDM*fminfRD*gamma3*gamma1) / ( expfactor * pow2pluspow2**2) -(gamma2*gamma1) / ( expfactor * pow2pluspow2))


@njit()
def fmaxCalc(fRD,fDM,eta,chi):
    """Equation 20 arXiv:1508.07253 (called f_peak in paper)
    analytic location of maximum of AmpMRDAnsatz"""
    gammas = gamma_funs(eta,chi)
    gamma2 = gammas[1]
    gamma3 = gammas[2]

#  # NOTE: There's a problem with this expression from the paper becoming imaginary if gamma2>=1
#  # Fix: if gamma2 >= 1 then set the square root term to zero.
    if gamma2 <= 1:
        return np.abs(fRD + (fDM*(-1 + np.sqrt(1 - gamma2**2))*gamma3)/gamma2)
    else:
        return np.abs(fRD + (fDM*(-1)*gamma3)/gamma2)

###############/ Amplitude: Intermediate functions ############
#
## Phenom coefficients delta0, ..., delta4 determined from collocation method
## (constraining 3 values and 2 derivatives)
## #AmpIntAnsatzFunc[]

@njit()
def AmpIntAnsatz(Mfs,fRD,fDM,eta,chis,chia,chi,amp_mult=1.):
    """Ansatz for the intermediate amplitude. Equation 21 arXiv:1508.07253"""
    deltas = ComputeDeltasFromCollocation(eta,chis,chia,chi,fRD,fDM)
    #for itrf in range(NF_low,NF):
    #    Amp[itrf] = amp_mult*Mfs[itrf]**(-7/6)*(deltas[0] + deltas[1]*Mfs[itrf] + deltas[2]*Mfs[itrf]**2 + deltas[3]*Mfs[itrf]**3 + deltas[4]*Mfs[itrf]**4)
    Amp = amp_mult*1/Mfs**(7/6)*(deltas[0] + deltas[1]*Mfs + deltas[2]*Mfs**2 + deltas[3]*Mfs**3 + deltas[4]*Mfs**4)
    #Amp = amp_mult*Mfs**(-7/6)*(deltas[0] + deltas[1]*Mfs + deltas[2]*Mfs**2 + deltas[3]*Mfs**3 + deltas[4]*Mfs**4)
    return Amp

@njit()
def AmpIntColFitCoeff(eta,chi):
    """The function name stands for 'Amplitude Intermediate Collocation Fit Coefficient'
    This is the 'v2' value in Table 5 of arXiv:1508.07253"""
    xi = -1 + chi
    return 0.8149838730507785 + 2.5747553517454658*eta \
            + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta**2)*xi \
            + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta**2)*xi**2 \
            + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta**2)*xi**3

@njit()
def ComputeDeltasFromCollocation(eta,chis,chia,chi,MfRD,MfDM):
    """Calculates delta_i's
    Method described in arXiv:1508.07253 section 'Region IIa - intermediate'"""
    # Three evenly spaced collocation points in the interval [f1,f3].
    f1 = imrc.AMP_fJoin_INS
    f3 = fmaxCalc(MfRD,MfDM,eta,chi)
    dfx = (f3 - f1)/2
    f2 = f1 + dfx

    #amp_prefactors = AmpInsPrefactors(eta,chis,chia,rhos)

    #  # v1 is inspiral model evaluated at f1
    #  # d1 is derivative of inspiral model evaluated at f1
    #v1 = AmpInsAnsatz(np.zeros(1),np.array([f1]),eta,chis,chia,chi,0,1,1.)[0]*f1**(7/6)
    v1 = AmpInsAnsatz(f1,eta,chis,chia,chi,1.)*f1**(7/6)

    d1 = DAmpInsAnsatz(f1,eta,chis,chia,chi,1.)

    #  # v3 is merger-ringdown model evaluated at f3
    #  # d2 is derivative of merger-ringdown model evaluated at f3
    v3 = AmpMRDAnsatz(f3,MfRD,MfDM,eta,chi,1.)*f3**(7/6)
    d2 = DAmpMRDAnsatz(f3,MfRD,MfDM,eta,chi,1.)

    #  # v2 is the value of the amplitude evaluated at f2
    #  # they come from the fit of the collocation points in the intermediate region
    v2 = AmpIntColFitCoeff(eta, chi)

    #  # Now compute the delta_i's from the collocation coefficients
    #The following functions (delta{0,1,2,3,4}_fun) were derived
    #in mathematica according to
    #the constraints detailed in arXiv:1508.07253,
    #section 'Region IIa - intermediate'.
    #These are not given in the paper.
    #Can be rederived by solving Equation 21 for the constraints
    #given in Equations 22-26 in arXiv:1508.07253
    delta0 = -((d2*f1**5*f2**2*f3 - 2*d2*f1**4*f2**3*f3 + d2*f1**3*f2**4*f3 - d2*f1**5*f2*f3**2 + d2*f1**4*f2**2*f3**2
        - d1*f1**3*f2**3*f3**2 + d2*f1**3*f2**3*f3**2 + d1*f1**2*f2**4*f3**2 - d2*f1**2*f2**4*f3**2 + d2*f1**4*f2*f3**3 \
        + 2*d1*f1**3*f2**2*f3**3 - 2*d2*f1**3*f2**2*f3**3 - d1*f1**2*f2**3*f3**3 + d2*f1**2*f2**3*f3**3 - d1*f1*f2**4*f3**3 \
        - d1*f1**3*f2*f3**4 - d1*f1**2*f2**2*f3**4 + 2*d1*f1*f2**3*f3**4 + d1*f1**2*f2*f3**5 - d1*f1*f2**2*f3**5 \
        + 4*f1**2*f2**3*f3**2*v1 - 3*f1*f2**4*f3**2*v1 - 8*f1**2*f2**2*f3**3*v1 + 4*f1*f2**3*f3**3*v1 + f2**4*f3**3*v1 \
        + 4*f1**2*f2*f3**4*v1 + f1*f2**2*f3**4*v1 - 2*f2**3*f3**4*v1 - 2*f1*f2*f3**5*v1 + f2**2*f3**5*v1 - f1**5*f3**2*v2 \
        + 3*f1**4*f3**3*v2 - 3*f1**3*f3**4*v2 + f1**2*f3**5*v2 - f1**5*f2**2*v3 + 2*f1**4*f2**3*v3 - f1**3*f2**4*v3 \
        + 2*f1**5*f2*f3*v3 - f1**4*f2**2*f3*v3 - 4*f1**3*f2**3*f3*v3 + 3*f1**2*f2**4*f3*v3 - 4*f1**4*f2*f3**2*v3 \
        + 8*f1**3*f2**2*f3**2*v3 - 4*f1**2*f2**3*f3**2*v3) / ((f1 - f2)**2*(f1 - f3)**3*(f3-f2)**2))
    delta1 = -((-(d2*f1**5*f2**2) + 2*d2*f1**4*f2**3 - d2*f1**3*f2**4 - d2*f1**4*f2**2*f3 + 2*d1*f1**3*f2**3*f3 \
        + 2*d2*f1**3*f2**3*f3 - 2*d1*f1**2*f2**4*f3 - d2*f1**2*f2**4*f3 + d2*f1**5*f3**2 - 3*d1*f1**3*f2**2*f3**2 \
        - d2*f1**3*f2**2*f3**2 + 2*d1*f1**2*f2**3*f3**2 - 2*d2*f1**2*f2**3*f3**2 + d1*f1*f2**4*f3**2 + 2*d2*f1*f2**4*f3**2 \
        - d2*f1**4*f3**3 + d1*f1**2*f2**2*f3**3 + 3*d2*f1**2*f2**2*f3**3 - 2*d1*f1*f2**3*f3**3 - 2*d2*f1*f2**3*f3**3 \
        + d1*f2**4*f3**3 + d1*f1**3*f3**4 + d1*f1*f2**2*f3**4 - 2*d1*f2**3*f3**4 - d1*f1**2*f3**5 + d1*f2**2*f3**5 \
        - 8*f1**2*f2**3*f3*v1 + 6*f1*f2**4*f3*v1 + 12*f1**2*f2**2*f3**2*v1 - 8*f1*f2**3*f3**2*v1 - 4*f1**2*f3**4*v1 \
        + 2*f1*f3**5*v1 + 2*f1**5*f3*v2 - 4*f1**4*f3**2*v2 + 4*f1**2*f3**4*v2 - 2*f1*f3**5*v2 - 2*f1**5*f3*v3 \
        + 8*f1**2*f2**3*f3*v3 - 6*f1*f2**4*f3*v3 + 4*f1**4*f3**2*v3 - 12*f1**2*f2**2*f3**2*v3 + 8*f1*f2**3*f3**2*v3) \
        / ((f1 - f2)**2*(f1 - f3)**3*(-f2 + f3)**2))
    delta2 = -((d2*f1**5*f2 - d1*f1**3*f2**3 - 3*d2*f1**3*f2**3 + d1*f1**2*f2**4 + 2*d2*f1**2*f2**4 - d2*f1**5*f3 \
        + d2*f1**4*f2*f3 - d1*f1**2*f2**3*f3 + d2*f1**2*f2**3*f3 + d1*f1*f2**4*f3 - d2*f1*f2**4*f3 - d2*f1**4*f3**2 \
        + 3*d1*f1**3*f2*f3**2 + d2*f1**3*f2*f3**2 - d1*f1*f2**3*f3**2 + d2*f1*f2**3*f3**2 - 2*d1*f2**4*f3**2 - d2*f2**4*f3**2 \
        - 2*d1*f1**3*f3**3 + 2*d2*f1**3*f3**3 - d1*f1**2*f2*f3**3 - 3*d2*f1**2*f2*f3**3 + 3*d1*f2**3*f3**3 + d2*f2**3*f3**3 \
        + d1*f1**2*f3**4 - d1*f1*f2*f3**4 + d1*f1*f3**5 - d1*f2*f3**5 + 4*f1**2*f2**3*v1 - 3*f1*f2**4*v1 + 4*f1*f2**3*f3*v1 \
        - 3*f2**4*f3*v1 - 12*f1**2*f2*f3**2*v1 + 4*f2**3*f3**2*v1 + 8*f1**2*f3**3*v1 - f1*f3**4*v1 - f3**5*v1 - f1**5*v2 \
        - f1**4*f3*v2 + 8*f1**3*f3**2*v2 - 8*f1**2*f3**3*v2 + f1*f3**4*v2 + f3**5*v2 + f1**5*v3 - 4*f1**2*f2**3*v3 + 3*f1*f2**4*v3 \
        + f1**4*f3*v3 - 4*f1*f2**3*f3*v3 + 3*f2**4*f3*v3 - 8*f1**3*f3**2*v3 + 12*f1**2*f2*f3**2*v3 - 4*f2**3*f3**2*v3) \
        / ((f1 - f2)**2*(f1 - f3)**3*(-f2 + f3)**2))
    delta3 = -((-2*d2*f1**4*f2 + d1*f1**3*f2**2 + 3*d2*f1**3*f2**2 - d1*f1*f2**4 - d2*f1*f2**4 + 2*d2*f1**4*f3 \
        - 2*d1*f1**3*f2*f3 - 2*d2*f1**3*f2*f3 + d1*f1**2*f2**2*f3 - d2*f1**2*f2**2*f3 + d1*f2**4*f3 + d2*f2**4*f3 \
        + d1*f1**3*f3**2 - d2*f1**3*f3**2 - 2*d1*f1**2*f2*f3**2 + 2*d2*f1**2*f2*f3**2 + d1*f1*f2**2*f3**2 - d2*f1*f2**2*f3**2 \
        + d1*f1**2*f3**3 - d2*f1**2*f3**3 + 2*d1*f1*f2*f3**3 + 2*d2*f1*f2*f3**3 - 3*d1*f2**2*f3**3 - d2*f2**2*f3**3 \
        - 2*d1*f1*f3**4 + 2*d1*f2*f3**4 - 4*f1**2*f2**2*v1 + 2*f2**4*v1 + 8*f1**2*f2*f3*v1 - 4*f1*f2**2*f3*v1 \
        - 4*f1**2*f3**2*v1 + 8*f1*f2*f3**2*v1 - 4*f2**2*f3**2*v1 - 4*f1*f3**3*v1 + 2*f3**4*v1 + 2*f1**4*v2 \
        - 4*f1**3*f3*v2 + 4*f1*f3**3*v2 - 2*f3**4*v2 - 2*f1**4*v3 + 4*f1**2*f2**2*v3 - 2*f2**4*v3 + 4*f1**3*f3*v3 \
        - 8*f1**2*f2*f3*v3 + 4*f1*f2**2*f3*v3 + 4*f1**2*f3**2*v3 - 8*f1*f2*f3**2*v3 + 4*f2**2*f3**2*v3) \
        / ((f1 - f2)**2*(f1 - f3)**3*(-f2 + f3)**2))
    delta4 = -((d2*f1**3*f2 - d1*f1**2*f2**2 - 2*d2*f1**2*f2**2 + d1*f1*f2**3 + d2*f1*f2**3 - d2*f1**3*f3 + 2*d1*f1**2*f2*f3 \
        + d2*f1**2*f2*f3 - d1*f1*f2**2*f3 + d2*f1*f2**2*f3 - d1*f2**3*f3 - d2*f2**3*f3 - d1*f1**2*f3**2 + d2*f1**2*f3**2 \
        - d1*f1*f2*f3**2 - 2*d2*f1*f2*f3**2 + 2*d1*f2**2*f3**2 + d2*f2**2*f3**2 + d1*f1*f3**3 - d1*f2*f3**3 + 3*f1*f2**2*v1 \
        - 2*f2**3*v1 - 6*f1*f2*f3*v1 + 3*f2**2*f3*v1 + 3*f1*f3**2*v1 - f3**3*v1 - f1**3*v2 + 3*f1**2*f3*v2 - 3*f1*f3**2*v2 \
        + f3**3*v2 + f1**3*v3 - 3*f1*f2**2*v3 + 2*f2**3*v3 - 3*f1**2*f3*v3 + 6*f1*f2*f3*v3 - 3*f2**2*f3*v3) \
        / ((f1 - f2)**2*(f1 - f3)**3*(-f2 + f3)**2))

    return (delta0,delta1,delta2,delta3,delta4)

###############/ Amplitude: glueing function ##############

#@njit()
def IMRPhenDAmplitude(Mfs,eta,chis,chia,NF,amp_mult=1.):
    """This function computes the IMR amplitude given phenom coefficients.
    Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    The inspiral, intermediate and merger-ringdown amplitude parts"""

    #  # Transition frequencies
    chi = chiPN(eta,chis,chia)

    finspin = FinalSpin0815(eta, chis, chia) #FinalSpin0815 - 0815 is like a version number

    fRD,fDM = fringdown(eta,chis,chia,finspin)

    fMRDJoinAmp = fmaxCalc(fRD,fDM,eta,chi)


    Amps = np.zeros(NF)

    if Mfs[-1]>imrc.f_CUT:
        itrFCut = np.searchsorted(Mfs,imrc.f_CUT,side='right')
    else:
        itrFCut = NF

    if Mfs[itrFCut-1]<imrc.AMP_fJoin_INS:
        itrfMRDAmp = itrFCut
        itrfInt = itrFCut
    elif Mfs[itrFCut-1]<fMRDJoinAmp:
        itrfMRDAmp = itrFCut
        itrfInt = np.searchsorted(Mfs,imrc.AMP_fJoin_INS)
    else:
        itrfMRDAmp = np.searchsorted(Mfs,fMRDJoinAmp)
        itrfInt = np.searchsorted(Mfs,imrc.AMP_fJoin_INS)

    #   split the calculation to just 1 of 3 possible mutually exclusive ranges
    #if itrfInt>0:
    amp0 = amp_mult*amp0Func(eta)
    Amps[0:itrfInt] = AmpInsAnsatz(Mfs[0:itrfInt],eta,chis,chia,chi,amp_mult=amp0) #Inspiral range
    #if itrfInt<itrfMRDAmp:
    Amps[itrfInt:itrfMRDAmp] = AmpIntAnsatz(Mfs[itrfInt:itrfMRDAmp],fRD,fDM,eta,chis,chia,chi,amp_mult=amp0) #Intermediate range
    #if itrfMRDAmp<NF:
    Amps[itrfMRDAmp:itrFCut] = AmpMRDAnsatz(Mfs[itrfMRDAmp:itrFCut],fRD,fDM,eta,chi,amp_mult=amp0) # MRD range

    #Amps *= amp0Func(eta)#/fs**(7/6)
    return Amps

#/********************************* Phase functions *********************************/
#
################ Phase: Ringdown functions #############/
@njit()
def alphaFits(eta,chi):
    """alpha_i i=1,2,3,4,5 are the phenomenological intermediate coefficients depending on eta and chiPN
    PhiRingdownAnsatz is the ringdown phasing in terms of the alpha_i coefficients
    See corresponding row in Table 5 arXiv:1508.07253"""
    xi = -1 + chi
    alpha1 = 43.31514709695348 + 638.6332679188081*eta \
            + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta**2)*xi \
            + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta**2)*xi**2 \
            + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta**2)*xi**3
    alpha2 = -0.07020209449091723 - 0.16269798450687084*eta \
            + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta**2)*xi \
            + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta**2)*xi**2 \
            + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta**2)*xi**3
    alpha3 = 9.5988072383479 - 397.05438595557433*eta \
            + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta**2)*xi \
            + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta**2)*xi**2 \
            + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta**2)*xi**3
    alpha4 = -0.02989487384493607 + 1.4022106448583738*eta \
            + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta**2)*xi \
            + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta**2)*xi**2 \
            + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta**2)*xi**3
    alpha5 = 0.9974408278363099 - 0.007884449714907203*eta \
            + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta**2)*xi \
            + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta**2)*xi**2 \
            + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta**2)*xi**3
    return (alpha1,alpha2,alpha3,alpha4,alpha5)

@njit()
def PhiMRDAnsatzInt(Mf,fRD,fDM,eta,chi):
    """Ansatz for the merger-ringdown phase Equation 14 arXiv:1508.07253"""
    alphas = alphaFits(eta,chi)
    fq = np.sqrt(np.sqrt(Mf))
    return alphas[0]/eta*Mf - alphas[1]/eta/Mf + 4/3/eta*alphas[2]*fq**3 + alphas[3]/eta*np.arctan((Mf-alphas[4]*fRD)/fDM)
    #return  alphas[0]/eta*Mf-alphas[1]/eta/Mf+ 4/3/eta*alphas[2]*Mf**(3/4)+alphas[3]/eta*np.arctan((Mf-alphas[4]*fRD)/fDM)

@njit()
def DPhiMRD(Mf,fRD,fDM,eta,chi):
    """First frequency derivative of PhiMRDAnsatzInt"""
    alphas = alphaFits(eta,chi)
    fq = np.sqrt(np.sqrt(Mf))
    return alphas[0]/eta + alphas[1]/eta/Mf**2 + alphas[2]/eta/fq + alphas[3]/eta*fDM/(fDM**2+(Mf-alphas[4]*fRD)**2)
    #return alphas[0]/eta+alphas[1]/eta/Mf**2+alphas[2]/eta/Mf**(1/4)+alphas[3]/eta/fDM/(1+(Mf-alphas[4]*fRD)**2/fDM**2)

@njit()
def DDPhiMRD(Mf,fRD,fDM,eta,chi):
    """First frequency derivative of PhiMRDAnsatzInt"""
    alphas = alphaFits(eta,chi)
    fq = np.sqrt(np.sqrt(Mf))
    return -2*alphas[1]/eta/Mf**3 - 1/4*alphas[2]/eta/fq**5 - 2*alphas[3]/eta*fDM*(Mf-alphas[4]*fRD)/(fDM**2+(Mf-alphas[4]*fRD)**2)**2
    #return -2*alphas[1]/eta/Mf**3-1/4*alphas[2]/eta/Mf**(5/4)-2*alphas[3]/eta*fDM*(Mf-alphas[4]*fRD)/(fDM**2+(Mf-alphas[4]*fRD)**2)**2

###############/ Phase: Intermediate functions ##############/

@njit()
def betaFits(eta,chi):
    """beta_i i=1,2,3 are the phenomenological intermediate coefficients depending on eta and chiPN
    PhiIntAnsatz is the intermediate phasing in terms of the beta_i coefficients
    [Beta]1Fit = PhiIntFitCoeff[Chi]PNFunc[[Eta], [Chi]PN][[1]]"""
    xi = -1 + chi
    beta1 = 97.89747327985583 - 42.659730877489224*eta \
            + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta**2)*xi \
            + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta**2)*xi**2 \
            + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta**2)*xi**3
    beta2 = -3.282701958759534 - 9.051384468245866*eta \
            + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta**2)*xi \
            + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta**2)*xi**2 \
            + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta**2)*xi**3
    beta3 = -0.000025156429818799565 + 0.000019750256942201327*eta \
            + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta**2)*xi \
            + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta**2)*xi**2 \
            + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta**2)*xi**3
    return (beta1,beta2,beta3)

@njit()
def PhiIntAnsatz(Mf,eta,chi):
    """ansatz for the intermediate phase defined by Equation 16 arXiv:1508.07253"""
    #   ComputeIMRPhenDPhaseConnectionCoefficients
    #   IMRPhenDPhase
    betas = betaFits(eta,chi)
    return  betas[0]/eta*Mf-betas[2]/eta/3/Mf**3+betas[1]/eta*np.log(Mf)

@njit()
def DPhiIntAnsatz(Mf,eta,chi):
    """First frequency derivative of PhiIntAnsatz
    (this time with 1./eta explicitly factored in)"""
    betas = betaFits(eta,chi)
    return betas[0]/eta+betas[2]/eta/Mf**4+betas[1]/eta/Mf

@njit()
def DDPhiIntAnsatz(Mf,eta,chi):
    """First frequency derivative of PhiIntAnsatz
    (this time with 1./eta explicitly factored in)"""
    betas = betaFits(eta,chi)
    return -4*betas[2]/eta/Mf**5-betas[1]/eta/Mf**2


###############/ Phase: Inspiral functions ##############/

@njit()
def sigmaFits(eta,chi):
    """sigma_i i=1,2,3,4 are the phenomenological inspiral coefficients depending on eta and chiPN
    PhiInsAnsatzInt is a souped up TF2 phasing which depends on the sigma_i coefficients
    See corresponding row in Table 5 arXiv:1508.07253"""
    xi = -1 + chi
    sigma1 = 2096.551999295543 + 1463.7493168261553*eta \
        + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta**2)*xi \
        + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta**2)*xi**2 \
        + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta**2)*xi**3
    sigma2 = -10114.056472621156 - 44631.01109458185*eta \
        + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta**2)*xi \
        + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta**2)*xi**2 \
        + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta**2)*xi**3
    sigma3 = 22933.658273436497 + 230960.00814979506*eta \
        + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta**2)*xi \
        + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta**2)*xi**2 \
        + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta**2)*xi**3
    sigma4 = -14621.71522218357 - 377812.8579387104*eta \
        + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta**2)*xi \
        + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta**2)*xi**2 \
        + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta**2)*xi**3
    return (sigma1,sigma2,sigma3,sigma4)

@njit()
def PhiInsPrefactors(eta,chis,chia,chi):
    v,vlogv = PNPhasingSeriesTaylorF2(eta,chis,chia)
    #  # PN phasing series
    minus_five_thirds = v[0]/np.pi**(5/3)
    minus_one = v[2]/np.pi
    minus_two_thirds = v[3]/np.pi**(2/3)
    minus_third = v[4]/np.pi**(1/3)
    initial_phasing = v[5]-np.pi/4
    third = v[6]*np.pi**(1/3)
    two_thirds = v[7]*np.pi**(2/3)

    zero_with_logv = vlogv[5]
    third_with_logv = vlogv[6]*np.pi**(1/3)

    #higher order terms that were calibrated for PhenomD
    sigmas = sigmaFits(eta,chi)
    one = sigmas[0]/eta
    four_thirds = 3/4*sigmas[1]/eta
    five_thirds = 3/5*sigmas[2]/eta
    two = 1/2*sigmas[3]/eta
    prefactors_ini = (minus_five_thirds,minus_one,minus_two_thirds,minus_third, \
                        initial_phasing,third,two_thirds,one,four_thirds,five_thirds,two)
                        #initial_phasing,third,two_thirds,0.,0.,0.,0.)
    prefactors_log = (zero_with_logv,third_with_logv)
    return prefactors_ini,prefactors_log

@njit()
def PhiInsAnsatzInt(Mfs,eta,chis,chia,chi):
    """Ansatz for the inspiral phase.
    We call the LAL TF2 coefficients here.
    The exact values of the coefficients used are given
    as comments in the top of this file
    Defined by Equation 27 and 28 arXiv:1508.07253"""
    #Assemble PN phasing series
    prefactors_ini,prefactors_log = PhiInsPrefactors(eta,chis,chia,chi)

    fv = Mfs**(1/3)
    logv = 1/3*np.log(np.pi)+np.log(fv)
    Phi = 1/fv**5*(prefactors_ini[0] \
        + prefactors_ini[1]*fv**2 \
        + prefactors_ini[2]*fv**3\
        + prefactors_ini[3]*fv**4\
        + (prefactors_ini[4]+prefactors_log[0]*logv)*fv**5\
        + (prefactors_ini[5]+prefactors_log[1]*logv)*fv**6\
        + prefactors_ini[6]*fv**7\
        + prefactors_ini[7]*fv**8\
        + prefactors_ini[8]*fv**9\
        + prefactors_ini[9]*fv**10\
        + prefactors_ini[10]*fv**11\
        )
    return Phi


@njit()
def DPhiInsAnsatzInt(Mfs,eta,chis,chia,chi):
    """First frequency derivative of PhiInsAnsatzInt"""
    #Assemble PN phasing series
    fv = Mfs**(1/3)
    logfv = 1/3*np.log(np.pi)+np.log(fv)
    prefactors_ini,prefactors_log = PhiInsPrefactors(eta,chis,chia,chi)
    dPhi = 1/fv**8*(0 \
        - 5/3*prefactors_ini[0] \
        - 3/3*prefactors_ini[1]*fv**2 \
        - 2/3*prefactors_ini[2]*fv**3\
        - 1/3*prefactors_ini[3]*fv**4\
        + 1/3*prefactors_log[0]*fv**5 \
        + 1/3*prefactors_log[1]*fv**6*logfv \
        + 1/3*(prefactors_log[1]+prefactors_ini[5])*fv**6 \
        + 2/3*prefactors_ini[6]*fv**7\
        + 3/3*prefactors_ini[7]*fv**8\
        + 4/3*prefactors_ini[8]*fv**9\
        + 5/3*prefactors_ini[9]*fv**10\
        + 6/3*prefactors_ini[10]*fv**11\
            )
    return dPhi

@njit()
def DDPhiInsAnsatzInt(Mfs,eta,chis,chia,chi):
    """First frequency derivative of PhiInsAnsatzInt"""
    #Assemble PN phasing series
    fv = Mfs**(1/3)
    logfv = 1/3*np.log(np.pi)+np.log(fv)
    prefactors_ini,prefactors_log = PhiInsPrefactors(eta,chis,chia,chi)
    ddPhi = 1/fv**11*(0 \
        + 40/9*prefactors_ini[0] \
        + 18/9*prefactors_ini[1]*fv**2 \
        + 10/9*prefactors_ini[2]*fv**3\
        +  4/9*prefactors_ini[3]*fv**4\
        -  3/9*prefactors_log[0]*fv**5 \
        -  2/9*prefactors_log[1]*fv**6*logfv \
        -  1/9*(prefactors_log[1]+2*prefactors_ini[5])*fv**6 \
        -  2/9*prefactors_ini[6]*fv**7\
        +  4/9*prefactors_ini[8]*fv**9\
        + 10/9*prefactors_ini[9]*fv**10\
        + 18/9*prefactors_ini[10]*fv**11\
            )
    return ddPhi

################/ Phase: glueing function ################

def NextPow2(n):
    """ use pow here, not bit-wise shift, as the latter seems to run against an upper cutoff long before SIZE_MAX, at least on some platforms"""
    return np.int64(2**np.ceil(np.log2(n)))

@njit()
def ComputeIMRPhenDPhaseConnectionCoefficients(fRD,fDM,eta,chis,chia,chi,fMRDJoinPhi):
    """This function aligns the three phase parts (inspiral, intermediate and merger-rindown)
    such that they are c^1 continuous at the transition frequencies
    Defined in VIII. Full IMR Waveforms arXiv:1508.07253"""
#   Compute C1Int and C2Int coeffs
#   Equations to solve for to get C(1) continuous join
#   PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
#   Joining at fInsJoin
#   PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
#   PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
    DPhiIns = DPhiInsAnsatzInt(imrc.PHI_fJoin_INS,eta,chis,chia,chi)
    DPhiInt = DPhiIntAnsatz(imrc.PHI_fJoin_INS,eta,chi)
    C2Int = DPhiIns - DPhiInt

    phiC1_ref = PhiInsAnsatzInt(imrc.PHI_fJoin_INS,eta,chis,chia,chi)
    C1Int = phiC1_ref-PhiIntAnsatz(imrc.PHI_fJoin_INS,eta,chi) - C2Int*imrc.PHI_fJoin_INS

#   Compute C1MRD and C2MRD coeffs
#   Equations to solve for to get C(1) continuous join
#   PhiInsInt (f)  =   PhiMRD (f) + C1MRD + C2MRD f
#   Joining at fMRDJoin
#   Where \[Phi]InsInt(f) is the \[Phi]Ins+\[Phi]Int joined function
#   PhiInsInt (fMRDJoin)  =   PhiMRD (fMRDJoin) + C1MRD + C2MRD fMRDJoin
#   PhiInsInt'(fMRDJoin)  =   PhiMRD'(fMRDJoin) + C2MRD
#   temporary Intermediate Phase function to Join up the Merger-Ringdown
    PhiIntTempVal = PhiIntAnsatz(fMRDJoinPhi,eta,chi) + C1Int + C2Int*fMRDJoinPhi
    DPhiIntTempVal = C2Int+DPhiIntAnsatz(fMRDJoinPhi,eta,chi)
    DPhiMRDVal = DPhiMRD(fMRDJoinPhi,fRD,fDM,eta,chi)
    C2MRD = DPhiIntTempVal - DPhiMRDVal
    C1MRD = PhiIntTempVal - PhiMRDAnsatzInt(fMRDJoinPhi,fRD,fDM,eta,chi) - C2MRD*fMRDJoinPhi
    return C1Int,C2Int,C1MRD,C2MRD

#@njit()
def IMRPhenDPhase(Mfs,Mt_sec,eta,chis,chia,NF,fRef_in,phi0):
    """This function computes the IMR phase given phenom coefficients.
    Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    The inspiral, intermediate and merger-ringdown phase parts
    split the calculation to just 1 of 3 possible mutually exclusive ranges
    Mfs must be sorted"""
    chi = chiPN(eta,chis,chia)

    finspin = FinalSpin0815(eta, chis, chia) #FinalSpin0815 - 0815 is like a version number

    fRD,fDM = fringdown(eta,chis,chia,finspin)

    #   Transition frequencies
    #   Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    fMRDJoinPhi = fRD/2
    fMRDJoinAmp = fmaxCalc(fRD,fDM,eta,chi)

    # Compute coefficients to make phase C^1 continuous (phase and first derivative)
    C1Int,C2Int,C1MRD,C2MRD = ComputeIMRPhenDPhaseConnectionCoefficients(fRD,fDM,eta,chis,chia,chi,fMRDJoinPhi)

    #time shift so that peak amplitude is approximately at t=0
    #For details see https:#www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timPD_EDOMain
    TTRef = -DPhiMRD(fMRDJoinAmp,fRD,fDM,eta,chi)

    # NOTE: opposite Fourier convention with respect to PhenomD - to ensure 22 mode has power for positive f
    if Mfs[-1]>imrc.f_CUT:
        itrFCut = np.searchsorted(Mfs,imrc.f_CUT,side='right')
    else:
        itrFCut = NF

    # NOTE: previously fRef=0 was by default fRef=fmin, now fRef defaults to fmaxCalc (fpeak in the paper)
    # If fpeak is outside of the frequency range, take the last frequency
    if fRef_in == 0.:
        MfRef = min(fMRDJoinAmp, Mfs[itrFCut-1])
    else:
        MfRef = fRef_in
    if MfRef<imrc.PHI_fJoin_INS:
        phifRef = PhiInsAnsatzInt(MfRef,eta,chis,chia,chi)
    elif MfRef<fMRDJoinPhi:
        phifRef = PhiIntAnsatz(MfRef,eta,chi)+C1Int+C2Int*MfRef
    else:
        phifRef = PhiMRDAnsatzInt(MfRef,fRD,fDM,eta,chi)+C1MRD+C2MRD*MfRef#MRD range
    phifRef += 2*phi0+TTRef*MfRef
    phifRefIns = phifRef
    phifRefInt = phifRef-C1Int
    phifRefMRD = phifRef-C1MRD

    TTRefIns = TTRef
    TTRefInt = TTRef+C2Int
    TTRefMRD = TTRef+C2MRD

    Phis = np.zeros(NF)
    if Mfs[itrFCut-1]<imrc.PHI_fJoin_INS:
        itrfMRDPhi = itrFCut
        itrfInt = itrFCut
    elif Mfs[itrFCut-1]<fMRDJoinPhi:
        itrfMRDPhi = itrFCut
        itrfInt = np.searchsorted(Mfs,imrc.PHI_fJoin_INS)
    else:
        itrfMRDPhi = np.searchsorted(Mfs,fMRDJoinPhi)
        itrfInt = np.searchsorted(Mfs,imrc.PHI_fJoin_INS)

    Phis[0:itrfInt] = PhiInsAnsatzInt(Mfs[0:itrfInt],eta,chis,chia,chi)-phifRefIns+TTRefIns*Mfs[0:itrfInt] #Ins range
    Phis[itrfInt:itrfMRDPhi] = PhiIntAnsatz(Mfs[itrfInt:itrfMRDPhi],eta,chi)-phifRefInt+TTRefInt*Mfs[itrfInt:itrfMRDPhi] #intermediate range
    Phis[itrfMRDPhi:itrFCut] = PhiMRDAnsatzInt(Mfs[itrfMRDPhi:itrFCut],fRD,fDM,eta,chi)-phifRefMRD+TTRefMRD*Mfs[itrfMRDPhi:itrFCut]#MRD range
    #Phis[:itrFCut] -= t0*Mfs[:itrFCut]

    times = np.zeros(NF)
    if imrc.findT:
        times[0:itrfInt] = DPhiInsAnsatzInt(Mfs[0:itrfInt],eta,chis,chia,chi)+TTRefIns #Ins range
        times[itrfInt:itrfMRDPhi] = DPhiIntAnsatz(Mfs[itrfInt:itrfMRDPhi],eta,chi)+TTRefInt #intermediate range
        times[itrfMRDPhi:itrFCut] = DPhiMRD(Mfs[itrfMRDPhi:itrFCut],fRD,fDM,eta,chi)+TTRefMRD#MRD range

    times[:itrFCut] *= Mt_sec/(2*np.pi)

    return Phis,times,TTRef,MfRef,itrFCut
