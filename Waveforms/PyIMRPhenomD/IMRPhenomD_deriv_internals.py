"""Python implementation of IMRPhenomD behavior by Matthew Digman copyright 2021
Contains new behavior that includes derivatives, different from the C version"""
import numpy as np
#nb.parfors.parfor.sequential_parfor_lowering = True
from numba import njit,prange
import Waveforms.PyIMRPhenomD.IMRPhenomD_const as imrc
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import chiPN,FinalSpin0815,fringdown,fmaxCalc,ComputeIMRPhenDPhaseConnectionCoefficients
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import DPhiInsAnsatzInt,PhiInsAnsatzInt,DPhiIntAnsatz,PhiIntAnsatz,DPhiMRD,PhiMRDAnsatzInt
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import PNPhasingSeriesTaylorF2
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import sigmaFits,betaFits,alphaFits
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import amp0Func,ComputeDeltasFromCollocation,gamma_funs,rho_funs
#from IMRPhenomD_internals import AmpIn

@njit()
def PhiInsPrefactorsMt(eta,Mt_sec,chis,chia,chi):
    """Helper function to get the prefactors for PhiIns"""
    v,vlogv = PNPhasingSeriesTaylorF2(eta,chis,chia)
    #  # PN phasing series
    minus_five_thirds = v[0]/Mt_sec**(5/3)/np.pi**(5/3)
    minus_one = v[2]/Mt_sec**(3/3)/np.pi
    minus_two_thirds = v[3]/Mt_sec**(2/3)/np.pi**(2/3)
    minus_third = v[4]/Mt_sec**(1/3)/np.pi**(1/3)
    initial_phasing = (v[5]-np.pi/4)#/Mt_sec**(5/3)
    third = v[6]*Mt_sec**(1/3)*np.pi**(1/3)
    two_thirds = v[7]*Mt_sec**(2/3)*np.pi**(2/3)

    zero_with_logv = vlogv[5]#/Mt_sec**(5/3)
    third_with_logv = vlogv[6]*Mt_sec**(1/3)*np.pi**(1/3)

    #higher order terms that were calibrated for PhenomD
    #TODO check pis on these terms
    sigmas = sigmaFits(eta,chi)
    one = sigmas[0]/eta*Mt_sec**(3/3)
    four_thirds = 3/4*sigmas[1]/eta*Mt_sec**(4/3)
    five_thirds = 3/5*sigmas[2]/eta*Mt_sec**(5/3)
    two = 1/2*sigmas[3]/eta*Mt_sec**(6/3)
    prefactors_ini = (minus_five_thirds,minus_one,minus_two_thirds,minus_third, \
                        initial_phasing,third,two_thirds,one,four_thirds,five_thirds,two)
                        #initial_phasing,third,two_thirds,0.,0.,0.,0.)
    prefactors_log = (zero_with_logv,third_with_logv)
    return prefactors_ini,prefactors_log

@njit()
def AmpInsPrefactorsMt(Mt_sec,eta,chis,chia,rhos):
    """Helper function to get the prefactors for AmpIns"""
    rho1 = rhos[0]
    rho2 = rhos[1]
    rho3 = rhos[2]

    if eta<0.25:
        delta = np.sqrt(1-4*eta)
    else:
        delta = 0.

    two_thirds = 1/672*np.pi**(2/3)*Mt_sec**(2/3)*(-969 + 1804*eta)
    one = 1/24*np.pi*Mt_sec*(81*(chis+chia*delta)-44*chis*eta)
    four_thirds = 1/8128512*np.pi**(4/3)*Mt_sec**(4/3)\
                            * (-27312085 - 41150592*chia*chis*delta\
                            + 254016*chis**2*(-81 + 68*eta) + 254016*chia**2*(-81 + 256*eta)\
                            + 24*eta*(-1975055 + 1473794*eta))
    five_thirds =1/16128*np.pi**(5/3)*Mt_sec**(5/3)\
                            * (chia*delta*(285197 - 6316*eta) + chis*(285197 - 136*eta*(2703 + 262*eta))\
                            + 21420*np.pi*(-1 + 4*eta))
    two = 1/60085960704*Mt_sec**2*np.pi**2\
            * (-1242641879927 + 6544617945468*eta\
            + 931392*chia**2*(1614569 + 4*eta*(-1873643 + 832128*eta))\
            + 1862784*chia*delta*(chis*(1614569 - 1991532*eta) + 83328*np.pi)\
            + 336*(2772*chis**2*(1614569 + 16*eta*(-184173 + 57451*eta)) - 14902272*chis*(-31 + 28*eta)*np.pi\
            + eta*(eta*(-3248849057 + 965246212*eta) - 763741440*np.pi**2)))
    seven_thirds = rho1*Mt_sec**(7/3)
    eight_thirds = rho2*Mt_sec**(8/3)
    three = rho3*Mt_sec**3
    return (two_thirds,one,four_thirds,five_thirds,two,seven_thirds,eight_thirds,three)

@njit()
def AmpInsAnsatzInplace(Amps,fs,Mt_sec,eta,chis,chia,chi,amp_mult,NF_low,NF):
    """The Newtonian term in LAL is fine and we should use exactly the same (either hardcoded or call).
    We just use the Mathematica expression for convenience.
    Inspiral amplitude plus rho phenom coefficents. rho coefficients computed in rho_funs function.
    Amplitude is a re-expansion. See 1508.07253 and Equation 29, 30 and Appendix B arXiv:1508.07253 for details"""
    rhos = rho_funs(eta,chi)
    amp_prefactors = AmpInsPrefactorsMt(Mt_sec,eta,chis,chia,rhos)
    amp0 = amp_mult/Mt_sec**(7/6)

    floc = fs[NF_low:NF]
    #fv = floc**(1/3)
    fv = floc**(1/3)

    Amps[NF_low:NF] = 1/np.sqrt(fv**7)*( \
              amp0 \
            + amp0*amp_prefactors[0]*fv**2 \
            + amp0*amp_prefactors[1]*floc \
            + amp0*amp_prefactors[2]*fv**4 \
            + amp0*amp_prefactors[3]*fv**5 \
            + amp0*amp_prefactors[4]*fv**6 \
            + amp0*amp_prefactors[5]*fv**7 \
            + amp0*amp_prefactors[6]*fv**8 \
            + amp0*amp_prefactors[7]*fv**9 \
            )

    return Amps

@njit()
def AmpIntAnsatzInplace(Amps,fs,Mt_sec,fRD,fDM,eta,chis,chia,chi,amp_mult,NF_low,NF):
    """Ansatz for the intermediate amplitude. Equation 21 arXiv:1508.07253"""
    deltas = ComputeDeltasFromCollocation(eta,chis,chia,chi,fRD,fDM)
    amp0 = amp_mult/Mt_sec**(7/6)

    floc = fs[NF_low:NF]

    Amps[NF_low:NF] = amp0*1/floc**(7/6)*(deltas[0] + deltas[1]*Mt_sec*floc + deltas[2]*Mt_sec**2*floc**2 + deltas[3]*Mt_sec**3*floc**3 + deltas[4]*Mt_sec**4*floc**4)
    return Amps

@njit()
def AmpMRDAnsatzInplace(Amps,fs,Mt_sec,fRD,fDM,eta,chi,amp_mult,NF_low,NF):
    """Ansatz for the merger-ringdown amplitude. Equation 19 arXiv:1508.07253"""
    gammas = gamma_funs(eta,chi)
    gamma1 = gammas[0]
    gamma2 = gammas[1]
    gamma3 = gammas[2]

    amp0 = amp_mult/Mt_sec**(7/6)

    fDMgamma3 = fDM*gamma3/Mt_sec

    floc = fs[NF_low:NF]

    fminfRD = floc - fRD/Mt_sec

    Amps[NF_low:NF] = amp0*fDMgamma3/Mt_sec*gamma1*1/(floc**(7/6)*(fminfRD**2+fDMgamma3**2))*np.exp(-(gamma2/fDMgamma3)*fminfRD)
    return Amps

@njit()
def AmpPhaseSeriesInsAnsatz(Phi,dPhi,ddPhi,Amps,fs,Mt_sec,eta,chis,chia,chi,phi_ref,TTRef,amp_mult,NF_low,NF):
    """Ansatz for the inspiral phase. and amplitude
    We call the LAL TF2 coefficients here.
    The exact values of the coefficients used are given
    as comments in the top of this file
    Defined by Equation 27 and 28 arXiv:1508.07253"""
    #Assemble PN phasing series
    prefactors_ini,prefactors_log = PhiInsPrefactorsMt(eta,Mt_sec,chis,chia,chi)
    rhos = rho_funs(eta,chi)
    amp_prefactors = AmpInsPrefactorsMt(Mt_sec,eta,chis,chia,rhos)
    amp0 = amp_mult/Mt_sec**(7/6)

    dm = 1/(2*np.pi)

    for itrf in prange(NF_low,NF):
        f = fs[itrf]
        fv = f**(1/3)
        logfv = 1/3*np.log(Mt_sec*np.pi)+1/3*np.log(f)

        Amps[itrf] = 1/np.sqrt(fv**7)*(
                  amp0 \
                + amp0*amp_prefactors[0]*fv**2 \
                + amp0*amp_prefactors[1]*f \
                + amp0*amp_prefactors[2]*fv**4 \
                + amp0*amp_prefactors[3]*fv**5 \
                + amp0*amp_prefactors[4]*fv**6 \
                + amp0*amp_prefactors[5]*fv**7 \
                + amp0*amp_prefactors[6]*fv**8 \
                + amp0*amp_prefactors[7]*fv**9 \
            )

        Phi[itrf] = 1/fv**5*(prefactors_ini[0] \
            + prefactors_ini[1]*fv**2 \
            + prefactors_ini[2]*f\
            + prefactors_ini[3]*fv**4 \
            )\
            + prefactors_ini[5]*fv\
            + prefactors_ini[6]*fv**2\
            + (prefactors_ini[7]+TTRef/dm)*f\
            + prefactors_ini[8]*fv**4\
            + prefactors_ini[9]*fv**5\
            + prefactors_ini[10]*f**2\
            + prefactors_log[0]*logfv\
            + prefactors_log[1]*logfv*fv\
            + (prefactors_ini[4]-phi_ref)\

        if imrc.findT:
            dPhi[itrf] = 1/fv**8*( \
                - 5/3*dm*prefactors_ini[0] \
                - 3/3*dm*prefactors_ini[1]*fv**2 \
                - 2/3*dm*prefactors_ini[2]*f\
                - 1/3*dm*prefactors_ini[3]*fv**4\
                + 1/3*dm*prefactors_log[0]*fv**5 \
                + 1/3*dm*(prefactors_log[1]+prefactors_ini[5])*f**2 \
                + 2/3*dm*prefactors_ini[6]*fv**7\
                )\
                + 4/3*dm*prefactors_ini[8]*fv\
                + 5/3*dm*prefactors_ini[9]*fv**2\
                + 6/3*dm*prefactors_ini[10]*f\
                + 1/3*dm*prefactors_log[1]*1/fv**2*logfv\
                + 3/3*dm*prefactors_ini[7]+TTRef\

            ddPhi[itrf] = 1/fv**11*( \
                + 40/9*dm*prefactors_ini[0] \
                + 18/9*dm*prefactors_ini[1]*fv**2 \
                + 10/9*dm*prefactors_ini[2]*f\
                +  4/9*dm*prefactors_ini[3]*fv**4\
                -  3/9*dm*prefactors_log[0]*fv**5 \
                -  1/9*dm*(prefactors_log[1]+2*prefactors_ini[5])*f**2 \
                -  2/9*dm*prefactors_ini[6]*fv**7\
                +  4/9*dm*prefactors_ini[8]*f**3\
                + 10/9*dm*prefactors_ini[9]*fv**10\
                )\
                -  2/9*dm*prefactors_log[1]*1/fv**5*logfv \
                + 18/9*dm*prefactors_ini[10]\


    return Phi,dPhi,ddPhi,Amps

@njit()
def PhiSeriesInsAnsatz(Phi,dPhi,ddPhi,fs,Mt_sec,eta,chis,chia,chi,phi_ref,TTRef,NF_low,NF):
    """Ansatz for the inspiral phase.
    We call the LAL TF2 coefficients here.
    The exact values of the coefficients used are given
    as comments in the top of this file
    Defined by Equation 27 and 28 arXiv:1508.07253"""
    #Assemble PN phasing series
    prefactors_ini,prefactors_log = PhiInsPrefactorsMt(eta,Mt_sec,chis,chia,chi)
    dm = 1/(2*np.pi)
    for itrf in prange(NF_low,NF):
        floc = fs[itrf]

        fv = floc**(1/3)
        logfv = 1/3*np.log(np.pi*Mt_sec)+1/3*np.log(floc)

        Phi[itrf] = 1/fv**5*(prefactors_ini[0] \
            + prefactors_ini[1]*fv**2 \
            + prefactors_ini[2]*floc\
            + prefactors_ini[3]*fv**4\
            )\
            + prefactors_ini[5]*fv\
            + prefactors_ini[6]*fv**2\
            + (prefactors_ini[7]+TTRef/dm)*floc \
            + prefactors_ini[8]*fv**4\
            + prefactors_ini[9]*fv**5\
            + prefactors_ini[10]*fv**6\
            + prefactors_log[0]*logfv \
            + prefactors_log[1]*logfv*fv\
            + prefactors_ini[4]-phi_ref\

        if imrc.findT:
            dPhi[itrf] =1/fv**8*( \
                - 5/3*dm*prefactors_ini[0] \
                - 3/3*dm*prefactors_ini[1]*fv**2 \
                - 2/3*dm*prefactors_ini[2]*floc\
                - 1/3*dm*prefactors_ini[3]*fv**4\
                + 1/3*dm*prefactors_log[0]*fv**5 \
                + 1/3*dm*prefactors_log[1]*fv**6*logfv \
                + 1/3*dm*(prefactors_log[1]+prefactors_ini[5])*fv**6 \
                + 2/3*dm*prefactors_ini[6]*fv**7\
                )\
                + 4/3*dm*prefactors_ini[8]*fv\
                + 5/3*dm*prefactors_ini[9]*fv**2\
                + 6/3*dm*prefactors_ini[10]*floc\
                + TTRef + 3/3*dm*prefactors_ini[7]\


            ddPhi[itrf] = 1/fv**11*( \
                + 40/9*dm*prefactors_ini[0] \
                + 18/9*dm*prefactors_ini[1]*fv**2 \
                + 10/9*dm*prefactors_ini[2]*floc\
                +  4/9*dm*prefactors_ini[3]*fv**4\
                -  3/9*dm*prefactors_log[0]*fv**5 \
                -  2/9*dm*prefactors_log[1]*fv**6*logfv \
                -  1/9*dm*(prefactors_log[1]+2*prefactors_ini[5])*fv**6 \
                -  2/9*dm*prefactors_ini[6]*fv**7\
                +  4/9*dm*prefactors_ini[8]*fv**9\
                + 10/9*dm*prefactors_ini[9]*fv**10\
                ) \
                + 18/9*dm*prefactors_ini[10]\

    return Phi,dPhi,ddPhi

@njit()
def PhiSeriesIntAnsatz(Phi,dPhi,ddPhi,fs,Mt_sec,eta,chi,phi_ref,TTRef,NF_low,NF):
    """ansatz for the intermediate phase defined by Equation 16 arXiv:1508.07253"""
    #   ComputeIMRPhenDPhaseConnectionCoefficients
    #   IMRPhenDPhase
    betas = betaFits(eta,chi)
    coeff0 = betas[0]/eta*Mt_sec
    coeff1 = betas[1]/eta
    coeff2 = -1/3*betas[2]/eta/Mt_sec**3

    dm = 1/(2*np.pi)

    if imrc.findT:
        floc = fs[NF_low:NF]

        Phi[NF_low:NF] = coeff1*np.log(Mt_sec) - phi_ref\
                            + (2*np.pi*TTRef+coeff0)*floc \
                            + coeff2/floc**3 \
                            + coeff1*np.log(floc)

        dPhi[NF_low:NF] = TTRef+dm*coeff0 \
                            + 1/floc**4*(- 3*dm*coeff2 + dm*coeff1*floc**3)

        ddPhi[NF_low:NF] =    1/floc**5*(+12*dm*coeff2 - dm*coeff1*floc**3)
    else:
        floc = fs[NF_low:NF]

        Phi[NF_low:NF] = coeff1*np.log(Mt_sec) - phi_ref\
                            + (2*np.pi*TTRef+coeff0)*floc \
                            + coeff2/floc**3 \
                            + coeff1*np.log(floc)


    return Phi,dPhi,ddPhi

@njit()
def PhiSeriesMRDAnsatz(Phi,dPhi,ddPhi,fs,Mt_sec,MfRD,MfDM,eta,chi,phi_ref,TTRef,NF_low,NF):
    """Ansatz for the merger-ringdown phase Equation 14 arXiv:1508.07253"""

    alphas = alphaFits(eta,chi)
    coeff0 = alphas[0]/eta*Mt_sec
    coeff1 = -alphas[1]/eta/Mt_sec
    coeff2 = 4/3*alphas[2]/eta*Mt_sec**(3/4)
    coeff3 = alphas[3]/eta

    dm = 1/(2*np.pi)
    fDM = MfDM/Mt_sec
    fRD = MfRD/Mt_sec

    #numba cannot fuse loops across the conditional so need to write everything that needs to be fused twice
    if imrc.findT:
        floc = fs[NF_low:NF]

        fq = np.sqrt(np.sqrt(floc**3))*floc
        fadj = (floc-alphas[4]*fRD)/fDM
        Phi[NF_low:NF] = -phi_ref \
                            + (2*np.pi*TTRef+coeff0)*floc \
                            + 1/floc * (coeff1 + coeff2*fq) \
                            + coeff3*np.arctan(fadj)

        dPhi[NF_low:NF] = TTRef + dm*coeff0 \
                            +1/floc**2*(- dm*coeff1 + 3/4*dm*coeff2*fq) \
                            + dm*coeff3/fDM*(1/(1+fadj**2))
        ddPhi[NF_low:NF] =  + 1/floc**3*(2*dm*coeff1 - 3/16*dm*coeff2*fq) \
                            - 2*dm*coeff3/fDM**2*fadj*(1/(1+fadj**2))**2
    else:
        floc = fs[NF_low:NF]

        fq = np.sqrt(np.sqrt(floc**3))*floc
        fadj = (floc-alphas[4]*fRD)/fDM
        Phi[NF_low:NF] = -phi_ref \
                            + (2*np.pi*TTRef+coeff0)*floc \
                            + 1/floc * (coeff1 + coeff2*fq) \
                            + coeff3*np.arctan(fadj)

    return Phi,dPhi,ddPhi


################/ Phase: glueing function ################
@njit()
def IMRPhenDPhaseFI(Phis,times,timeps,fs,Mt_sec,eta,chis,chia,NF,MfRef_in,phi0):
    """This function computes the IMR phase given phenom coefficients.
    Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    The inspiral, intermediate and merger-ringdown phase parts
    split the calculation to just 1 of 3 possible mutually exclusive ranges
    Mfs must be sorted
    modified to anchor frequencies to FI at t=0"""
    chi = chiPN(eta,chis,chia)
    finspin = FinalSpin0815(eta, chis, chia)
    fRD,fDM = fringdown(eta,chis,chia,finspin)

    #   Transition frequencies
    #   Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    MfMRDJoinPhi = fRD/2
    MfMRDJoinAmp = fmaxCalc(fRD,fDM,eta,chi)

    # Compute coefficients to make phase C^1 continuous (phase and first derivative)
    C1Int,C2Int,C1MRD,C2MRD = ComputeIMRPhenDPhaseConnectionCoefficients(fRD,fDM,eta,chis,chia,chi,MfMRDJoinPhi)

    #time shift so that peak amplitude is approximately at t=0
    #For details see https:#www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timPD_EDOMain
    #t0 = DPhiMRD(fMRDJoinAmp,fRD,fDM,eta,chi)

    # NOTE: opposite Fourier convention with respect to PhenomD - to ensure 22 mode has power for positive f
    if fs[-1]>imrc.f_CUT/Mt_sec:
        itrFCut = np.searchsorted(fs,imrc.f_CUT/Mt_sec,side='right')
    else:
        itrFCut = NF

    # NOTE: previously MfRef=0 was by default MfRef=fmin, now MfRef defaults to MfmaxCalc (fpeak in the paper)
    # If fpeak is outside of the frequency range, take the last frequency
    if MfRef_in == 0.:
        MfRef = min(MfMRDJoinAmp, Mt_sec*fs[itrFCut-1])
    else:
        MfRef = MfRef_in

    if MfRef<imrc.PHI_fJoin_INS:
        TTRef =  -DPhiInsAnsatzInt(MfRef,eta,chis,chia,chi)
        phifRef = PhiInsAnsatzInt(MfRef,eta,chis,chia,chi)
    elif MfRef<MfMRDJoinPhi:
        TTRef =  -DPhiIntAnsatz(MfRef,eta,chi)-C2Int
        phifRef = PhiIntAnsatz(MfRef,eta,chi)+C1Int+C2Int*MfRef
    else:
        TTRef = -DPhiMRD(MfRef,fRD,fDM,eta,chi)-C2MRD
        phifRef = PhiMRDAnsatzInt(MfRef,fRD,fDM,eta,chi)+C1MRD+C2MRD*MfRef#MRD range
    #TODO check factors of pi/4 in phifref
    #make TTRef have units of seconds
    TTRef *= Mt_sec/(2*np.pi)
    phifRef += 2*np.pi*TTRef*MfRef/Mt_sec+2*phi0

    if fs[itrFCut-1]<imrc.PHI_fJoin_INS/Mt_sec:
        itrfMRDPhi = itrFCut
        itrfIntPhi = itrFCut
    elif fs[itrFCut-1]<MfMRDJoinPhi/Mt_sec:
        itrfMRDPhi = itrFCut
        itrfIntPhi = np.searchsorted(fs,imrc.PHI_fJoin_INS/Mt_sec)
    else:
        itrfMRDPhi = np.searchsorted(fs,MfMRDJoinPhi/Mt_sec)
        itrfIntPhi = np.searchsorted(fs,imrc.PHI_fJoin_INS/Mt_sec)

    #itrfMRDPhi = itrFCut
    #itrfIntPhi = itrFCut

    TTRefIns = TTRef
    TTRefInt = TTRef+C2Int*Mt_sec/(2*np.pi)
    TTRefMRD = TTRef+C2MRD*Mt_sec/(2*np.pi)

    phifRefIns = phifRef
    phifRefInt = phifRef-C1Int
    phifRefMRD = phifRef-C1MRD

    if itrfIntPhi>0:
        Phis,times,timeps = PhiSeriesInsAnsatz(Phis,times,timeps,fs,Mt_sec,eta,chis,chia,chi,phifRefIns,TTRefIns,0,itrfIntPhi) #Ins range
    if itrfIntPhi<itrfMRDPhi:
        Phis,times,timeps = PhiSeriesIntAnsatz(Phis,times,timeps,fs,Mt_sec,eta,chi,phifRefInt,TTRefInt,itrfIntPhi,itrfMRDPhi) #intermediate range
    if itrfMRDPhi<itrFCut:
        Phis,times,timeps = PhiSeriesMRDAnsatz(Phis,times,timeps,fs,Mt_sec,fRD,fDM,eta,chi,phifRefMRD,TTRefMRD,itrfMRDPhi,itrFCut) #MRD range

    Phis[itrFCut:] = 0.
    times[itrFCut:] = 0.
    timeps[itrFCut:] = 0.

    return Phis,times,timeps,TTRef,MfRef,itrFCut

#@njit()
def IMRPhenDAmplitudeFI(Amps,fs,Mt_sec,eta,chis,chia,NF,amp_mult=1.):
    """This function computes the IMR amplitude given phenom coefficients.
    Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    The inspiral, intermediate and merger-ringdown amplitude parts"""

    #  # Transition frequencies
    chi = chiPN(eta,chis,chia)
    finspin = FinalSpin0815(eta, chis, chia)
    fRD,fDM = fringdown(eta,chis,chia,finspin)

    MfMRDJoinAmp = fmaxCalc(fRD,fDM,eta,chi)

    if fs[-1]>imrc.f_CUT/Mt_sec:
        itrFCut = np.searchsorted(fs,imrc.f_CUT/Mt_sec,side='right')
    else:
        itrFCut = NF

#    itrfMRDAmp = itrFCut
#    itrfIntAmp = itrFCut
    if fs[itrFCut-1]<imrc.AMP_fJoin_INS/Mt_sec:
        itrfMRDAmp = itrFCut
        itrfIntAmp = itrFCut
    elif fs[itrFCut-1]<MfMRDJoinAmp/Mt_sec:
        itrfMRDAmp = itrFCut
        itrfIntAmp = np.searchsorted(fs,imrc.AMP_fJoin_INS/Mt_sec)
    else:
        itrfMRDAmp = np.searchsorted(fs,MfMRDJoinAmp/Mt_sec)
        itrfIntAmp = np.searchsorted(fs,imrc.AMP_fJoin_INS/Mt_sec)

    #   split the calculation to just 1 of 3 possible mutually exclusive ranges
    amp0 = amp_mult*amp0Func(eta)
    if itrfIntAmp>0:
        Amps = AmpInsAnsatzInplace(Amps,fs,Mt_sec,eta,chis,chia,chi,amp0,0,itrfIntAmp) #Inspiral range
    if itrfIntAmp<itrfMRDAmp:
        Amps = AmpIntAnsatzInplace(Amps,fs,Mt_sec,fRD,fDM,eta,chis,chia,chi,amp0,itrfIntAmp,itrfMRDAmp) #Intermediate range
    if itrfMRDAmp<itrFCut:
        Amps = AmpMRDAnsatzInplace(Amps,fs,Mt_sec,fRD,fDM,eta,chi,amp0,itrfMRDAmp,itrFCut) # MRD range
    Amps[itrFCut:] = 0.

    return Amps

@njit()
def IMRPhenDAmpPhaseFI_get_TTRef(Mt_sec,eta,chis,chia,MfRef_in,imr_default_t=False,t_offset=0.):
    """get only TTRef given input FI at MfRef_in if imr_default_t is true, use the phasing convention from IMRPhenomD,
    otherwise try to set MfRef_in=Mf at t=0"""

    chi = chiPN(eta,chis,chia)
    finspin = FinalSpin0815(eta, chis, chia)
    fRD,fDM = fringdown(eta,chis,chia,finspin)
    #print('frd,fdm',fRD,fDM)

    #   Transition frequencies
    #   Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    MfMRDJoinPhi = fRD/2
    MfMRDJoinAmp = fmaxCalc(fRD,fDM,eta,chi)

    # NOTE: previously MfRef=0 was by default MfRef=fmin, now MfRef defaults to MfmaxCalc (fpeak in the paper)
    # If fpeak is outside of the frequency range, take the last frequency
    if MfRef_in == 0.:
        MfRef = MfMRDJoinAmp
    else:
        MfRef = MfRef_in

    # Compute coefficients to make phase C^1 continuous (phase and first derivative)
    _,C2Int,_,C2MRD = ComputeIMRPhenDPhaseConnectionCoefficients(fRD,fDM,eta,chis,chia,chi,MfMRDJoinPhi)

    if imr_default_t:
        dPhifRef = -DPhiMRD(MfMRDJoinAmp,fRD,fDM,eta,chi)
    else:
        #TODO safe to pic a default here?
        if MfRef<imrc.PHI_fJoin_INS:
            dPhifRef =  -DPhiInsAnsatzInt(MfRef,eta,chis,chia,chi)
        elif MfRef<MfMRDJoinPhi:
            dPhifRef =  -DPhiIntAnsatz(MfRef,eta,chi)-C2Int
        else:
            dPhifRef = -DPhiMRD(MfRef,fRD,fDM,eta,chi)-C2MRD

    dm = Mt_sec/(2*np.pi)
    TTRef = dPhifRef*dm+t_offset

    return TTRef

@njit()
def IMRPhenDAmpPhaseFI(Phis,times,timeps,Amps,fs,Mt_sec,eta,chis,chia,NF,MfRef_in,phi0,amp_mult,imr_default_t=False,t_offset=0.):
    """get both amplitude and phase in place at the same time given input FI at MfRef_in if imr_default_t is true, use the phasing convention from IMRPhenomD,
    otherwise try to set MfRef_in=Mf at t=0"""

    chi = chiPN(eta,chis,chia)
    finspin = FinalSpin0815(eta, chis, chia)
    fRD,fDM = fringdown(eta,chis,chia,finspin)
    #print('frd,fdm',fRD,fDM)

    #   Transition frequencies
    #   Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    MfMRDJoinPhi = fRD/2
    MfMRDJoinAmp = fmaxCalc(fRD,fDM,eta,chi)

    # NOTE: previously MfRef=0 was by default MfRef=fmin, now MfRef defaults to MfmaxCalc (fpeak in the paper)
    # If fpeak is outside of the frequency range, take the last frequency
    if MfRef_in == 0.:
        MfRef = MfMRDJoinAmp
    else:
        MfRef = MfRef_in

    # Compute coefficients to make phase C^1 continuous (phase and first derivative)
    C1Int,C2Int,C1MRD,C2MRD = ComputeIMRPhenDPhaseConnectionCoefficients(fRD,fDM,eta,chis,chia,chi,MfMRDJoinPhi)

    if imr_default_t:
        dPhifRef = -DPhiMRD(MfMRDJoinAmp,fRD,fDM,eta,chi)
    else:
        #TODO safe to pic a default here?
        if MfRef<imrc.PHI_fJoin_INS:
            dPhifRef =  -DPhiInsAnsatzInt(MfRef,eta,chis,chia,chi)
        elif MfRef<MfMRDJoinPhi:
            dPhifRef =  -DPhiIntAnsatz(MfRef,eta,chi)-C2Int
        else:
            dPhifRef = -DPhiMRD(MfRef,fRD,fDM,eta,chi)-C2MRD

    dm = Mt_sec/(2*np.pi)
    TTRef = dPhifRef*dm+t_offset

    if MfRef<imrc.PHI_fJoin_INS:
        phifRef = PhiInsAnsatzInt(MfRef,eta,chis,chia,chi)
    elif MfRef<MfMRDJoinPhi:
        phifRef = PhiIntAnsatz(MfRef,eta,chi)+C1Int+C2Int*MfRef
    else:
        phifRef = PhiMRDAnsatzInt(MfRef,fRD,fDM,eta,chi)+C1MRD+C2MRD*MfRef#MRD range

    #TODO watch out for this factor of 2 on phi0
    #TODO check factors of pi/4 in phifref
    phifRef = phifRef + dPhifRef*MfRef + t_offset/dm*MfRef + 2*phi0

    Phis,times,timeps,Amps,itrFCut = IMRPhenDAmpPhase_tc(Phis,times,timeps,Amps,fs,Mt_sec,eta,chis,chia,NF,TTRef,phifRef,amp_mult)
    return Phis,times,timeps,Amps,TTRef,MfRef,itrFCut

@njit()
def IMRPhenDAmpPhase_tc(Phis,times,timeps,Amps,fs,Mt_sec,eta,chis,chia,NF,TTRef,phifRef,amp_mult):
    """get both amplitude and phase in place at the same time given input TTRef"""
    #TODO reabsorb this now redundant function
    chi = chiPN(eta,chis,chia)
    finspin = FinalSpin0815(eta, chis, chia)
    fRD,fDM = fringdown(eta,chis,chia,finspin)


    #   Transition frequencies
    #   Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    MfMRDJoinPhi = fRD/2
    MfMRDJoinAmp = fmaxCalc(fRD,fDM,eta,chi)

    MfLast = fs[NF-1]*Mt_sec
    if MfLast>imrc.f_CUT:
        itrFCut = np.searchsorted(fs,imrc.f_CUT/Mt_sec,side='right')
        MfLast = fs[itrFCut-1]*Mt_sec
    else:
        itrFCut = NF

    #TODO duplicate logic
    if MfLast<imrc.AMP_fJoin_INS:
        itrfMRDAmp = itrFCut
        itrfIntAmp = itrFCut
    elif MfLast<MfMRDJoinAmp:
        itrfMRDAmp = itrFCut
        itrfIntAmp = np.searchsorted(fs,imrc.AMP_fJoin_INS/Mt_sec)
    else:
        itrfIntAmp = np.searchsorted(fs,imrc.AMP_fJoin_INS/Mt_sec)
        itrfMRDAmp = np.searchsorted(fs,MfMRDJoinAmp/Mt_sec)

    if MfLast<imrc.PHI_fJoin_INS:
        itrfMRDPhi = itrFCut
        itrfIntPhi = itrFCut
    elif MfLast<MfMRDJoinPhi:
        itrfMRDPhi = itrFCut
        itrfIntPhi = np.searchsorted(fs,imrc.PHI_fJoin_INS/Mt_sec)
    else:
        itrfMRDPhi = np.searchsorted(fs,MfMRDJoinPhi/Mt_sec)
        itrfIntPhi = np.searchsorted(fs,imrc.PHI_fJoin_INS/Mt_sec)

    itrfIntMax = max(itrfIntPhi,itrfIntAmp)

    #time shift so that reference frequency is at t=0
    #For details see https:#www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timPD_EDOMain
    # NOTE: opposite Fourier convention with respect to PhenomD - to ensure 22 mode has power for positive f

    # Compute coefficients to make phase C^1 continuous (phase and first derivative)
    C1Int,C2Int,C1MRD,C2MRD = ComputeIMRPhenDPhaseConnectionCoefficients(fRD,fDM,eta,chis,chia,chi,MfMRDJoinPhi)

    dm = Mt_sec/(2*np.pi)

    TTRefIns = TTRef
    TTRefInt = TTRefIns+C2Int*dm
    TTRefMRD = TTRefIns+C2MRD*dm

    phifRefIns = phifRef
    phifRefInt = phifRef-C1Int
    phifRefMRD = phifRef-C1MRD

    amp0 = amp_mult*amp0Func(eta)

    #Technically, this wastes a small amount of operations filling values that will be overwritten by the intermediate.
    #In practice the combined method is so much faster that it justifies the wasted computation
    #and it would unnecessarily increase code complexity to avoid it.
    if itrfIntMax>0:
        Phis,times,timeps,Amps = AmpPhaseSeriesInsAnsatz(Phis,times,timeps,Amps,fs,Mt_sec,eta,chis,chia,chi,phifRefIns,TTRefIns,amp0,0,itrfIntMax) #Ins range

    #   split the calculation to just 1 of 3 possible mutually exclusive ranges
    if itrfIntAmp<itrfMRDAmp:
        Amps = AmpIntAnsatzInplace(Amps,fs,Mt_sec,fRD,fDM,eta,chis,chia,chi,amp0,itrfIntAmp,itrfMRDAmp) #Intermediate range
    if itrfMRDAmp<itrFCut:
        Amps = AmpMRDAnsatzInplace(Amps,fs,Mt_sec,fRD,fDM,eta,chi,amp0,itrfMRDAmp,itrFCut) # MRD range

    if itrfIntPhi<itrfMRDPhi:
        Phis,times,timeps = PhiSeriesIntAnsatz(Phis,times,timeps,fs,Mt_sec,eta,chi,phifRefInt,TTRefInt,itrfIntPhi,itrfMRDPhi) #intermediate range
    if itrfMRDPhi<itrFCut:
        Phis,times,timeps = PhiSeriesMRDAnsatz(Phis,times,timeps,fs,Mt_sec,fRD,fDM,eta,chi,phifRefMRD,TTRefMRD,itrfMRDPhi,itrFCut) #MRD range

    if itrFCut<NF:
        Amps[itrFCut:] = 0.
        Phis[itrFCut:] = 0.
        times[itrFCut:] = 0.
        timeps[itrFCut:] = 0.

    return Phis,times,timeps,Amps,itrFCut
