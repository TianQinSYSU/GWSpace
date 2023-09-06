"""Python implementation of IMRPhenomD by Matthew Digman (C) 2021"""
#
 # Copyright (C) 2015 Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
 #
 #  This program is free software; you can redistribute it and/or modify
 #  it under the terms of the GNU General Public License as published by
 #  the Free Software Foundation; either version 2 of the License, or
 #  (at your option) any later version.
 #
 #  This program is distributed in the hope that it will be useful,
 #  but WITHOUT ANY WARRANTY; without even the implied warranty of
 #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 #  GNU General Public License for more details.
 #
 #  You should have received a copy of the GNU General Public License
 #  along with with program; see the file COPYING. If not, write to the
 #  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 #  MA  02111-1307  USA
 #/

# LAL independent code (C) 2017 Michael Puerrer
import numpy as np
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import FinalSpin0815,DPhiMRD
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import IMRPhenDPhase,IMRPhenDAmplitude,NextPow2,fmaxCalc
from Waveforms.PyIMRPhenomD.IMRPhenomD_internals import AmpPhaseFDWaveform,COMPLEX16FrequencySeries
from Waveforms.PyIMRPhenomD.IMRPhenomD_deriv_internals import IMRPhenDAmpPhaseFI
import Waveforms.PyIMRPhenomD.IMRPhenomD_const as imrc

#*
 # @addtogroup LALSimIMRPhenom_c
 # @{
 #
 # @name Routines for IMR Phenomenological Model "D"
 # @{
 #
 # @author Michael Puerrer, Sebastian Khan, Frank Ohme
 #
 # @brief C code for IMRPhenomD phenomenological waveform model.
 #
 # This is an aligned-spin frequency domain model.
 # See Husa et al \cite Husa:2015iqa, and Khan et al \cite Khan:2015jqa
 # for details. Any studies that use this waveform model should include
 # a reference to both of these papers.
 #
 # @note The model was calibrated to mass-ratios [1:1,1:4,1:8,1:18].
 # * Along the mass-ratio 1:1 line it was calibrated to spins  [-0.95, +0.98].
 # * Along the mass-ratio 1:4 line it was calibrated to spins  [-0.75, +0.75].
 # * Along the mass-ratio 1:8 line it was calibrated to spins  [-0.85, +0.85].
 # * Along the mass-ratio 1:18 line it was calibrated to spins [-0.8, +0.4].
 # The calibration points will be given in forthcoming papers.
 #
 # @attention The model is usable outside this parameter range,
 # and in tests to date gives sensible physical results,
 # but conclusive statements on the physical fidelity of
 # the model for these parameters await comparisons against further
 # numerical-relativity simulations. For more information, see the review wiki
 # under https:#www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview
 #/

def IMRPhenomDGenerateFD(phi0,fRef_in,deltaF,m1_SI,m2_SI,chi1,chi2,f_min,f_max,distance):
    """Driver routine to compute the spin-aligned, inspiral-merger-ringdown
    phenomenological waveform IMRPhenomD in the frequency domain.

    Reference:
    - Waveform: Eq. 35 and 36 in arXiv:1508.07253
    - Coefficients: Eq. 31 and Table V in arXiv:1508.07253

    All input parameters should be in SI units. Angles should be in radians."""
    # external: SI; internal: solar masses
    m1 = m1_SI/imrc.MSUN_SI
    m2 = m2_SI/imrc.MSUN_SI

# check inputs for sanity
    if fRef_in < 0.:
        raise ValueError("fRef_in must be positive (or 0 for 'ignore')")
    if deltaF <= 0.:
        raise ValueError("deltaF must be positive")
    if m1 <= 0.:
        raise ValueError("m1 must be positive")
    if m2 <= 0.:
        raise ValueError("m2 must be positive")
    if f_min <= 0.:
        raise ValueError("f_min must be positive")
    if f_max < 0.:
        raise ValueError("f_max must be greater than 0")
    if distance <= 0.:
        raise ValueError("distance must be positive")
    if m1 > m2:
        q = m1/m2
    else:
        q = m2/m1

    assert q>1.

    if not (-1.<=chi1<=1. and -1.<=chi2<=1.):
        raise ValueError("Spins outside the range [-1,1] are not supported")

    # NOTE: we changed the prescription, now fRef defaults to fmaxCalc (fpeak in the paper)
    # if no reference frequency given, set it to the starting GW frequency

    Mt_sec = (m1+m2)*imrc.MTSUN_SI # Conversion factor Hz -> dimensionless frequency
    fCut = imrc.f_CUT/Mt_sec # convert Mf -> Hz
    # Somewhat arbitrary end point for the waveform.
    # Chosen so that the end of the waveform is well after the ringdown.
    if fCut <= f_min:
        print("(fCut = %g Hz) <= f_min = %g"%(fCut, f_min))

    # default f_max to Cut
    f_max_prime = f_max
    if f_max:
        f_max_prime = f_max
    else:
        f_max_prime = fCut

    if f_max_prime>fCut:
        f_max_prime = fCut

    htilde = IMRPhenomDGenerateFD_internal(phi0, fRef_in, deltaF,m1, m2, chi1, chi2,f_min, f_max_prime, distance)

    if f_max_prime < f_max:
        # The user has requested a higher f_max than Mf=fCut.
        # Resize the frequency series to fill with zeros beyond the cutoff frequency.
        n = htilde.length
        n_full = NextPow2(f_max / deltaF) + 1 # we actually want to have the length be a power of 2 + 1
        print("Failed to resize waveform COMPLEX16FrequencySeries of length %5d (for internal fCut=%f) to new length %5d (for user-requested f_max=%f)."%(n, fCut, n_full, f_max))

def IMRPhenomDGenerateh22FDAmpPhase(h22,freq,phi0,fRef_in,m1_SI,m2_SI,chi1,chi2,distance):
    """SM: similar to IMRPhenomDGenerateFD, but generates h22 FD amplitude and phase on a given set of frequencies"""
    m1 = m1_SI/imrc.MSUN_SI
    m2 = m2_SI/imrc.MSUN_SI

    f_min = freq[0]
    f_max = freq[-1]

    # check inputs for sanity
    #if np.all(freq==0.):
    #    raise ValueError("freq is null")
    if fRef_in<0.0:
        raise ValueError("fRef_in must be positive (or 0 for 'ignore')")
    if m1 <= 0.0:
        raise ValueError("m1 must be positive")
    if m2 <= 0.0:
        raise ValueError("m2 must be positive")
    if f_min <= 0.0:
        raise ValueError("f_min must be positive")
    if f_max < 0.0:
        raise ValueError("f_max must be greater than 0")
    if distance <= 0.0:
        raise ValueError("distance must be positive")

    if m1>m2:
        q = m1/m2
    else:
        q = m2/m1
    assert q>1.

#if (q > MAX_ALLOWED_MASS_RATIO) PRINT_WARNING("Warning: The model is not supported for high mass ratio, see MAX_ALLOWED_MASS_RATIO\n");

    if not (-1.<chi1<=1. and -1.<chi2<=1.):
        raise ValueError("Spins outside the range [-1,1] are not supported")

    # NOTE: we changed the prescription, now fRef defaults to fmaxCalc (fpeak in the paper)
    # if no reference frequency given, set it to the starting GW frequency
    # double fRef = (fRef_in == 0.0) ? f_min : fRef_in;

    Mt_sec = (m1+m2)*imrc.MTSUN_SI # Conversion factor Hz -> dimensionless frequency
    fCut = imrc.f_CUT/Mt_sec # convert Mf -> Hz
    # Somewhat arbitrary end point for the waveform.
    # Chosen so that the end of the waveform is well after the ringdown.
    if fCut<=f_min:
        print("(fCut = %g Hz) <= f_min = %g"%(fCut,f_min))
    # Check that at least the first of the output frequencies is strictly positive - note that we don't check for monotonicity
    if f_min <= 0:
        print("(f_min = %g Hz) <= 0"%(f_min))
    h22 = IMRPhenomDGenerateh22FDAmpPhase_internal(h22,freq, phi0, fRef_in, m1, m2, chi1, chi2, distance)
    return h22

def IMRPhenomDGenerateFD_internal(phi0,fRef_in,deltaF,m1_in,m2_in,chi1_in,chi2_in,f_min,f_max,distance):
    """The following private function generates IMRPhenomD frequency-domain waveforms
    given coefficients"""
    # LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; # = {0, 0}
    ligotimegps_zero = 0

    if m1_in>m2_in:
        chi1 = chi1_in
        chi2 = chi2_in
        m1   = m1_in
        m2   = m2_in
    else: # swap spins and masses
        chi1 = chi2_in
        chi2 = chi1_in
        m1   = m2_in
        m2   = m1_in

    Mt = m1 + m2
    eta = m1 * m2 /Mt**2

    if not 0<=eta<=0.25:
        raise ValueError("Unphysical eta. Must be between 0. and 0.25")

    Mt_sec = Mt * imrc.MTSUN_SI

    # Compute the amplitude pre-factor
    amp0 = 2*np.sqrt(5./(64.*np.pi))*Mt**2*imrc.MRSUN_SI*imrc.MTSUN_SI/distance

    # Coalesce at t=0
    # shift by overall length in time
    ligotimegps_zero += -1./deltaF

    # Allocate htilde
    nf = NextPow2(f_max / deltaF) + 1
    htilde = COMPLEX16FrequencySeries(ligotimegps_zero, 0.0, deltaF, nf)

    # range that will have actual non-zero waveform values generated
    ind_min = np.int64(f_min/deltaF)
    ind_max = np.int64(f_max/deltaF)
    if not ind_min<=ind_max<=nf:
        raise ValueError("minimum freq index %5d and maximum freq index %5d do not fulfill 0<=ind_min<=ind_max<=htilde->data>length=%5d."%(ind_min, ind_max,nf))

    # Calculate phenomenological parameters
    chis = (chi1+chi2)/2
    chia = (chi1-chi2)/2
    finspin = FinalSpin0815(eta, chis, chia) #FinalSpin0815 - 0815 is like a version number

    if finspin < imrc.MIN_FINAL_SPIN:
        print("Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here."%(finspin))

    # Now generate the waveform
    Mfs = Mt_sec*deltaF*np.arange(ind_min,ind_max) # geometric frequency
    phis,times,t0,MfRef,itrFCut = IMRPhenDPhase(Mfs[ind_min:ind_max],Mt_sec,eta,chis,chia,ind_max-ind_min,fRef_in,phi0)
    amps = IMRPhenDAmplitude(Mfs[ind_min:ind_max],eta,chis,chia,ind_max-ind_min,amp_mult=amp0)
    htilde.data[:ind_max-ind_min] =  amps[:ind_max-ind_min]*np.exp(-1j*phis[:ind_max-ind_min])

    for i in range(ind_min,ind_max):
        phi = phis[i-ind_min]
        amp = amps[i-ind_min]
        htilde.data[i] = amp*np.exp(-1j*phi)
    return htilde

########################
# END OF REVIEWED CODE ############
########################

def IMRPhenomDGenerateh22FDAmpPhase_internal(h22,freq,phi0,fRef_in,m1_in,m2_in,chi1_in,chi2_in,distance):
    """SM: similar to IMRPhenomDGenerateFD_internal, but generates h22 FD amplitude and phase on a given set of frequencies"""
    nf = freq.size
    if m1_in>m2_in:
        chi1 = chi1_in
        chi2 = chi2_in
        m1   = m1_in
        m2   = m2_in
    else: # swap spins and masses
        chi1 = chi2_in
        chi2 = chi1_in
        m1   = m2_in
        m2   = m1_in

    Mt = m1+m2
    eta = m1*m2/Mt**2

    if not 0.<=eta<=0.25:
        raise ValueError("Unphysical eta. Must be between 0. and 0.25")

    Mt_sec = Mt*imrc.MTSUN_SI

    # Compute the amplitude pre-factor
    # NOTE: we will output the amplitude of the 22 mode - so we remove the factor 2. * sqrt(5. / (64.*PI)), which is part of the Y22 spherical harmonic factor
    amp0 = Mt**2*imrc.MRSUN_SI*imrc.MTSUN_SI/distance

    # Max frequency covered by PhenomD
    fCut = imrc.f_CUT/Mt_sec # convert Mf -> Hz

    # Calculate phenomenological parameters
    chis = (chi1+chi2)/2
    chia = (chi1-chi2)/2
    finspin = FinalSpin0815(eta, chis, chia) #FinalSpin0815 - 0815 is like a version number

    if finspin < imrc.MIN_FINAL_SPIN:
        print("Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here."%(finspin))

    # Now generate the waveform on the frequencies given by freq
    f = freq

    #Mfs = Mt_sec*f #geometric frequency

    # for frequencies exceeding the maximal frequency covered by PhenomD, put 0 amplitude and phase
    #phase,time,t0,MfRef,itrFCut = IMRPhenDPhase(Mfs,Mt,eta,chis,chia,nf,fRef_in,phi0)
    #amp = IMRPhenDAmplitude(Mfs,eta,chis,chia,nf,amp_mult=amp0)
    h22.phase,h22.time,h22.timep,h22.amp,h22.t0,MfRef,itrFCut = IMRPhenDAmpPhaseFI(h22.phase,h22.time,h22.timep,h22.amp,freq,Mt_sec,eta,chis,chia,nf,fRef_in,phi0,amp0,True)
    h22.fRef = MfRef/Mt_sec

    #for itrf in range(0,nf):
    #    print("%5d %+.8e %+.8e %+.8e %+.8e"%(itrf,freq[itrf],phase[itrf],amp[itrf],time[itrf]))

    return h22
