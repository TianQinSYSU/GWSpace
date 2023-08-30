/*
 * Copyright (C) 2015 Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

// LAL independent code (C) 2017 Michael Puerrer

//#include <math.h>

#include "IMRPhenomD.h"
#include "IMRPhenomD_internals.c"

#ifndef _OPENMP
#define omp ignore
#endif


UsefulPowers powers_of_pi;	// declared in LALSimIMRPhenomD_internals.c

/*
 * private function prototypes; all internal functions use solar masses.
 *
 */

static int IMRPhenomDGenerateFD_internal(
    COMPLEX16FrequencySeries **htilde, /**< [out] FD waveform */
    const double phi0,                  /**< phase at fRef */
    const double fRef,                  /**< reference frequency [Hz] */
    const double deltaF,                /**< frequency resolution */
    const double m1,                    /**< mass of companion 1 [solar masses] */
    const double m2,                    /**< mass of companion 2 [solar masses] */
    const double chi1,                  /**< aligned-spin of companion 1 */
    const double chi2,                  /**< aligned-spin of companion 2 */
    const double f_min,                 /**< start frequency */
    const double f_max,                 /**< end frequency */
    const double distance               /**< distance to source (m) */
);
// SM: similar to IMRPhenomDGenerateFD_internal, but generates h22 FD amplitude and phase on a given set of frequencies
int IMRPhenomDGenerateh22FDAmpPhase_internal(
    AmpPhaseFDWaveform** h22,           /**< [out] FD waveform */
    real_vector** tf,                   /**< [out] tf from analytic derivative of the phase */
    double* fpeak,                      /**< [out] Approximate peak frequency (Hz) */
    double* tpeak,                      /**< [out] tf at peak frequency (s) */
    double* phipeak,                    /**< [out] phase 22 at peak frequency */
    double* fstart,                      /**< [out] Starting frequency (Hz) */
    double* tstart,                      /**< [out] tf at starting frequency (s) */
    double* phistart,                    /**< [out] phase 22 at starting frequency */
    real_vector* freq,                  /**< Input: frequencies (Hz) on which to evaluate h22 FD - will be copied in the output AmpPhaseFDWaveform. Frequencies exceeding max freq covered by PhenomD will be given 0 amplitude and phase. */
    const double m1_in,                 /**< Mass of companion 1 (solar masses) */
    const double m2_in,                 /**< Mass of companion 2 (solar masses) */
    const double chi1_in,               /**< Aligned-spin parameter of companion 1 */
    const double chi2_in,               /**< Aligned-spin parameter of companion 2 */
    const double distance,              /**< Distance of source (Mpc) */
    const double tRef,                  /**< Time at fRef_for_tRef (s) */
    const double phiRef,                /**< Orbital phase at fRef_for_phiRef (rad) */
    const double fRef_for_tRef_in,      /**< Ref. frequency (Hz) for tRef */
    const double fRef_for_phiRef_in,    /**< Ref. frequency (Hz) for phiRef */
    const int force_phiRef_fRef,        /**< Flag to force phiRef at fRef after adjusting tRef */
    const double Deltat,                /**< Time shift (s) applied a posteriori */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
);

/**
 * @addtogroup LALSimIMRPhenom_c
 * @{
 *
 * @name Routines for IMR Phenomenological Model "D"
 * @{
 *
 * @author Michael Puerrer, Sebastian Khan, Frank Ohme
 *
 * @brief C code for IMRPhenomD phenomenological waveform model.
 *
 * This is an aligned-spin frequency domain model.
 * See Husa et al \cite Husa:2015iqa, and Khan et al \cite Khan:2015jqa
 * for details. Any studies that use this waveform model should include
 * a reference to both of these papers.
 *
 * @note The model was calibrated to mass-ratios [1:1,1:4,1:8,1:18].
 * * Along the mass-ratio 1:1 line it was calibrated to spins  [-0.95, +0.98].
 * * Along the mass-ratio 1:4 line it was calibrated to spins  [-0.75, +0.75].
 * * Along the mass-ratio 1:8 line it was calibrated to spins  [-0.85, +0.85].
 * * Along the mass-ratio 1:18 line it was calibrated to spins [-0.8, +0.4].
 * The calibration points will be given in forthcoming papers.
 *
 * @attention The model is usable outside this parameter range,
 * and in tests to date gives sensible physical results,
 * but conclusive statements on the physical fidelity of
 * the model for these parameters await comparisons against further
 * numerical-relativity simulations. For more information, see the review wiki
 * under https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview
 */


/**
 * Driver routine to compute the spin-aligned, inspiral-merger-ringdown
 * phenomenological waveform IMRPhenomD in the frequency domain.
 *
 * Reference:
 * - Waveform: Eq. 35 and 36 in arXiv:1508.07253
 * - Coefficients: Eq. 31 and Table V in arXiv:1508.07253
 *
 *  All input parameters should be in SI units. Angles should be in radians.
 */
int IMRPhenomDGenerateFD(
    COMPLEX16FrequencySeries **htilde, /**< [out] FD waveform */
    const double phi0,                  /**< Orbital phase at fRef (rad) */
    const double fRef_in,               /**< reference frequency (Hz) */
    const double deltaF,                /**< Sampling frequency (Hz) */
    const double m1_SI,                 /**< Mass of companion 1 (kg) */
    const double m2_SI,                 /**< Mass of companion 2 (kg) */
    const double chi1,                  /**< Aligned-spin parameter of companion 1 */
    const double chi2,                  /**< Aligned-spin parameter of companion 2 */
    const double f_min,                 /**< Starting GW frequency (Hz) */
    const double f_max,                 /**< End frequency; 0 defaults to Mf = \ref f_CUT */
    const double distance               /**< Distance of source (m) */
) {
  /* external: SI; internal: solar masses */
  const double m1 = m1_SI / MSUN_SI;
  const double m2 = m2_SI / MSUN_SI;

  char errstr[200];

  /* check inputs for sanity */
  CHECK(0 != htilde, ERROR_EFAULT, "htilde is null");
  if (*htilde) ERROR(ERROR_EFAULT, "");
  if (fRef_in < 0) ERROR(ERROR_EDOM, "fRef_in must be positive (or 0 for 'ignore')\n");
  if (deltaF <= 0) ERROR(ERROR_EDOM, "deltaF must be positive\n");
  if (m1 <= 0) ERROR(ERROR_EDOM, "m1 must be positive\n");
  if (m2 <= 0) ERROR(ERROR_EDOM, "m2 must be positive\n");
  if (f_min <= 0) ERROR(ERROR_EDOM, "f_min must be positive\n");
  if (f_max < 0) ERROR(ERROR_EDOM, "f_max must be greater than 0\n");
  if (distance <= 0) ERROR(ERROR_EDOM, "distance must be positive\n");

  const double q = (m1 > m2) ? (m1 / m2) : (m2 / m1);

  if (q > MAX_ALLOWED_MASS_RATIO)
    WARNING("Warning: The model is not supported for high mass ratio, see MAX_ALLOWED_MASS_RATIO\n");

  if (chi1 > 1.0 || chi1 < -1.0 || chi2 > 1.0 || chi2 < -1.0)
    ERROR(ERROR_EDOM, "Spins outside the range [-1,1] are not supported\n");

  // NOTE: we changed the prescription, now fRef defaults to fmaxCalc (fpeak in the paper)
  // if no reference frequency given, set it to the starting GW frequency
  // double fRef = (fRef_in == 0.0) ? f_min : fRef_in;

  const double M_sec = (m1+m2) * MTSUN_SI; // Conversion factor Hz -> dimensionless frequency
  const double fCut = f_CUT/M_sec; // convert Mf -> Hz
  //printf("Stas m1=%g, m2=%g, Msec=%g, f_CUT=%g \n", m1, m2, M_sec, f_CUT);
  //printf("Stas f_min = %g, fCut = %g, fRef = %g\n", f_min, fCut, fRef);
  // Somewhat arbitrary end point for the waveform.
  // Chosen so that the end of the waveform is well after the ringdown.
  if (fCut <= f_min) {
      snprintf(errstr, strlen(errstr), "(fCut = %g Hz) <= f_min = %g\n", fCut, f_min);
      ERROR(ERROR_EDOM, errstr);
  }

    /* default f_max to Cut */
  double f_max_prime = f_max;
  f_max_prime = f_max ? f_max : fCut;
  f_max_prime = (f_max_prime > fCut) ? fCut : f_max_prime;
  if (f_max_prime <= f_min)
    ERROR(ERROR_EDOM, "f_max <= f_min\n");

  int status = IMRPhenomDGenerateFD_internal(htilde, phi0, fRef_in, deltaF,
                                    m1, m2, chi1, chi2,
                                    f_min, f_max_prime, distance);
  CHECK(SUCCESS == status, ERROR_EFUNC, "Failed to generate IMRPhenomD waveform.");

  //printf("Stas, htilde length = %d", (*htilde)->length);
  if (f_max_prime < f_max) {
    // The user has requested a higher f_max than Mf=fCut.
    // Resize the frequency series to fill with zeros beyond the cutoff frequency.
    size_t n = (*htilde)->length;
    size_t n_full = NextPow2(f_max / deltaF) + 1; // we actually want to have the length be a power of 2 + 1
    //*htilde = XLALResizeCOMPLEX16FrequencySeries(*htilde, 0, n_full);
    *htilde = ResizeCOMPLEX16FrequencySeries(*htilde, n_full);
    snprintf(errstr, strlen(errstr), "Failed to resize waveform COMPLEX16FrequencySeries of length %zu (for internal fCut=%f) to new length %zu (for user-requested f_max=%f).", n, fCut, n_full, f_max);
    CHECK ( *htilde, ERROR_ENOMEM, errstr);
  }

  return SUCCESS;
}

// SM: similar to IMRPhenomDGenerateFD, but generates h22 FD amplitude and phase on a given set of frequencies
int IMRPhenomDGenerateh22FDAmpPhase(
    AmpPhaseFDWaveform** h22,           /**< [out] FD waveform */
    real_vector** tf,                   /**< [out] tf from analytic derivative of the phase */
    double* fpeak,                      /**< [out] Approximate peak frequency (Hz) */
    double* tpeak,                      /**< [out] tf at peak frequency (s) */
    double* phipeak,                    /**< [out] phase 22 at peak frequency */
    double* fstart,                      /**< [out] Starting frequency (Hz) */
    double* tstart,                      /**< [out] tf at starting frequency (s) */
    double* phistart,                    /**< [out] phase 22 at starting frequency */
    real_vector* freq,                  /**< Input: frequencies (Hz) on which to evaluate h22 FD - will be copied in the output AmpPhaseFDWaveform. Frequencies exceeding max freq covered by PhenomD will be given 0 amplitude and phase. */
    const double m1,                    /**< Mass of companion 1 (solar masses) */
    const double m2,                    /**< Mass of companion 2 (solar masses) */
    const double chi1,                  /**< Aligned-spin parameter of companion 1 */
    const double chi2,                  /**< Aligned-spin parameter of companion 2 */
    const double distance,              /**< Distance of source (Mpc) */
    const double tRef,                  /**< Time at fRef_for_tRef (s) */
    const double phiRef,                /**< Orbital phase at fRef_for_phiRef (rad) */
    const double fRef_for_tRef_in,      /**< Ref. frequency (Hz) for tRef */
    const double fRef_for_phiRef_in,    /**< Ref. frequency (Hz) for phiRef */
    const int force_phiRef_fRef,        /**< Flag to force phiRef at fRef after adjusting tRef */
    const double Deltat,                /**< Time shift (s) applied a posteriori */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
) {

  size_t n = freq->size;
  double f_min = freq->data[0];
  double f_max = freq->data[n-1];

  char errstr[200];

  /* check inputs for sanity */
  CHECK(0 != freq, ERROR_EFAULT, "freq is null");
  CHECK(0 != h22, ERROR_EFAULT, "h22 is null");
  if (*h22) ERROR(ERROR_EFAULT, "");
  if (fRef_for_tRef_in < 0) ERROR(ERROR_EDOM, "fRef_in must be positive (or 0 for 'ignore')\n");
  if (fRef_for_phiRef_in < 0) ERROR(ERROR_EDOM, "fRef_in must be positive (or 0 for 'ignore')\n");
  //if (deltaF <= 0) ERROR(ERROR_EDOM, "deltaF must be positive\n");
  if (m1 <= 0) ERROR(ERROR_EDOM, "m1 must be positive\n");
  if (m2 <= 0) ERROR(ERROR_EDOM, "m2 must be positive\n");
  if (f_min <= 0) ERROR(ERROR_EDOM, "f_min must be positive\n");
  if (f_max < 0) ERROR(ERROR_EDOM, "f_max must be greater than 0\n");
  if (distance <= 0) ERROR(ERROR_EDOM, "distance must be positive\n");

  const double q = (m1 > m2) ? (m1 / m2) : (m2 / m1);

  if (q > MAX_ALLOWED_MASS_RATIO)
    WARNING("Warning: The model is not supported for high mass ratio, see MAX_ALLOWED_MASS_RATIO\n");

  if (chi1 > 1.0 || chi1 < -1.0 || chi2 > 1.0 || chi2 < -1.0)
    ERROR(ERROR_EDOM, "Spins outside the range [-1,1] are not supported\n");

  // NOTE: we changed the prescription, now fRef defaults to fmaxCalc (fpeak in the paper)
  // if no reference frequency given, set it to the starting GW frequency
  // double fRef = (fRef_in == 0.0) ? f_min : fRef_in;

  const double M_sec = (m1+m2) * MTSUN_SI; // Conversion factor Hz -> dimensionless frequency
  const double fCut = f_CUT/M_sec; // convert Mf -> Hz
  // Somewhat arbitrary end point for the waveform.
  // Chosen so that the end of the waveform is well after the ringdown.
  if (fCut <= f_min) {
      snprintf(errstr, strlen(errstr), "(fCut = %g Hz) <= f_min = %g\n", fCut, f_min);
      ERROR(ERROR_EDOM, errstr);
  }
  // Check that at least the first of the output frequencies is strictly positive - note that we don't check for monotonicity
  if (f_min <= 0) {
      snprintf(errstr, strlen(errstr), "(f_min = %g Hz) <= 0\n", f_min);
      ERROR(ERROR_EDOM, errstr);
  }

  int status = IMRPhenomDGenerateh22FDAmpPhase_internal(h22, tf, fpeak, tpeak, phipeak, fstart, tstart, phistart, freq, m1, m2, chi1, chi2, distance, tRef, phiRef, fRef_for_tRef_in, fRef_for_phiRef_in, force_phiRef_fRef, Deltat, extraparams, modgrparams);
  CHECK(SUCCESS == status, ERROR_EFUNC, "Failed to generate IMRPhenomD h22 FD amp/phase waveform.");

  return SUCCESS;
}

/** @} */

/** @} */

/* *********************************************************************************/
/* The following private function generates IMRPhenomD frequency-domain waveforms  */
/* given coefficients */
/* *********************************************************************************/

int IMRPhenomDGenerateFD_internal(
    COMPLEX16FrequencySeries **htilde, /**< [out] FD waveform */
    const double phi0,                  /**< phase at fRef */
    const double fRef_in,               /**< reference frequency [Hz] */
    const double deltaF,                /**< frequency resolution */
    const double m1_in,                 /**< mass of companion 1 [solar masses] */
    const double m2_in,                 /**< mass of companion 2 [solar masses] */
    const double chi1_in,               /**< aligned-spin of companion 1 */
    const double chi2_in,               /**< aligned-spin of companion 2 */
    const double f_min,                 /**< start frequency */
    const double f_max,                 /**< end frequency */
    const double distance               /**< distance to source (m) */
) {

  /* Modified GR parameters are not supported here */
  ModGRParams* modgrparams = NULL;

  // LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; // = {0, 0}
  long ligotimegps_zero = 0;

  char errstr[200];

  double chi1, chi2, m1, m2;
  if (m1_in>m2_in) {
     chi1 = chi1_in;
     chi2 = chi2_in;
     m1   = m1_in;
     m2   = m2_in;
  } else { // swap spins and masses
     chi1 = chi2_in;
     chi2 = chi1_in;
     m1   = m2_in;
     m2   = m1_in;
  }

  int status = init_useful_powers(&powers_of_pi, PI);
  CHECK(SUCCESS == status, ERROR_EFUNC, "Failed to initiate useful powers of pi.");

  const double M = m1 + m2;
  const double eta = m1 * m2 / (M * M);

  if (eta > 0.25 || eta < 0.0)
    ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");

  const double M_sec = M * MTSUN_SI;

  /* Compute the amplitude pre-factor */
  const double amp0 = 2. * sqrt(5. / (64.*PI)) * M * MRSUN_SI * M * MTSUN_SI / distance;

  /* Coalesce at t=0 */
  // shift by overall length in time
  //CHECK ( XLALGPSAdd(&ligotimegps_zero, -1. / deltaF), ERROR_EFUNC, "Failed to shift coalescence time to t=0, tried to apply shift of -1.0/deltaF with deltaF=%g.", deltaF);
  ligotimegps_zero += -1. / deltaF;

  /* Allocate htilde */
  size_t n = NextPow2(f_max / deltaF) + 1;

  //*htilde = XLALCreateCOMPLEX16FrequencySeries("htilde: FD waveform", &ligotimegps_zero, 0.0, deltaF, &lalStrainUnit, n);
  *htilde = CreateCOMPLEX16FrequencySeries("htilde: FD waveform", ligotimegps_zero, 0.0, deltaF, n);
  snprintf(errstr, strlen(errstr), "Failed to allocated waveform COMPLEX16FrequencySeries of length %zu for f_max=%f, deltaF=%g.", n, f_max, deltaF);
  CHECK ( *htilde, ERROR_ENOMEM, errstr);

  //memset((*htilde)->data->data, 0, n * sizeof(COMPLEX16)); // now done internally
  //XLALUnitMultiply(&((*htilde)->sampleUnits), &((*htilde)->sampleUnits), &lalSecondUnit);

  /* range that will have actual non-zero waveform values generated */
  size_t ind_min = (size_t) (f_min / deltaF);
  size_t ind_max = (size_t) (f_max / deltaF);
  snprintf(errstr, strlen(errstr), "minimum freq index %zu and maximum freq index %zu do not fulfill 0<=ind_min<=ind_max<=htilde->data>length=%zu.", ind_min, ind_max, n);
  CHECK ( (ind_max<=n) && (ind_min<=ind_max), ERROR_EDOM, errstr);

  // Calculate phenomenological parameters
  const double finspin = FinalSpin0815(eta, chi1, chi2); //FinalSpin0815 - 0815 is like a version number

  if (finspin < MIN_FINAL_SPIN) {
    snprintf(errstr, strlen(errstr),
        "Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here.", finspin);
    WARNING(errstr);
  }

  IMRPhenomDAmplitudeCoefficients *pAmp = malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1, chi2, finspin);
  if (!pAmp) ERROR(ERROR_EFUNC, "");
  // spin order LAL_SIM_INSPIRAL_SPIN_ORDER_35PN
  IMRPhenomDPhaseCoefficients *pPhi = malloc(sizeof(IMRPhenomDPhaseCoefficients));
  ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1, chi2, finspin);
  if (!pPhi) ERROR(ERROR_EFUNC, "");
  PNPhasingSeries *pn = NULL;
  TaylorF2AlignedPhasing(&pn, m1, m2, chi1, chi2);
  if (!pn) ERROR(ERROR_EFUNC, "");

  // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
  // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but

  // was not available when PhenomD was tuned.
  pn->v[6] -= (Subtract3PNSS(m1, m2, M, eta, chi1, chi2) * pn->v[0]);

  PhiInsPrefactors phi_prefactors;
  status = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_phi_ins_prefactors failed");

  // Compute coefficients to make phase C^1 continuous (phase and first derivative)
  ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors, 1., 1.);

  //time shift so that peak amplitude is approximately at t=0
  //For details see https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timERROR_EDOMain
  const double t0 = DPhiMRD(pAmp->fmaxCalc, pPhi, 1., 1.);

  AmpInsPrefactors amp_prefactors;
  status = init_amp_ins_prefactors(&amp_prefactors, pAmp);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_amp_ins_prefactors failed");

  // NOTE: previously fRef=0 was by default fRef=fmin, now fRef defaults to fmaxCalc (fpeak in the paper)
  // If fpeak is outside of the frequency range, take the last frequency
  double fRef = (fRef_in == 0.0) ? fmin(pAmp->fmaxCalc, f_max) : fRef_in;

  // incorporating fRef
  const double MfRef = M_sec * fRef;
  UsefulPowers powers_of_MfRef;
  status = init_useful_powers(&powers_of_MfRef, MfRef);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for MfRef");
  const double phifRef = IMRPhenDPhase(MfRef, pPhi, pn, &powers_of_MfRef, &phi_prefactors, 1., 1., modgrparams);

  // factor of 2 b/c phi0 is orbital phase
  const double phi_precalc = 2.*phi0 + phifRef;

  int status_in_for = SUCCESS;
  /* Now generate the waveform */
  #pragma omp parallel for
  for (size_t i = ind_min; i < ind_max; i++)
  {
    double Mf = M_sec * i * deltaF; // geometric frequency

    UsefulPowers powers_of_Mf;
    status_in_for = init_useful_powers(&powers_of_Mf, Mf);
    if (SUCCESS != status_in_for)
    {
      snprintf(errstr, strlen(errstr), "init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
      ERROR(ERROR_EFUNC, errstr);
      status = status_in_for;
    }
    else
    {
      double amp = IMRPhenDAmplitude(Mf, pAmp, &powers_of_Mf, &amp_prefactors);
      double phi = IMRPhenDPhase(Mf, pPhi, pn, &powers_of_Mf, &phi_prefactors, 1., 1., modgrparams);

      phi -= t0*(Mf-MfRef) + phi_precalc;
      ((*htilde)->data)[i] = amp0 * amp * cexp(-I * phi);
    }
  }

  free(pAmp);
  free(pPhi);
  free(pn);

  return status;
}

////////////////////////////////////////////////
// END OF REVIEWED CODE ////////////////////////
////////////////////////////////////////////////

// SM: similar to IMRPhenomDGenerateFD_internal, but generates h22 FD amplitude and phase on a given set of frequencies

int IMRPhenomDGenerateh22FDAmpPhase_internal(
    AmpPhaseFDWaveform** h22,           /**< [out] FD waveform */
    real_vector** tf,                   /**< [out] tf from analytic derivative of the phase */
    double* fpeak,                      /**< [out] Approximate peak frequency (Hz) */
    double* tpeak,                      /**< [out] tf at peak frequency (s) */
    double* phipeak,                    /**< [out] phase 22 at peak frequency */
    double* fstart,                      /**< [out] Starting frequency (Hz) */
    double* tstart,                      /**< [out] tf at starting frequency (s) */
    double* phistart,                    /**< [out] phase 22 at starting frequency */
    real_vector* freq,                  /**< Input: frequencies (Hz) on which to evaluate h22 FD - will be copied in the output AmpPhaseFDWaveform. Frequencies exceeding max freq covered by PhenomD will be given 0 amplitude and phase. */
    const double m1_in,                 /**< Mass of companion 1 (solar masses) */
    const double m2_in,                 /**< Mass of companion 2 (solar masses) */
    const double chi1_in,               /**< Aligned-spin parameter of companion 1 */
    const double chi2_in,               /**< Aligned-spin parameter of companion 2 */
    const double distance,              /**< Distance of source (Mpc) */
    const double tRef,                  /**< Time at fRef_for_tRef (s) */
    const double phiRef,                /**< Orbital phase at fRef_for_phiRef (rad) */
    const double fRef_for_tRef_in,      /**< Ref. frequency (Hz) for tRef */
    const double fRef_for_phiRef_in,    /**< Ref. frequency (Hz) for phiRef */
    const int force_phiRef_fRef,        /**< Flag to force phiRef at fRef after adjusting tRef */
    const double Deltat,                /**< Time shift (s) applied a posteriori */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
) {

  if (modgrparams != NULL) printf("%.16e\n", modgrparams->alpha);

  size_t n = freq->size;

  char errstr[200];

  double chi1, chi2, m1, m2;
  if (m1_in>m2_in) {
     chi1 = chi1_in;
     chi2 = chi2_in;
     m1   = m1_in;
     m2   = m2_in;
  } else { // swap spins and masses
     chi1 = chi2_in;
     chi2 = chi1_in;
     m1   = m2_in;
     m2   = m1_in;
  }

  int status = init_useful_powers(&powers_of_pi, PI);
  CHECK(SUCCESS == status, ERROR_EFUNC, "Failed to initiate useful powers of pi.");

  double M = m1 + m2;
  double eta = m1 * m2 / (M * M);

  if (eta > 0.25 || eta < 0.0)
    ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");

  double M_sec = M * MTSUN_SI;

  double MfRef_for_tRef_in = M_sec * fRef_for_tRef_in;
  double MfRef_for_phiRef_in = M_sec * fRef_for_phiRef_in;
  double tMRef = tRef / M_sec;
  double twopiDeltatM = Deltat * 2*PI / M_sec;

  /* Compute the amplitude pre-factor */
  //double amp0 = 2. * sqrt(5. / (64.*PI)) * M * MRSUN_SI * M * MTSUN_SI / distance;
  /* NOTE: we will output the amplitude of the 22 mode - so we remove the factor 2. * sqrt(5. / (64.*PI)), which is part of the Y22 spherical harmonic factor */
  double amp0 = M * MRSUN_SI * M * MTSUN_SI / (distance * 1e6 * PC_SI);

  /* Max frequency covered by PhenomD */
  double fCut = f_CUT/M_sec; // convert Mf -> Hz

  /* Allocate h22 */
  *h22 = CreateAmpPhaseFDWaveform(n);
  snprintf(errstr, strlen(errstr), "Failed to allocated waveform AmpPhaseFDWaveform of length %zu.", n);
  CHECK ( *h22, ERROR_ENOMEM, errstr);

  /* Allocate tf */
  *tf = real_vector_alloc(n);

  /* range that will have actual non-zero waveform values generated */
  // size_t ind_min = (size_t) (f_min / deltaF);
  // size_t ind_max = (size_t) (f_max / deltaF);
  // snprintf(errstr, strlen(errstr), "minimum freq index %zu and maximum freq index %zu do not fulfill 0<=ind_min<=ind_max<=htilde->data>length=%zu.", ind_min, ind_max, n);
  // CHECK ( (ind_max<=n) && (ind_min<=ind_max), ERROR_EDOM, errstr);

  // Calculate phenomenological parameters
  double finspin = FinalSpin0815(eta, chi1, chi2); //FinalSpin0815 - 0815 is like a version number

  if (finspin < MIN_FINAL_SPIN) {
    snprintf(errstr, strlen(errstr),
        "Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here.", finspin);
    WARNING(errstr);
  }

  IMRPhenomDAmplitudeCoefficients *pAmp = malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1, chi2, finspin);
  if (!pAmp) ERROR(ERROR_EFUNC, "");
  // spin order LAL_SIM_INSPIRAL_SPIN_ORDER_35PN
  IMRPhenomDPhaseCoefficients *pPhi = malloc(sizeof(IMRPhenomDPhaseCoefficients));
  ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1, chi2, finspin);
  if (!pPhi) ERROR(ERROR_EFUNC, "");
  PNPhasingSeries *pn = NULL;
  TaylorF2AlignedPhasing(&pn, m1, m2, chi1, chi2);
  if (!pn) ERROR(ERROR_EFUNC, "");

  // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
  // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but

  // was not available when PhenomD was tuned.
  pn->v[6] -= (Subtract3PNSS(m1, m2, M, eta, chi1, chi2) * pn->v[0]);

  PhiInsPrefactors phi_prefactors;
  status = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_phi_ins_prefactors failed");

  // Compute coefficients to make phase C^1 continuous (phase and first derivative)
  ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors, 1., 1.);

  AmpInsPrefactors amp_prefactors;
  status = init_amp_ins_prefactors(&amp_prefactors, pAmp);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_amp_ins_prefactors failed");

  /* Reference frequency to set tRef -- if 0, use fpeak */
  double MfRef_for_tRef = (MfRef_for_tRef_in == 0.0) ? pAmp->fmaxCalc : MfRef_for_tRef_in;
  /* Reference frequency to set phiRef -- if 0, use fpeak */
  double MfRef_for_phiRef = (MfRef_for_phiRef_in == 0.0) ? pAmp->fmaxCalc : MfRef_for_phiRef_in;

  //time shift so that peak amplitude is approximately at t=0
  //For details see https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timERROR_EDOMain
  //double t0 = DPhiMRD(pAmp->fmaxCalc, pPhi);

  /* Time shift to be applied later to ensure time is tRef at fRef_for_tRef */
  //double twopitM_shiftRef = DPhiMRD(MfRef_for_tRef, pPhi);
  double twopitM_shiftRef = 0.;
  UsefulPowers powers_of_MfRef;
  status = init_useful_powers(&powers_of_MfRef, MfRef_for_tRef);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for MfRef_for_tRef");
  if ((extraparams==NULL) || (!(extraparams->use_buggy_LAL_tpeak))) {
    twopitM_shiftRef = IMRPhenDPhaseDerivative(MfRef_for_tRef, pPhi, pn, &powers_of_MfRef, &phi_prefactors, 1., 1., modgrparams);
  }
  else {
    twopitM_shiftRef = DPhiMRD(MfRef_for_tRef, pPhi, 1., 1.);
  }

  /* To set the phase at fRef_for_phiRef to phiRef, first compute value before shift */
  status = init_useful_powers(&powers_of_MfRef, MfRef_for_phiRef);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for MfRef_for_phiRef");
  double phi_shiftRef = IMRPhenDPhase(MfRef_for_phiRef, pPhi, pn, &powers_of_MfRef, &phi_prefactors, 1., 1., modgrparams);

  /* NOTE: phiRef interpreted as an orbital phase */
  /* Therefore enters phase of the 22 mode in given direction as -2phiRef */
  double phi_shift_total = -2.*phiRef - phi_shiftRef;
  double twopitMRef = 2*PI*tMRef;
  /* If force_phiRef_fRef, we ensure the adjustment of time leaves phiRef invariant */
  /* so that phi(fRef_for_phiRef) = -2*phiRef */
  /* Else, adjusting tRef changes the phase at the phiRef like a normal timeshift does */
  /* so that phi(fRef_for_phiRef) = -2*phiRef + 2pi*fRef_for_phiRef*tRef */
  double const_to_avoid_changing_phiRef = 0.;
  if (force_phiRef_fRef) const_to_avoid_changing_phiRef = MfRef_for_phiRef;

  int status_in_for = SUCCESS;
  /* Now generate the waveform on the frequencies given by freq */
  double inv2pi = 1./(2*PI);
  double* f = freq->data;
  double* freqwf = (*h22)->freq;
  double* amp = (*h22)->amp;
  double* phase = (*h22)->phase;
  double* tf_data = (*tf)->data;
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++)
  {
    freqwf[i] = f[i];
    if(f[i]>fCut) { // for frequencies exceeding the maximal frequency covered by PhenomD, put 0 amplitude and phase
      amp[i] = 0.;
      phase[i] = 0.;
    }
    else {
      double Mf = M_sec * f[i]; // geometric frequency

      UsefulPowers powers_of_Mf;
      status_in_for = init_useful_powers(&powers_of_Mf, Mf);
      if (SUCCESS != status_in_for)
      {
        snprintf(errstr, strlen(errstr), "init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
        ERROR(ERROR_EFUNC, errstr);
        status = status_in_for;
      }
      else
      {
        double a = IMRPhenDAmplitude(Mf, pAmp, &powers_of_Mf, &amp_prefactors);
        double phi = IMRPhenDPhase(Mf, pPhi, pn, &powers_of_Mf, &phi_prefactors, 1., 1., modgrparams);
        double tfM_val = inv2pi * IMRPhenDPhaseDerivative(Mf, pPhi, pn, &powers_of_Mf, &phi_prefactors, 1., 1., modgrparams);

        /* After this, phi(fRef_for_phiRef) = -2*phiRef and tf(fRef_for_tRef) = 0 */
        phi += -twopitM_shiftRef * (Mf-MfRef_for_phiRef) + phi_shift_total;
        /* After this, phi(fRef_for_phiRef) = -2*phiRef if force_phiRef_fRef */
        /* phi(fRef_for_phiRef) = -2*phiRef + 2pi*fRef_for_phiRef*tRef if not force_phiRef_fRef */
        /* and tf(fRef_for_tRef) = tRef */
        phi += twopitMRef * (Mf-const_to_avoid_changing_phiRef);
        phi += twopiDeltatM * Mf;
        tfM_val += (-twopitM_shiftRef + twopitMRef + twopiDeltatM) * inv2pi;
        amp[i] = amp0 * a;
        phase[i] = phi; /* NOTE: opposite Fourier convention with respect to LAL */
        tf_data[i] = tfM_val * M_sec; /* NOTE: we output times in physical units, not t/M */
      }
    }
  }

  /* Output the internally computed approximate peak frequency, tf and phase 22 */
  double Mfpeak_val = pAmp->fmaxCalc;
  UsefulPowers powers_of_Mfpeak;
  status = init_useful_powers(&powers_of_Mfpeak, Mfpeak_val);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for Mfpeak_val");
  double twopitMpeak_val = IMRPhenDPhaseDerivative(Mfpeak_val, pPhi, pn, &powers_of_Mfpeak, &phi_prefactors, 1., 1., modgrparams);
  twopitMpeak_val += -twopitM_shiftRef;
  twopitMpeak_val += twopitMRef;
  twopitMpeak_val += twopiDeltatM;
  double phipeak_val = IMRPhenDPhase(Mfpeak_val, pPhi, pn, &powers_of_Mfpeak, &phi_prefactors, 1., 1., modgrparams);
  phipeak_val += -twopitM_shiftRef * (Mfpeak_val-MfRef_for_phiRef) + phi_shift_total;
  phipeak_val += twopitMRef * (Mfpeak_val-const_to_avoid_changing_phiRef);
  phipeak_val += twopiDeltatM * Mfpeak_val;
  *fpeak = Mfpeak_val / M_sec;
  *tpeak = twopitMpeak_val * M_sec / (2*PI);
  *phipeak = phipeak_val;

  /* Output the starting frequency, tf and phase 22 */
  double Mfstart_val = M_sec * f[0];
  UsefulPowers powers_of_Mfstart;
  status = init_useful_powers(&powers_of_Mfstart, Mfstart_val);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for Mfstart_val");
  double twopitMstart_val = IMRPhenDPhaseDerivative(Mfstart_val, pPhi, pn, &powers_of_Mfstart, &phi_prefactors, 1., 1., modgrparams);
  twopitMstart_val += -twopitM_shiftRef;
  twopitMstart_val += twopitMRef;
  twopitMstart_val += twopiDeltatM;
  double phistart_val = IMRPhenDPhase(Mfstart_val, pPhi, pn, &powers_of_Mfstart, &phi_prefactors, 1., 1., modgrparams);
  phistart_val += -twopitM_shiftRef * (Mfstart_val-MfRef_for_phiRef) + phi_shift_total;
  phistart_val += twopitMRef * (Mfstart_val-const_to_avoid_changing_phiRef);
  phistart_val += twopiDeltatM * Mfstart_val;
  *fstart = f[0];
  *tstart = twopitMstart_val * M_sec / (2*PI);
  *phistart = phistart_val;

  free(pAmp);
  free(pPhi);
  free(pn);

  return status;
}

double IMRPhenomDInternalTimeOfFrequency(
  const double Mf,
  const double twopitM_shiftRef,
  const double twopitMRef,
  const double twopiDeltatM,
  PNPhasingSeries* pn,
  IMRPhenomDPhaseCoefficients* pPhi,
  PhiInsPrefactors* phi_prefactors,
  const ModGRParams* modgrparams
) {
  UsefulPowers powers_of_Mf;
  int status = init_useful_powers(&powers_of_Mf, Mf);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for Mf");
  double twopitM = IMRPhenDPhaseDerivative(Mf, pPhi, pn, &powers_of_Mf, phi_prefactors, 1., 1., modgrparams);
  twopitM += -twopitM_shiftRef;
  twopitM += twopitMRef;
  twopitM += twopiDeltatM;
  return twopitM / (2*PI);
}

/* Function that computes t(f), same interface as AmpPhase function */
/* NOTE: fair amount of code duplication / initialization repeated */
double IMRPhenomDComputeTimeOfFrequency(
    const double f,                     /**< Input frequency (Hz): we compute t(f) */
    const double m1_in,                 /**< Mass of companion 1 (solar masses) */
    const double m2_in,                 /**< Mass of companion 2 (solar masses) */
    const double chi1_in,               /**< Aligned-spin parameter of companion 1 */
    const double chi2_in,               /**< Aligned-spin parameter of companion 2 */
    const double distance,              /**< Distance of source (Mpc) */
    const double tRef,                  /**< Time at fRef_for_tRef (s) */
    const double phiRef,                /**< Orbital phase at fRef_for_phiRef (rad) */
    const double fRef_for_tRef_in,      /**< Ref. frequency (Hz) for tRef */
    const double fRef_for_phiRef_in,    /**< Ref. frequency (Hz) for phiRef */
    const int force_phiRef_fRef,        /**< Flag to force phiRef at fRef after adjusting tRef */
    const double Deltat,                /**< Time shift (s) applied a posteriori */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
) {
  // size_t n = freq->size;

  char errstr[200];

  double chi1, chi2, m1, m2;
  if (m1_in>m2_in) {
     chi1 = chi1_in;
     chi2 = chi2_in;
     m1   = m1_in;
     m2   = m2_in;
  } else { // swap spins and masses
     chi1 = chi2_in;
     chi2 = chi1_in;
     m1   = m2_in;
     m2   = m1_in;
  }

  int status = init_useful_powers(&powers_of_pi, PI);
  CHECK(SUCCESS == status, ERROR_EFUNC, "Failed to initiate useful powers of pi.");

  double M = m1 + m2;
  double eta = m1 * m2 / (M * M);

  if (eta > 0.25 || eta < 0.0)
    ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");

  double M_sec = M * MTSUN_SI;

  double MfRef_for_tRef_in = M_sec * fRef_for_tRef_in;
  //double MfRef_for_phiRef_in = M_sec * fRef_for_phiRef_in;
  double tMRef = tRef / M_sec;
  double twopiDeltatM = Deltat * 2*PI / M_sec;

  /* Compute the amplitude pre-factor */
  //double amp0 = 2. * sqrt(5. / (64.*PI)) * M * MRSUN_SI * M * MTSUN_SI / distance;
  /* NOTE: we will output the amplitude of the 22 mode - so we remove the factor 2. * sqrt(5. / (64.*PI)), which is part of the Y22 spherical harmonic factor */
  //double amp0 = M * MRSUN_SI * M * MTSUN_SI / (distance * 1e6 * PC_SI);

  /* Max frequency covered by PhenomD */
  //double fCut = f_CUT/M_sec; // convert Mf -> Hz

  // /* Allocate h22 */
  // *h22 = CreateAmpPhaseFDWaveform(n);
  // snprintf(errstr, strlen(errstr), "Failed to allocated waveform AmpPhaseFDWaveform of length %zu.", n);
  // CHECK ( *h22, ERROR_ENOMEM, errstr);

  /* range that will have actual non-zero waveform values generated */
  // size_t ind_min = (size_t) (f_min / deltaF);
  // size_t ind_max = (size_t) (f_max / deltaF);
  // snprintf(errstr, strlen(errstr), "minimum freq index %zu and maximum freq index %zu do not fulfill 0<=ind_min<=ind_max<=htilde->data>length=%zu.", ind_min, ind_max, n);
  // CHECK ( (ind_max<=n) && (ind_min<=ind_max), ERROR_EDOM, errstr);

  // Calculate phenomenological parameters
  double finspin = FinalSpin0815(eta, chi1, chi2); //FinalSpin0815 - 0815 is like a version number

  if (finspin < MIN_FINAL_SPIN) {
    snprintf(errstr, strlen(errstr),
        "Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here.", finspin);
    WARNING(errstr);
  }

  IMRPhenomDAmplitudeCoefficients *pAmp = malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1, chi2, finspin);
  if (!pAmp) ERROR(ERROR_EFUNC, "");
  // spin order LAL_SIM_INSPIRAL_SPIN_ORDER_35PN
  IMRPhenomDPhaseCoefficients *pPhi = malloc(sizeof(IMRPhenomDPhaseCoefficients));
  ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1, chi2, finspin);
  if (!pPhi) ERROR(ERROR_EFUNC, "");
  PNPhasingSeries *pn = NULL;
  TaylorF2AlignedPhasing(&pn, m1, m2, chi1, chi2);
  if (!pn) ERROR(ERROR_EFUNC, "");

  // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
  // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but

  // was not available when PhenomD was tuned.
  pn->v[6] -= (Subtract3PNSS(m1, m2, M, eta, chi1, chi2) * pn->v[0]);

  PhiInsPrefactors phi_prefactors;
  status = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_phi_ins_prefactors failed");

  // Compute coefficients to make phase C^1 continuous (phase and first derivative)
  ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors, 1., 1.);

  AmpInsPrefactors amp_prefactors;
  status = init_amp_ins_prefactors(&amp_prefactors, pAmp);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_amp_ins_prefactors failed");

  /* Reference frequency to set tRef -- if 0, use fpeak */
  double MfRef_for_tRef = (MfRef_for_tRef_in == 0.0) ? pAmp->fmaxCalc : MfRef_for_tRef_in;
  /* Reference frequency to set phiRef -- if 0, use fpeak */
  //double MfRef_for_phiRef = (MfRef_for_phiRef_in == 0.0) ? pAmp->fmaxCalc : MfRef_for_phiRef_in;

  //time shift so that peak amplitude is approximately at t=0
  //For details see https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timERROR_EDOMain
  //double t0 = DPhiMRD(pAmp->fmaxCalc, pPhi);

  /* Time shift to be applied later to ensure time is tRef at fRef_for_tRef */
  //double twopitM_shiftRef = DPhiMRD(MfRef_for_tRef, pPhi);
  double twopitM_shiftRef = 0.;
  UsefulPowers powers_of_MfRef;
  status = init_useful_powers(&powers_of_MfRef, MfRef_for_tRef);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for MfRef_for_tRef");
  if ((extraparams==NULL) || (!(extraparams->use_buggy_LAL_tpeak))) {
    twopitM_shiftRef = IMRPhenDPhaseDerivative(MfRef_for_tRef, pPhi, pn, &powers_of_MfRef, &phi_prefactors, 1., 1., modgrparams);
  }
  else {
    twopitM_shiftRef = DPhiMRD(MfRef_for_tRef, pPhi, 1., 1.);
  }

  /* To set the phase at fRef_for_phiRef to phiRef, first compute value before shift */
  // status = init_useful_powers(&powers_of_MfRef, MfRef_for_phiRef);
  // CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for MfRef_for_phiRef");
  // double phi_shiftRef = IMRPhenDPhase(MfRef_for_phiRef, pPhi, pn, &powers_of_MfRef, &phi_prefactors, modgrparams);

  /* NOTE: phiRef interpreted as an orbital phase */
  /* Therefore enters phase of the 22 mode in given direction as -2phiRef */
  //double phi_shift_total = -2.*phiRef - phi_shiftRef;
  double twopitMRef = 2*PI*tMRef;
  /* If force_phiRef_fRef, we ensure the adjustment of time leaves phiRef invariant */
  /* so that phi(fRef_for_phiRef) = -2*phiRef */
  /* Else, adjusting tRef changes the phase at the phiRef like a normal timeshift does */
  /* so that phi(fRef_for_phiRef) = -2*phiRef + 2pi*fRef_for_phiRef*tRef */
  // double const_to_avoid_changing_phiRef = 0.;
  // if (force_phiRef_fRef) const_to_avoid_changing_phiRef = MfRef_for_phiRef;

  double Mf = f * M_sec;

  double tM = IMRPhenomDInternalTimeOfFrequency(Mf, twopitM_shiftRef, twopitMRef, twopiDeltatM, pn, pPhi, &phi_prefactors, modgrparams);
  double t = tM * M_sec;

  free(pAmp);
  free(pPhi);
  free(pn);

  return t;
}

/* Function that computes f(t), same interface as AmpPhase function */
/* NOTE: fair amount of code duplication / initialization repeated */
double IMRPhenomDComputeInverseFrequencyOfTime(
    const double t,                     /**< Input time (s): we compute f(t) */
    const double f_estimate,            /**< Estimate of f(t), use to initialize */
    const double t_acc,                 /**< Target accuracy of t(f) where to stop refining f */
    const double m1_in,                 /**< Mass of companion 1 (solar masses) */
    const double m2_in,                 /**< Mass of companion 2 (solar masses) */
    const double chi1_in,               /**< Aligned-spin parameter of companion 1 */
    const double chi2_in,               /**< Aligned-spin parameter of companion 2 */
    const double distance,              /**< Distance of source (Mpc) */
    const double tRef,                  /**< Time at fRef_for_tRef (s) */
    const double phiRef,                /**< Orbital phase at fRef_for_phiRef (rad) */
    const double fRef_for_tRef_in,      /**< Ref. frequency (Hz) for tRef */
    const double fRef_for_phiRef_in,    /**< Ref. frequency (Hz) for phiRef */
    const int force_phiRef_fRef,        /**< Flag to force phiRef at fRef after adjusting tRef */
    const double Deltat,                /**< Time shift (s) applied a posteriori */
    const int max_iter,                 /**< Maximal number of iterations in bisection */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
) {
  // size_t n = freq->size;

  char errstr[200];

  double chi1, chi2, m1, m2;
  if (m1_in>m2_in) {
     chi1 = chi1_in;
     chi2 = chi2_in;
     m1   = m1_in;
     m2   = m2_in;
  } else { // swap spins and masses
     chi1 = chi2_in;
     chi2 = chi1_in;
     m1   = m2_in;
     m2   = m1_in;
  }

  int status = init_useful_powers(&powers_of_pi, PI);
  CHECK(SUCCESS == status, ERROR_EFUNC, "Failed to initiate useful powers of pi.");

  double M = m1 + m2;
  double eta = m1 * m2 / (M * M);

  if (eta > 0.25 || eta < 0.0)
    ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");

  double M_sec = M * MTSUN_SI;

  double MfRef_for_tRef_in = M_sec * fRef_for_tRef_in;
  //double MfRef_for_phiRef_in = M_sec * fRef_for_phiRef_in;
  double tMRef = tRef / M_sec;
  double twopiDeltatM = Deltat * 2*PI / M_sec;

  /* Compute the amplitude pre-factor */
  //double amp0 = 2. * sqrt(5. / (64.*PI)) * M * MRSUN_SI * M * MTSUN_SI / distance;
  /* NOTE: we will output the amplitude of the 22 mode - so we remove the factor 2. * sqrt(5. / (64.*PI)), which is part of the Y22 spherical harmonic factor */
  //double amp0 = M * MRSUN_SI * M * MTSUN_SI / (distance * 1e6 * PC_SI);

  /* Max frequency covered by PhenomD */
  //double fCut = f_CUT/M_sec; // convert Mf -> Hz

  // /* Allocate h22 */
  // *h22 = CreateAmpPhaseFDWaveform(n);
  // snprintf(errstr, strlen(errstr), "Failed to allocated waveform AmpPhaseFDWaveform of length %zu.", n);
  // CHECK ( *h22, ERROR_ENOMEM, errstr);

  /* range that will have actual non-zero waveform values generated */
  // size_t ind_min = (size_t) (f_min / deltaF);
  // size_t ind_max = (size_t) (f_max / deltaF);
  // snprintf(errstr, strlen(errstr), "minimum freq index %zu and maximum freq index %zu do not fulfill 0<=ind_min<=ind_max<=htilde->data>length=%zu.", ind_min, ind_max, n);
  // CHECK ( (ind_max<=n) && (ind_min<=ind_max), ERROR_EDOM, errstr);

  // Calculate phenomenological parameters
  double finspin = FinalSpin0815(eta, chi1, chi2); //FinalSpin0815 - 0815 is like a version number

  if (finspin < MIN_FINAL_SPIN) {
    snprintf(errstr, strlen(errstr),
        "Final spin (Mf=%g) and ISCO frequency of this system are small, the model might misbehave here.", finspin);
    WARNING(errstr);
  }

  IMRPhenomDAmplitudeCoefficients *pAmp = malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1, chi2, finspin);
  if (!pAmp) ERROR(ERROR_EFUNC, "");
  // spin order LAL_SIM_INSPIRAL_SPIN_ORDER_35PN
  IMRPhenomDPhaseCoefficients *pPhi = malloc(sizeof(IMRPhenomDPhaseCoefficients));
  ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1, chi2, finspin);
  if (!pPhi) ERROR(ERROR_EFUNC, "");
  PNPhasingSeries *pn = NULL;
  TaylorF2AlignedPhasing(&pn, m1, m2, chi1, chi2);
  if (!pn) ERROR(ERROR_EFUNC, "");

  // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
  // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but

  // was not available when PhenomD was tuned.
  pn->v[6] -= (Subtract3PNSS(m1, m2, M, eta, chi1, chi2) * pn->v[0]);

  PhiInsPrefactors phi_prefactors;
  status = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_phi_ins_prefactors failed");

  // Compute coefficients to make phase C^1 continuous (phase and first derivative)
  ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors, 1., 1.);

  AmpInsPrefactors amp_prefactors;
  status = init_amp_ins_prefactors(&amp_prefactors, pAmp);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_amp_ins_prefactors failed");

  /* Reference frequency to set tRef -- if 0, use fpeak */
  double MfRef_for_tRef = (MfRef_for_tRef_in == 0.0) ? pAmp->fmaxCalc : MfRef_for_tRef_in;
  /* Reference frequency to set phiRef -- if 0, use fpeak */
  //double MfRef_for_phiRef = (MfRef_for_phiRef_in == 0.0) ? pAmp->fmaxCalc : MfRef_for_phiRef_in;

  //time shift so that peak amplitude is approximately at t=0
  //For details see https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timERROR_EDOMain
  //double t0 = DPhiMRD(pAmp->fmaxCalc, pPhi);

  /* Time shift to be applied later to ensure time is tRef at fRef_for_tRef */
  //double twopitM_shiftRef = DPhiMRD(MfRef_for_tRef, pPhi);
  double twopitM_shiftRef = 0.;
  UsefulPowers powers_of_MfRef;
  status = init_useful_powers(&powers_of_MfRef, MfRef_for_tRef);
  CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for MfRef_for_tRef");
  if ((extraparams==NULL) || (!(extraparams->use_buggy_LAL_tpeak))) {
    twopitM_shiftRef = IMRPhenDPhaseDerivative(MfRef_for_tRef, pPhi, pn, &powers_of_MfRef, &phi_prefactors, 1., 1., modgrparams);
  }
  else {
    twopitM_shiftRef = DPhiMRD(MfRef_for_tRef, pPhi, 1., 1.);
  }

  /* To set the phase at fRef_for_phiRef to phiRef, first compute value before shift */
  // status = init_useful_powers(&powers_of_MfRef, MfRef_for_phiRef);
  // CHECK(SUCCESS == status, ERROR_EFUNC, "init_useful_powers failed for MfRef_for_phiRef");
  // double phi_shiftRef = IMRPhenDPhase(MfRef_for_phiRef, pPhi, pn, &powers_of_MfRef, &phi_prefactors, modgrparams);

  /* NOTE: phiRef interpreted as an orbital phase */
  /* Therefore enters phase of the 22 mode in given direction as -2phiRef */
  //double phi_shift_total = -2.*phiRef - phi_shiftRef;
  double twopitMRef = 2*PI*tMRef;
  /* If force_phiRef_fRef, we ensure the adjustment of time leaves phiRef invariant */
  /* so that phi(fRef_for_phiRef) = -2*phiRef */
  /* Else, adjusting tRef changes the phase at the phiRef like a normal timeshift does */
  /* so that phi(fRef_for_phiRef) = -2*phiRef + 2pi*fRef_for_phiRef*tRef */
  // double const_to_avoid_changing_phiRef = 0.;
  // if (force_phiRef_fRef) const_to_avoid_changing_phiRef = MfRef_for_phiRef;

  // int status_in_for = SUCCESS;
  // /* Now generate the waveform on the frequencies given by freq */
  // double* f = freq->data;
  // double* freqwf = (*h22)->freq;
  // double* amp = (*h22)->amp;
  // double* phase = (*h22)->phase;
  // #pragma omp parallel for
  // for (size_t i = 0; i < n; i++)
  // {
  //   freqwf[i] = f[i];
  //   if(f[i]>fCut) { // for frequencies exceeding the maximal frequency covered by PhenomD, put 0 amplitude and phase
  //     amp[i] = 0.;
  //     phase[i] = 0.;
  //   }
  //   else {
  //     double Mf = M_sec * f[i]; // geometric frequency
  //
  //     UsefulPowers powers_of_Mf;
  //     status_in_for = init_useful_powers(&powers_of_Mf, Mf);
  //     if (SUCCESS != status_in_for)
  //     {
  //       snprintf(errstr, strlen(errstr), "init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
  //       ERROR(ERROR_EFUNC, errstr);
  //       status = status_in_for;
  //     }
  //     else
  //     {
  //       double a = IMRPhenDAmplitude(Mf, pAmp, &powers_of_Mf, &amp_prefactors);
  //       double phi = IMRPhenDPhase(Mf, pPhi, pn, &powers_of_Mf, &phi_prefactors);
  //
  //       /* After this, phi(fRef_for_phiRef) = -2*phiRef and tf(fRef_for_tRef) = 0 */
  //       phi += -twopitM_shiftRef * (Mf-MfRef_for_phiRef) + phi_shift_total;
  //       /* After this, phi(fRef_for_phiRef) = -2*phiRef if force_phiRef_fRef */
  //       /* phi(fRef_for_phiRef) = -2*phiRef + 2pi*fRef_for_phiRef*tRef if not force_phiRef_fRef */
  //       /* and tf(fRef_for_tRef) = tRef */
  //       phi += twopitMRef * (Mf-const_to_avoid_changing_phiRef);
  //       amp[i] = amp0 * a;
  //       phase[i] = phi; /* NOTE: opposite Fourier convention with respect to LAL */
  //     }
  //   }
  // }

  double tM = t / M_sec;
  double Mf_estimate = f_estimate * M_sec;
  double tM_acc = t_acc / M_sec;
  double tM_computed = 0;

  int i = 0;

  double Mflow = Mf_estimate;
  double Mfhigh = Mf_estimate;
  double Mf = 0;

  tM_computed = IMRPhenomDInternalTimeOfFrequency(Mf_estimate, twopitM_shiftRef, twopitMRef, twopiDeltatM, pn, pPhi, &phi_prefactors, modgrparams);
  i = 0;
  while ( (i<max_iter) && (tM_computed <= tM) ) {
    Mfhigh = 1.2*Mfhigh;
    tM_computed = IMRPhenomDInternalTimeOfFrequency(Mfhigh, twopitM_shiftRef, twopitMRef, twopiDeltatM, pn, pPhi, &phi_prefactors, modgrparams);
    i++;
  }

  tM_computed = IMRPhenomDInternalTimeOfFrequency(Mf_estimate, twopitM_shiftRef, twopitMRef, twopiDeltatM, pn, pPhi, &phi_prefactors, modgrparams);
  i = 0;
  while ( (i<max_iter) && (tM_computed >= tM) ) {
    Mflow = 0.8*Mflow;
    tM_computed = IMRPhenomDInternalTimeOfFrequency(Mflow, twopitM_shiftRef, twopitMRef, twopiDeltatM, pn, pPhi, &phi_prefactors, modgrparams);
    i++;
  }

  Mf = sqrt(Mflow * Mfhigh);
  tM_computed = IMRPhenomDInternalTimeOfFrequency(Mf, twopitM_shiftRef, twopitMRef, twopiDeltatM, pn, pPhi, &phi_prefactors, modgrparams);
  i = 0;
  while ( (i<max_iter) && (fabs(tM_computed - tM) > tM_acc) ) {
    if (tM_computed > tM) {
      Mfhigh = Mf;
    }
    else {
      Mflow = Mf;
    }
    Mf = sqrt(Mflow * Mfhigh);
    tM_computed = IMRPhenomDInternalTimeOfFrequency(Mf, twopitM_shiftRef, twopitMRef, twopiDeltatM, pn, pPhi, &phi_prefactors, modgrparams);
    i++;
  }

  double f = Mf / M_sec;
  CHECKP(i < max_iter, "Reached maximal number of iterations.");
  CHECKP(Mf < pAmp->fmaxCalc, "Reached higher than peak frequency, where tf becomes unreliable.");
  if ((!(i<max_iter)) | (!(Mf < pAmp->fmaxCalc)))
    f = ERROR_EFUNC;

  free(pAmp);
  free(pPhi);
  free(pn);

  return f;
}

// /**
//  * Function to return the frequency (in Hz) of the peak of the frequency
//  * domain amplitude for the IMRPhenomD model.
//  *
//  * The peak is a parameter in the PhenomD model given by Eq. 20 in 1508.07253
//  * where it is called f_peak in the paper.
//  */
// double XLALIMRPhenomDGetPeakFreq(
//     const double m1_in,                 /**< mass of companion 1 [Msun] */
//     const double m2_in,                 /**< mass of companion 2 [Msun] */
//     const double chi1_in,               /**< aligned-spin of companion 1 */
//     const double chi2_in                /**< aligned-spin of companion 2 */
// ) {
//     // Ensure that m1 > m2 and that chi1 is the spin on m1
//     double chi1, chi2, m1, m2;
//     if (m1_in>m2_in) {
//        chi1 = chi1_in;
//        chi2 = chi2_in;
//        m1   = m1_in;
//        m2   = m2_in;
//     } else { // swap spins and masses
//        chi1 = chi2_in;
//        chi2 = chi1_in;
//        m1   = m2_in;
//        m2   = m1_in;
//     }
//
//     const double M = m1 + m2;
//     const double M_sec = M * MTSUN_SI; // Conversion factor Hz -> dimensionless frequency
//
//     double eta = m1 * m2 / (M * M);
//     if (eta > 0.25 || eta < 0.0)
//       ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");
//
//     // Calculate phenomenological parameters
//     double finspin = FinalSpin0815(eta, chi1, chi2);
//
//     if (finspin < MIN_FINAL_SPIN)
//           PRINT_WARNING("Final spin (Mf=%g) and ISCO frequency of this system are small, \
//                           the model might misbehave here.", finspin);
//     IMRPhenomDAmplitudeCoefficients *pAmp = ComputeIMRPhenomDAmplitudeCoefficients(eta, chi1, chi2, finspin);
//     if (!pAmp) ERROR(ERROR_EFUNC);
//
//     // PeakFreq, converted to Hz
//     double PeakFreq = ( pAmp->fmaxCalc ) / M_sec;
//
//     LALFree(pAmp);
//
//     return PeakFreq;
// }
//
//
// // protoype
// static double PhenDPhaseDerivFrequencyPoint(double Mf, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn);
//
// /**
//  * Helper function to return the value of the frequency derivative of the
//  * Fourier domain phase.
//  * This is function is wrapped by IMRPhenomDPhaseDerivative and used
//  * when estimating the length of the time domain version of the waveform.
//  * unreviewed
//  */
// static double PhenDPhaseDerivFrequencyPoint(double Mf, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn)
// {
//
//   // split the calculation to just 1 of 3 possible mutually exclusive ranges
//
//   if (!StepFunc_boolean(Mf, p->fInsJoin))	// Inspiral range
//   {
//       double DPhiIns = DPhiInsAnsatzInt(Mf, p, pn);
// 	  return DPhiIns;
//   }
//
//   if (StepFunc_boolean(Mf, p->fMRDJoin))	// MRD range
//   {
//       double DPhiMRDval = DPhiMRD(Mf, p) + p->C2MRD;
// 	  return DPhiMRDval;
//   }
//
//   //	Intermediate range
//   double DPhiInt = DPhiIntAnsatz(Mf, p) + p->C2Int;
//   return DPhiInt;
// }
//
// /**
// * Estimates the length of the time domain IMRPhenomD signal
// * This does NOT taking into account any tapering that is used to condition the
// * Fourier domain waveform to compute the inverse Fourer transform.
// * To estimate the length we assume that the waveform only reaches the
// * the highest physics frequency i.e. the ringdown frequency.
// * unreviewed
// */
// double XLALSimIMRPhenomDChirpTime(
//     const double m1_SI,                 /**< Mass of companion 1 (kg) */
//     const double m2_SI,                 /**< Mass of companion 2 (kg) */
//     const double chi1_in,               /**< aligned-spin of companion 1 */
//     const double chi2_in,               /**< aligned-spin of companion 2 */
//     const double fHzSt                  /**< arbitrary starting frequency in Hz */
// ) {
//
//     if (fHzSt <= 0) ERROR(ERROR_EDOM, "fHzSt must be positive\n");
//
//     if (chi1_in > 1.0 || chi1_in < -1.0 || chi2_in > 1.0 || chi2_in < -1.0)
//       ERROR(ERROR_EDOM, "Spins outside the range [-1,1] are not supported\n");
//
//     /* external: SI; internal: solar masses */
//     const double m1_in = m1_SI / MSUN_SI;
//     const double m2_in = m2_SI / MSUN_SI;
//
//     double chi1, chi2, m1, m2;
//     if (m1_in>m2_in) {
//        chi1 = chi1_in;
//        chi2 = chi2_in;
//        m1   = m1_in;
//        m2   = m2_in;
//     } else { // swap spins and masses
//        chi1 = chi2_in;
//        chi2 = chi1_in;
//        m1   = m2_in;
//        m2   = m1_in;
//     }
//
//     // check that starting frequency is not higher than the peak frequency
//     const double fHzPeak = XLALIMRPhenomDGetPeakFreq(m1, m2, chi1, chi2);
//     if (fHzSt > fHzPeak){
//         PRINT_WARNING("Starting frequency = %f Hz is higher IMRPhenomD peak frequency %f Hz. Results may be unreliable.", fHzSt, fHzPeak);
//     }
//
//     int status = init_useful_powers(&powers_of_pi, PI);
//     CHECK(SUCCESS == status, status, "Failed to initiate useful powers of pi.");
//
//     const double M = m1 + m2;
//     const double eta = m1 * m2 / (M * M);
//
//     if (eta > 0.25 || eta < 0.0)
//       ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");
//
//     // compute geometric frequency
//     const double M_sec = M * MTSUN_SI;
//     const double MfSt = M_sec * fHzSt;
//
//     // Calculate phenomenological parameters
//     const double finspin = FinalSpin0815(eta, chi1, chi2); //FinalSpin0815 - 0815 is like a version number
//
//     if (finspin < MIN_FINAL_SPIN)
//             PRINT_WARNING("Final spin (Mf=%g) and ISCO frequency of this system are small, \
//                             the model might misbehave here.", finspin);
//     IMRPhenomDPhaseCoefficients *pPhi = ComputeIMRPhenomDPhaseCoefficients(eta, chi1, chi2, finspin);
//     if (!pPhi) ERROR(ERROR_EFUNC);
//     PNPhasingSeries *pn = NULL;
//     TaylorF2AlignedPhasing(&pn, m1, m2, chi1, chi2);
//     if (!pn) ERROR(ERROR_EFUNC);
//
//     // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
//     // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
//     // was not available when PhenomD was tuned.
//     pn->v[6] -= (Subtract3PNSS(m1, m2, M, chi1, chi2) * pn->v[0]);
//
//
//     PhiInsPrefactors phi_prefactors;
//     status = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
//     CHECK(SUCCESS == status, status, "init_phi_ins_prefactors failed");
//
//     // Compute coefficients to make phase C^1 continuous (phase and first derivative)
//     ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors);
//
//     // We estimate the length of the time domain signal (i.e., the chirp time)
//     // By computing the difference between the values of the Fourier domain
//     // phase derivative at two frequencies.
//     // Here the starting frequency is an input i.e., fHzSt, converted to Geometric units MfSt
//     // and the ending frequency is fixed to be the frequency of the amplitude peak in Geometric units MfPeak
//     // XLALIMRPhenomDGetPeakFreq output is in Hz, covert to Mf via / M_sec
//     const double MfPeak = XLALIMRPhenomDGetPeakFreq(m1, m2, chi1, chi2) / M_sec;
//
//     // Compute phase derivative at starting frequency
//     const double dphifSt = PhenDPhaseDerivFrequencyPoint(MfSt, pPhi, pn);
//     // Compute phase derivative at ending (ringdown) frequency
//     const double dphifRD = PhenDPhaseDerivFrequencyPoint(MfPeak, pPhi, pn);
//     const double dphidiff = dphifRD - dphifSt;
//
//     // The length of time is estimated as dphidiff / 2 / pi * M (In units of seconds)
//     const double ChirpTimeSec = dphidiff / 2. / PI * M_sec;
//
//     LALFree(pPhi);
//     LALFree(pn);
//
//     return ChirpTimeSec;
//
// }

/**
 * Helper function used in PhenomHM and PhenomPv3HM
 * Returns the phenomD amplitude
 */
int IMRPhenomDAmpFrequencySequence(
    real_vector* amps,  /**< [out] phase evaluated at input freqs */
    real_vector* freqs, /**< Sequency of Geometric frequencies */
    IMRPhenomDAmplitudeCoefficients* pAmp, /**< IMRPhenomDAmplitudeCoefficients struct */
    AmpInsPrefactors* amp_prefactors, /**< AmpInsPrefactors struct */
    size_t ind_min,       /**< start index for frequency loop */
    size_t ind_max,       /**< end index for frequency loop */
    double  m1,             /**< mass of primary in solar masses */
    double  m2,             /**< mass of secondary in solar masses */
    double  chi1z,          /**< dimensionless aligned-spin of primary */
    double  chi2z           /**< dimensionless aligned-spin of secondary */
)
{
  int retcode = 0;

  int status_in_for = SUCCESS;
/* Now generate the waveform */
//#pragma omp parallel for
  for (size_t i = ind_min; i < ind_max; i++)
  {
    double  Mf = freqs->data[i]; // geometric frequency

    UsefulPowers powers_of_f;
    status_in_for = init_useful_powers(&powers_of_f, Mf);
    if (SUCCESS != status_in_for)
    {
      ERROR(ERROR_EFUNC, "init_useful_powers failed for Mf");
      retcode = status_in_for;
    }
    else
    {
      amps->data[i] = IMRPhenDAmplitude(Mf, pAmp, &powers_of_f, amp_prefactors);
    }
  }

  // free(pAmp);

  return SUCCESS;
}

/**
 * computes the time shift as the approximate time of the peak of the 22 mode.
 */
double  IMRPhenomDComputet0(
    double  eta,           /**< symmetric mass-ratio */
    double  chi1z,         /**< dimensionless aligned-spin of primary */
    double  chi2z,         /**< dimensionless aligned-spin of secondary */
    double  finspin        /**< final spin */
    // LALDict *extraParams /**< linked list containing the extra testing GR parameters */
)
{

  // if (extraParams == NULL)
  //   extraParams = XLALCreateDict();

  // pPhi = XLALMalloc(sizeof(IMRPhenomDPhaseCoefficients));
  IMRPhenomDPhaseCoefficients *pPhi = malloc(sizeof(IMRPhenomDPhaseCoefficients));
  ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1z, chi2z, finspin);// , extraParams);
  if (!pPhi)
    ERROR(ERROR_EFUNC, "");

  // pAmp = XLALMalloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  IMRPhenomDAmplitudeCoefficients *pAmp = malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1z, chi2z, finspin);
  if (!pAmp)
    ERROR(ERROR_EFUNC, "");

  // double Rholm = XLALSimIMRPhenomHMRholm(eta, chi1z, chi2z, ell, mm);
  // double Taulm = XLALSimIMRPhenomHMTaulm(eta, chi1z, chi2z, ell, mm);

  //time shift so that peak amplitude is approximately at t=0
  //For details see https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timedomain
  //NOTE: All modes will have the same time offset. So we use the 22 mode.
  //If we just use the 22 mode then we pass 1.0, 1.0 into DPhiMRD.
  const double t0 = DPhiMRD(pAmp->fmaxCalc, pPhi, 1.0, 1.0);

  free(pPhi);
  free(pAmp);

  return t0;
}

/**
* Function to return the final spin (spin of the remnant black hole)
* as predicted by the IMRPhenomD model. The final spin is calculated using
* the phenomenological fit described in PhysRevD.93.044006 Eq. 3.6.
* unreviewed
*/
double IMRPhenomDFinalSpin(
    const double m1_in,                 /**< mass of companion 1 [Msun] */
    const double m2_in,                 /**< mass of companion 2 [Msun] */
    const double chi1_in,               /**< aligned-spin of companion 1 */
    const double chi2_in               /**< aligned-spin of companion 2 */
) {
    // Ensure that m1 > m2 and that chi1 is the spin on m1
    double chi1, chi2, m1, m2;
    if (m1_in>m2_in) {
       chi1 = chi1_in;
       chi2 = chi2_in;
       m1   = m1_in;
       m2   = m2_in;
    } else { // swap spins and masses
       chi1 = chi2_in;
       chi2 = chi1_in;
       m1   = m2_in;
       m2   = m1_in;
    }

    const double M = m1 + m2;
    double eta = m1 * m2 / (M * M);

    if (eta > 0.25)
        PhenomInternal_nudge(&eta, 0.25, 1e-6);
    if (eta > 0.25 || eta < 0.0)
        ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");

    double finspin = FinalSpin0815(eta, chi1, chi2);

    // if (finspin < MIN_FINAL_SPIN)
    //       WARNING("Final spin and ISCO frequency of this system are small, \
    //                       the model might misbehave here.");

    return finspin;
}

/**
 * Helper function used in PhenomHM and PhenomPv3HM
 * Returns the final mass from the fit used in PhenomD
 */
double IMRPhenomDFinalMass(
    double m1,    /**< mass of primary in solar masses */
    double m2,    /**< mass of secondary in solar masses */
    double chi1z, /**< aligned-spin component on primary */
    double chi2z  /**< aligned-spin component on secondary */
)
{
  // int retcode = 0;
  // retcode = PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
  //     &m1,
  //     &m2,
  //     &chi1z,
  //     &chi2z);
  // CHECK(
  //     SUCCESS == retcode,
  //     XLAL_EFUNC,
  //     "PhenomInternal_AlignedSpinEnforcePrimaryIsm1 failed");
  // CHECK(
  //     m1>m2,
  //     ERROR_EDOM,
  //     "m1>m2 required.\n");
  double Mtot = m1 + m2;
  double eta = m1 * m2 / (Mtot * Mtot);

  if (eta > 0.25)
    PhenomInternal_nudge(&eta, 0.25, 1e-6);
  if (eta > 0.25 || eta < 0.0)
    ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");

  return (1.0 - EradRational0815(eta, chi1z, chi2z));
}

/**
 * Function to compute the amplitude and phase coefficients for PhenomD
 * Used to optimise the calls to IMRPhenDPhase and IMRPhenDAmplitude
 */
int IMRPhenomDSetupAmpAndPhaseCoefficients(
    PhenDAmpAndPhasePreComp *pDPreComp,
    double  m1,
    double  m2,
    double  chi1z,
    double  chi2z,
    const double  Rholm,
    const double  Taulm)
    // LALDict *extraParams)
{

  // Make a pointer to LALDict to circumvent a memory leak
  // At the end we will check if we created a LALDict in extraParams
  // and destroy it if we did.
  // LALDict *extraParams_in = extraParams;

  /* It's difficult to see in the code but you need to setup the
     * powers_of_pi.
     */
  int retcode = 0;
  retcode = init_useful_powers(&powers_of_pi, PI);
  CHECK(SUCCESS == retcode, retcode, "Failed to initiate useful powers of pi.");

  // PhenomInternal_AlignedSpinEnforcePrimaryIsm1(&m1, &m2, &chi1z, &chi2z);
  CHECK(m1 >= m2, ERROR_EINVAL, "Need m1 >= m2.");
  const double  Mtot = m1 + m2;
  const double  eta = m1 * m2 / (Mtot * Mtot);

  // Calculate phenomenological parameters
  const double  finspin = FinalSpin0815(eta, chi1z, chi2z); //FinalSpin0815 - 0815 is like a version number

  // if (finspin < MIN_FINAL_SPIN)
  //   WARNING("Final spin and ISCO frequency of this system are small, \
  //                           the model might misbehave here.");

  //start phase
  // if (extraParams == NULL)
  // {
  //   extraParams = XLALCreateDict();
  // }

  // XLALSimInspiralWaveformParamsInsertPNSpinOrder(extraParams, LAL_SIM_INSPIRAL_SPIN_ORDER_35PN);

  // IMRPhenomDPhaseCoefficients *pPhi;
  // pPhi = XLALMalloc(sizeof(IMRPhenomDPhaseCoefficients));
  IMRPhenomDPhaseCoefficients* pPhi = malloc(sizeof(IMRPhenomDPhaseCoefficients));
  ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi1z, chi2z, finspin);//, extraParams);
  if (!pPhi)
    ERROR(ERROR_EFUNC, "");
  PNPhasingSeries *pn = NULL;
  TaylorF2AlignedPhasing(&pn, m1, m2, chi1z, chi2z);//, extraParams);
  if (!pn)
    ERROR(ERROR_EFUNC, "");

  // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
  // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
  // was not available when PhenomD was tuned.
  double  testGRcor = 1.0;
  // testGRcor += XLALSimInspiralWaveformParamsLookupNonGRDChi6(extraParams);
  pn->v[6] -= (Subtract3PNSS(m1, m2, Mtot, eta, chi1z, chi2z) * pn->v[0]) * testGRcor;

  PhiInsPrefactors phi_prefactors;
  retcode = 0;
  retcode = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
  CHECK(SUCCESS == retcode, retcode, "init_phi_ins_prefactors failed");

  // Compute coefficients to make phase C^1 continuous (phase and first derivative)
  ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors, Rholm, Taulm);
  //end phase

  //start amp
  // IMRPhenomDAmplitudeCoefficients *pAmp;
  // pAmp = XLALMalloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  IMRPhenomDAmplitudeCoefficients* pAmp = malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
  ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi1z, chi2z, finspin);
  if (!pAmp)
    ERROR(ERROR_EFUNC, "");

  AmpInsPrefactors amp_prefactors;
  retcode = 0;
  retcode = init_amp_ins_prefactors(&amp_prefactors, pAmp);
  CHECK(SUCCESS == retcode, retcode, "init_amp_ins_prefactors failed");
  //end amp

  //output
  pDPreComp->pn = *pn;
  pDPreComp->pPhi = *pPhi;
  pDPreComp->phi_prefactors = phi_prefactors;

  pDPreComp->pAmp = *pAmp;
  pDPreComp->amp_prefactors = amp_prefactors;

  free(pn);
  free(pPhi);
  free(pAmp);

  /* If extraParams was allocated in this function and not passed in
  * we need to free it to prevent a leak */
  // if (extraParams && !extraParams_in) {
  //   XLALDestroyDict(extraParams);
  // } else {
  //   XLALSimInspiralWaveformParamsInsertPNSpinOrder(extraParams,LAL_SIM_INSPIRAL_SPIN_ORDER_ALL);
  // }

  return SUCCESS;
}

/**
 * Function to update the phase coefficients for PhenomD
 * To be called for the different modes with different values of Rholm, Taulm
 */
int IMRPhenomDUpdatePhaseCoefficients(
    PhenDAmpAndPhasePreComp *pDPreComp,
    double  m1,
    double  m2,
    double  chi1z,
    double  chi2z,
    const double  Rholm,
    const double  Taulm)
{
  IMRPhenomDPhaseCoefficients* pPhi = &(pDPreComp->pPhi);
  PNPhasingSeries* pn = &(pDPreComp->pn);
  PhiInsPrefactors* phi_prefactors = &(pDPreComp->phi_prefactors);

  // Compute coefficients to make phase C^1 continuous (phase and first derivative)
  ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, phi_prefactors, Rholm, Taulm);

  return SUCCESS;
}

/**
 * Function to return the phenomD phase using the
 * IMRPhenomDSetupAmpAndPhaseCoefficients struct
 */
double IMRPhenomDPhase_OneFrequency(
    double  Mf,
    PhenDAmpAndPhasePreComp pD,
    double  Rholm,
    double  Taulm,
    const ModGRParams* modgrparams)
{
  UsefulPowers powers_of_Mf;
  int status = init_useful_powers(&powers_of_Mf, Mf);
  CHECK(SUCCESS == status, status, "Failed to initiate init_useful_powers");
  double  phase = IMRPhenDPhase(Mf, &(pD.pPhi), &(pD.pn), &powers_of_Mf,
                              &(pD.phi_prefactors), Rholm, Taulm, modgrparams);
  return phase;
}

/**
 * Function to return the phenomD phase derivative using the
 * IMRPhenomDSetupAmpAndPhaseCoefficients struct
 */
double IMRPhenomDPhaseDerivative_OneFrequency(
    double Mf,
    PhenDAmpAndPhasePreComp pD,
    double Rholm,
    double Taulm,
    const ModGRParams* modgrparams)
{
  UsefulPowers powers_of_Mf;
  int status = init_useful_powers(&powers_of_Mf, Mf);
  CHECK(SUCCESS == status, status, "Failed to initiate init_useful_powers");
  double  phase = IMRPhenDPhaseDerivative(Mf, &(pD.pPhi), &(pD.pn), &powers_of_Mf,
                              &(pD.phi_prefactors), Rholm, Taulm, modgrparams);
  return phase;
}
