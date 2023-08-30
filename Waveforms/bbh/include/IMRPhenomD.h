#ifndef _LALSIM_IMR_PHENOMD_H
#define _LALSIM_IMR_PHENOMD_H

#include "IMRPhenomD_internals.h"
#include "IMRPhenomUtils.h"
#include "IMRPhenomInternalUtils.h"


int IMRPhenomDGenerateFD(
    COMPLEX16FrequencySeries **htilde,  /**< [out] FD waveform */
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
);

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
    const ExtraParams* extraparams,     /**< Additional parameters */
    const ModGRParams* modgrparams      /**< Modified GR parameters */
);

double IMRPhenomDComputeTimeOfFrequency(
    const double f,                     /**< Input frequency (Hz): we compute t(f) */
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
    const ExtraParams* extraparams,     /**< Additional parameters */
    const ModGRParams* modgrparams      /**< Modified GR parameters */
);

double IMRPhenomDComputeInverseFrequencyOfTime(
    const double t,                     /**< Input time (s): we compute f(t) */
    const double f_estimate,            /**< Estimate of f(t), use to initialize */
    const double t_acc,                 /**< Target accuracy of t(f) where to stop refining f */
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
    const int max_iter,                 /**< Maximal number of iterations in bisection */
    const ExtraParams* extraparams,     /**< Additional parameters */
    const ModGRParams* modgrparams      /**< Modified GR parameters */
);

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
    const double  Taulm);

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
    const double  Taulm);

/**
 * Function to return the phenomD phase using the
 * IMRPhenomDSetupAmpAndPhaseCoefficients struct
 */
double IMRPhenomDPhase_OneFrequency(
    double Mf,
    PhenDAmpAndPhasePreComp pD,
    double Rholm,
    double Taulm,
    const ModGRParams* modgrparams);

/**
 * Function to return the phenomD phase derivative using the
 * IMRPhenomDSetupAmpAndPhaseCoefficients struct
 */
double IMRPhenomDPhaseDerivative_OneFrequency(
    double Mf,
    PhenDAmpAndPhasePreComp pD,
    double Rholm,
    double Taulm,
    const ModGRParams* modgrparams);

#endif /* _LALSIM_IMR_PHENOMD_H */
