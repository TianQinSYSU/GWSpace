#ifndef _LALSIM_IMR_PHENOMD_H
#define _LALSIM_IMR_PHENOMD_H

#include "IMRPhenomD_internals.h"

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
    RealVector* freq,                   /**< Input: frequencies (Hz) on which to evaluate h22 FD - will be copied in the output AmpPhaseFDWaveform. Frequencies exceeding max freq covered by PhenomD will be given 0 amplitude and phase. */
    const double phi0,                  /**< Orbital phase at fRef (rad) */
    const double fRef_in,               /**< reference frequency (Hz) */
    const double m1_SI,                 /**< Mass of companion 1 (kg) */
    const double m2_SI,                 /**< Mass of companion 2 (kg) */
    const double chi1,                  /**< Aligned-spin parameter of companion 1 */
    const double chi2,                  /**< Aligned-spin parameter of companion 2 */
    const double distance               /**< Distance of source (m) */
);

#endif /* _LALSIM_IMR_PHENOMD_H */
