#ifndef _IMRPHENOMHM_H
#define _IMRPHENOMHM_H

// #include <lal/LALDatatypes.h>
// #include <lal/Sequence.h>
// #include <lal/LALDict.h>
// #include <lal/LALConstants.h>
// #include <lal/XLALError.h>
// #include <lal/FrequencySeries.h>
// #include <math.h>

#include "struct.h"
#include "constants.h"
#include "IMRPhenomInternalUtils.h"
#include "IMRPhenomUtils.h"
#include "RingdownCW.h"
#include "IMRPhenomD_internals.h"

#ifdef __cplusplus
extern "C" {
#endif

// #ifdef __GNUC__
// #define UNUSED __attribute__((unused))
// #else
// #define UNUSED
// #endif

/**
 * Dimensionless frequency (Mf) at which define the end of the waveform
 */
#define PHENOMHM_DEFAULT_MF_MAX 0.5

/**
 * eta is the symmetric mass-ratio.
 * This number corresponds to a mass-ratio of 20
 * The underlying PhenomD model was calibrated to mass-ratio 18
 * simulations. We choose mass-ratio 20 as a conservative
 * limit on where the model should be accurate.
 */
#define MAX_ALLOWED_ETA 0.045351

/**
 * Maximum number of (l,m) mode paris PhenomHM models.
 * Only count positive 'm' modes.
 * Used to set default mode array
 */
#define NMODES_MAX 6

/**
 * Highest ell multipole PhenomHM models + 1.
 * Used to set array sizes
 */
#define L_MAX_PLUS_1 5

/* Activates amplitude part of the model */
#define AmpFlagTrue 1
#define AmpFlagFalse 0

// LALDict *IMRPhenomHM_setup_mode_array(
//     LALDict *extraParams);

/**
 * useful powers in GW waveforms: 1/6, 1/3, 2/3, 4/3, 5/3, 2, 7/3, 8/3, -1, -1/6, -7/6, -1/3, -2/3, -5/3
 * calculated using only one invocation of 'pow', the rest are just multiplications and divisions
 */
typedef struct tagPhenomHMUsefulPowers
{
    double third;
    double two_thirds;
    double four_thirds;
    double five_thirds;
    double two;
    double seven_thirds;
    double eight_thirds;
    double inv;
    double m_seven_sixths;
    double m_third;
    double m_two_thirds;
    double m_five_thirds;
} PhenomHMUsefulPowers;

/**
  * Useful powers of Mf: 1/6, 1/3, 2/3, 4/3, 5/3, 2, 7/3, 8/3, -7/6, -5/6, -1/2, -1/6, 1/2
  * calculated using only one invocation of 'pow' and one of 'sqrt'.
  * The rest are just multiplications and divisions.  Also including Mf itself in here.
  */
typedef struct tagPhenomHMUsefulMfPowers
{
    double itself;
    double sixth;
    double third;
    double two_thirds;
    double four_thirds;
    double five_thirds;
    double two;
    double seven_thirds;
    double eight_thirds;
    double m_seven_sixths;
    double m_five_sixths;
    double m_sqrt;
    double m_sixth;
    double sqrt;
} PhenomHMUsefulMfPowers;

/**
 * must be called before the first usage of *p
 */
int PhenomHM_init_useful_mf_powers(PhenomHMUsefulMfPowers *p, double number);

/**
 * must be called before the first usage of *p
 */
int PhenomHM_init_useful_powers(PhenomHMUsefulPowers *p, double number);

/**
  * Structure holding Higher Mode Phase pre-computations
  */
typedef struct tagHMPhasePreComp
{
    double ai;
    double bi;
    double am;
    double bm;
    double ar;
    double br;
    double fi;
    double fr;
    double PhDBconst;
    double PhDCconst;
    double PhDBAterm;
} HMPhasePreComp;

/**
 * Structure storing pre-determined quantities
 * that describe the frequency array
 * and tells you over which indices will contain non-zero values.
 */
typedef struct tagPhenomHMFrequencyBoundsStorage
{
    //double deltaF;
    double f_min;
    double f_max;
    double f_ref;
    int freq_is_uniform; /**< If = 1 then assume uniform spaced, If = 0 then assume arbitrarily spaced. */
    size_t npts;           /**< number of points in waveform array */
    size_t ind_min;        /**< start index containing non-zero values */
    size_t ind_max;        /**< end index containing non-zero values */
} PhenomHMFrequencyBoundsStorage;

int init_IMRPhenomHMGet_FrequencyBounds_storage(
    PhenomHMFrequencyBoundsStorage* p,
    real_vector* freqs,
    double Mtot,
    //double deltaF,
    double f_ref_in);

int IMRPhenomHM_is_freq_uniform(
    real_vector* freqs,
    double deltaF);

/**
 * Structure storing pre-determined quantities
 * complying to the conventions of the PhenomHM model.
 * convensions such as m1>=m2
 */
typedef struct tagPhenomHMStorage
{
    double m1;    /**< mass of larger body in solar masses */
    double m2;    /**< mass of lighter body in solar masses */
    double m1_SI; /**< mass of larger body in kg */
    double m2_SI; /**< mass of lighter body in kg */
    double Mtot;  /**< total mass in solar masses */
    double eta;   /**< symmetric mass-ratio */
    double chi1z; /**< dimensionless aligned component spin of larger body */
    double chi2z; /**< dimensionless aligned component spin of lighter body */
    double distance; /**< Luminosity distance to the source (Mpc) */
    double Ms;     /**< Total mass in seconds */
    double amp0;   /**< Amplitude prefactor to go back to physical strain */
    real_vector* freqs;
    double deltaF;
    double f_min;
    double f_max;
    double f_ref;
    double Mf_ref; /**< reference frequnecy in geometric units */
    double phiRef;
    int freq_is_uniform; /**< If = 1 then assume uniform spaced, If = 0 then assume arbitrarily spaced. */
    size_t npts;           /**< number of points in waveform array */
    size_t ind_min;        /**< start index containing non-zero values */
    size_t ind_max;        /**< end index containing non-zero values */
    double finmass;
    double finspin;
    double Mf_RD_22;
    double Mf_DM_22;
    double PhenomHMfring[L_MAX_PLUS_1][L_MAX_PLUS_1];
    double PhenomHMfdamp[L_MAX_PLUS_1][L_MAX_PLUS_1];
    double Rholm[L_MAX_PLUS_1][L_MAX_PLUS_1]; /**< ratio of (2,2) mode to (l,m) mode ringdown frequency */
    double Taulm[L_MAX_PLUS_1][L_MAX_PLUS_1]; /**< ratio of (l,m) mode to (2,2) mode damping time */
} PhenomHMStorage;

// static int init_PhenomHM_Storage(
//     PhenomHMStorage* p,   /**< [out] PhenomHMStorage struct */
//     const double m1_SI,    /**< mass of companion 1 (kg) */
//     const double m2_SI,    /**< mass of companion 2 (kg) */
//     const double chi1z,    /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
//     const double chi2z,    /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
//     real_vector* freqs, /**< Input frequency sequence (Hz) */
//     const double deltaF,   /**< frequency spacing (Hz) */
//     const double f_ref,    /**< reference GW frequency (hz) */
//     const double phiRef    /**< orbital phase at f_ref */
// );

// int IMRPhenomHMFDAddMode(
//     COMPLEX16FrequencySeries *hptilde,
//     COMPLEX16FrequencySeries *hctilde,
//     COMPLEX16FrequencySeries *hlmtilde,
//     double theta,
//     double phi,
//     int l,
//     int m,
//     int sym);

double IMRPhenomHMTrd(
    double Mf,
    double Mf_RD_22,
    double Mf_RD_lm,
    const int AmpFlag,
    const int ell,
    const int mm,
    PhenomHMStorage* pHM);

double IMRPhenomHMTi(
    double Mf,
    const int mm);

int IMRPhenomHMSlopeAmAndBm(
    double *Am,
    double *Bm,
    const int mm,
    double fi,
    double fr,
    double Mf_RD_22,
    double Mf_RD_lm,
    const int AmpFlag,
    const int ell,
    PhenomHMStorage* pHM);

int IMRPhenomHMMapParams(
    double *a,
    double *b,
    double flm,
    double fi,
    double fr,
    double Ai,
    double Bi,
    double Am,
    double Bm,
    double Ar,
    double Br);

int IMRPhenomHMFreqDomainMapParams(
    double *a,
    double *b,
    double *fi,
    double *fr,
    double *f1,
    const double flm,
    const int ell,
    const int mm,
    PhenomHMStorage* pHM,
    const int AmpFlag);

double IMRPhenomHMFreqDomainMap(
    double Mflm,
    const int ell,
    const int mm,
    PhenomHMStorage* pHM,
    const int AmpFlag);

int IMRPhenomHMPhasePreComp(
    HMPhasePreComp *q,
    const int ell,
    const int mm,
    PhenomHMStorage* pHM,
    PhenDAmpAndPhasePreComp* pDPreComp,
    const ModGRParams* modgrparams);
    // LALDict *extraParams);

double complex IMRPhenomHMOnePointFiveSpinPN(
    double fM,
    int l,
    int m,
    double M1,
    double M2,
    double X1z,
    double X2z);

// int IMRPhenomHMCore(
//     COMPLEX16FrequencySeries **hptilde,
//     COMPLEX16FrequencySeries **hctilde,
//     real_vector* freqs,
//     double m1_SI,
//     double m2_SI,
//     double chi1z,
//     double chi2z,
//     const double distance,
//     const double inclination,
//     const double phiRef,
//     const double deltaF,
//     double f_ref,
//     LALDict *extraParams);

int IMRPhenomHMEvaluateOnehlmMode(
    AmpPhaseFDMode* hlm,
    real_vector* amps,
    real_vector* phases,
    real_vector* tfs,
    real_vector* freqs_geom,
    PhenomHMStorage* pHM,
    PhenDAmpAndPhasePreComp* pDPreComp,
    int ell,
    int mm,
    double time_shift,
    double phi_shift,
    const ModGRParams* modgrparams);
    // double phi0);
    // LALDict *extraParams);

int IMRPhenomHMAmplitude(
    real_vector* amps,
    real_vector* freqs_geom,
    PhenomHMStorage* pHM,
    PhenDAmpAndPhasePreComp* pDPreComp,
    int ell,
    int mm);
    // LALDict *extraParams);

int IMRPhenomHMPhase(
    real_vector* phases,
    real_vector* tfs,
    real_vector* freqs_geom,
    PhenomHMStorage* pHM,
    PhenDAmpAndPhasePreComp* pDPreComp,
    int ell,
    int mm,
    const ModGRParams* modgrparams);
    // LALDict *extraParams);

int IMRPhenomHMAmpPNScaling(
    real_vector* amps,
    real_vector* freqs_amp,
    real_vector* freqs_geom,
    double m1,
    double m2,
    double chi1z,
    double chi2z,
    int ell,
    int mm);

int IMRPhenomHMGetRingdownFrequency(
    double *fringdown,
    double *fdamp,
    int ell,
    int mm,
    double finalmass,
    double finalspin);

/**
 * XLAL function that returns
 * a SphHarmFrequencySeries object
 * containing all the hlm modes
 * requested.
 * These have the correct relative phases between modes.
 * Note this has a similar interface to XLALSimIMRPhenomHM
 * because it is designed so it can be used independently.
 */
int IMRPhenomHMGethlmModes(
    ListAmpPhaseFDMode** hlms,  /**< [out] list of modes, FD amp/phase */
    double* fpeak,              /**< [out] Approximate 22 peak frequency (Hz) */
    double* tpeak,              /**< [out] tf 22 at peak frequency (s) */
    double* phipeak,            /**< [out] phase 22 at peak frequency */
    double* fstart,             /**< [out] Starting frequency (Hz) */
    double* tstart,             /**< [out] tf 22 at starting frequency (s) */
    double* phistart,           /**< [out] phase 22 at starting frequency */
    real_vector* freq_22,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_21,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_33,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_32,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_44,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_43,        /**< [in] frequency vector for lm in Hz */
    double m1,                   /**< primary mass [solar masses] */
    double m2,                   /**< secondary mass [solar masses] */
    double chi1z,                   /**< aligned spin of primary */
    double chi2z,                   /**< aligned spin of secondary */
    double distance,                /**< luminosity distance (Mpc) */
    //const double deltaF,            /**< frequency spacing */
    const double phiRef,            /**< orbital phase at f_ref */
    const double fRef_in,                   /**< reference GW frequency */
    const double Deltat,             /**< Time shift (s) applied a posteriori */
    const int scale_freq_hm,         /**< Scale mode freq by m/2 */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
    //LALDict *extraParams           /**< LALDict struct */
);

/* NOTE: for now ugly input/output by hand for each mode */
int IMRPhenomHMComputeTimeOfFrequencyModeByMode(
    double* tf22,             /**< [out] value of t_22 (s) */
    double* tf21,             /**< [out] value of t_21 (s) */
    double* tf33,             /**< [out] value of t_33 (s) */
    double* tf32,             /**< [out] value of t_32 (s) */
    double* tf44,             /**< [out] value of t_44 (s) */
    double* tf43,             /**< [out] value of t_43 (s) */
    double f22,              /**< [in] value of f_22 (Hz) */
    double f21,              /**< [in] value of f_21 (Hz) */
    double f33,              /**< [in] value of f_33 (Hz) */
    double f32,              /**< [in] value of f_32 (Hz) */
    double f44,              /**< [in] value of f_44 (Hz) */
    double f43,              /**< [in] value of f_43 (Hz) */
    double m1,                   /**< primary mass [solar masses] */
    double m2,                   /**< secondary mass [solar masses] */
    double chi1z,                   /**< aligned spin of primary */
    double chi2z,                   /**< aligned spin of secondary */
    double distance,                /**< luminosity distance (Mpc) */
    //const double deltaF,            /**< frequency spacing */
    const double phiRef,            /**< orbital phase at f_ref */
    const double fRef_in,                   /**< reference GW frequency */
    const double Deltat,             /**< Time shift (s) applied a posteriori */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
    //LALDict *extraParams           /**< LALDict struct */
);
/* NOTE: for now ugly input/output by hand for each mode */
int IMRPhenomHMComputeInverseFrequencyOfTimeModeByMode(
    double* f22,              /**< [out] value of f_22 (Hz) */
    double* f21,              /**< [out] value of f_21 (Hz) */
    double* f33,              /**< [out] value of f_33 (Hz) */
    double* f32,              /**< [out] value of f_32 (Hz) */
    double* f44,              /**< [out] value of f_44 (Hz) */
    double* f43,              /**< [out] value of f_43 (Hz) */
    double tf22,             /**< [in] value of t_22 (s) */
    double tf21,             /**< [in] value of t_21 (s) */
    double tf33,             /**< [in] value of t_33 (s) */
    double tf32,             /**< [in] value of t_32 (s) */
    double tf44,             /**< [in] value of t_44 (s) */
    double tf43,             /**< [in] value of t_43 (s) */
    double f22_estimate,     /**< [in] guess for the value of f22, will be scaled by m/2 */
    double t_acc,                 /**< Target accuracy of t(f) where to stop refining f */
    double m1,                   /**< primary mass [solar masses] */
    double m2,                   /**< secondary mass [solar masses] */
    double chi1z,                   /**< aligned spin of primary */
    double chi2z,                   /**< aligned spin of secondary */
    double distance,                /**< luminosity distance (Mpc) */
    //const double deltaF,            /**< frequency spacing */
    const double phiRef,            /**< orbital phase at f_ref */
    const double fRef_in,                   /**< reference GW frequency */
    const double Deltat,             /**< Time shift (s) applied a posteriori */
    const int max_iter,                 /**< Maximal number of iterations in bisection */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
    //LALDict *extraParams           /**< LALDict struct */
);

#ifdef __cplusplus
}
#endif

#endif /* _IMRPHENOMHM_H */
