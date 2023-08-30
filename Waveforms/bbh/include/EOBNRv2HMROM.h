/**
 * \author Sylvain Marsat, University of Maryland - NASA GSFC
 *
 * \brief C header for EOBNRv2HM reduced order model (non-spinning version).
 * See CQG 31 195010, 2014, arXiv:1402.4146 for details on the reduced order method.
 * See arXiv:1106.1021 for the EOBNRv2HM model.
 *
 * Borrows from the SEOBNR ROM LAL code written by Michael Puerrer and John Veitch.
 *
 * Put the untared data in the directory designated by the environment variable ROM_DATA_PATH.
 *
 * Parameter range:
 *   q = 1-12 (almost)
 *   No spin
 *   Mtot >= 10Msun for fstart=8Hz
 *
 */

#ifndef _EOBNRV2HMROM_H
#define _EOBNRV2HMROM_H

#define _XOPEN_SOURCE 500

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <stdbool.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_complex.h>

#include "constants.h"
#include "struct.h"
#include "EOBNRv2HMROMstruct.h"

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

   /********General definitions********/

#define nk_amp 10  /* number of SVD-modes == number of basis functions for amplitude */
#define nk_phi 20  /* number of SVD-modes == number of basis functions for phase */

/********External array for the list of modes********/
#define nbmodemax 5
extern const int listmode[nbmodemax][2];

/**************************************************/
/**************** Prototypes **********************/

/* Functions to load, initalize and cleanup data */
int EOBNRv2HMROM_Init_DATA(void);
int EOBNRv2HMROM_Init(const char dir[]);

void EOBNRHMROMdata_Init(EOBNRHMROMdata **data);
void EOBNRHMROMdata_interp_Init(EOBNRHMROMdata_interp **data_interp);
void EOBNRHMROMdata_coeff_Init(EOBNRHMROMdata_coeff **data_coeff);

void EOBNRHMROMdata_Cleanup(EOBNRHMROMdata *data);
void EOBNRHMROMdata_interp_Cleanup(EOBNRHMROMdata_interp *data_interp);
void EOBNRHMROMdata_coeff_Cleanup(EOBNRHMROMdata_coeff *data_coeff);

/* Function to read data */
int Read_Data_Mode(const char dir[], const int mode[2], EOBNRHMROMdata *data);

/* Functions to interpolate the data and to evaluate the interpolated data for a given q */

int Evaluate_Spline_Data(
  const double q,                            /* Input: q-value for which projection coefficients should be evaluated */
  const EOBNRHMROMdata_interp* data_interp,  /* Input: data in interpolated form */
  EOBNRHMROMdata_coeff* data_coeff           /* Output: vectors of projection coefficients and shifts in time and phase */
);

int Interpolate_Spline_Data(
  const EOBNRHMROMdata *data,           /* Input: data in vector/matrix form to interpolate */
  EOBNRHMROMdata_interp *data_interp    /* Output: interpolated data */
);

/* Functions for waveform reconstruction */

int EOBNRv2HMROMCore(
  ListAmpPhaseFDMode** listhlm,
  int nbmode,
  double tRef,
  double phiRef,
  double fRef,
  double Mtot_sec,
  double q,
  double distance,
  int setphiRefatfRef);

int SimEOBNRv2HMROM(
  ListAmpPhaseFDMode** listhlm,                  /* Output: list of modes in Frequency-domain amplitude and phase form */
  int nbmode,                                    /* Number of modes to generate (starting with the 22) */
  double tRef,                                   /* Time shift with respect to the 22-fit removed waveform (s) */
  double phiRef,                                 /* Phase at reference frequency */
  double fRef,                                   /* Reference frequency (Hz); 0 defaults to fLow */
  double m1SI,                                   /* Mass of companion 1 (kg) */
  double m2SI,                                   /* Mass of companion 2 (kg) */
  double distance,                               /* Distance of source (m) */
  int setphiRefatfRef);                          /* Flag for adjusting the FD phase at phiRef at the given fRef, which depends also on tRef - if false, treat phiRef simply as an orbital phase shift (minus an observer phase shift) */

int SimEOBNRv2HMROMExtTF2(
  ListAmpPhaseFDMode** listhlm,                  /* Output: list of modes in Frequency-domain amplitude and phase form */
  int nbmode,                                    /* Number of modes to generate (starting with the 22) */
  double Mf_match,                               /* Minimum frequency using EOBNRv2HMROM in inverse total mass units*/
  double minf,                                   /* Minimum frequency required */
  int tagexthm,                                  /* Tag to decide whether or not to extend the higher modes as well */
  double deltatRef,                              /* Time shift so that the peak of the 22 mode occurs at deltatRef */
  double phiRef,                                 /* Phase at reference frequency */
  double fRef,                                   /* Reference frequency (Hz); 0 defaults to fLow */
  double m1SI,                                   /* Mass of companion 1 (kg) */
  double m2SI,                                   /* Mass of companion 2 (kg) */
  double distance,                               /* Distance of source (m) */
  int setphiRefatfRef);                          /* Flag for adjusting the FD phase at phiRef at the given fRef, which depends also on tRef - if false, treat phiRef simply as an orbital phase shift (minus an observer phase shift) */

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _EOBNRV2HMROM_H */
