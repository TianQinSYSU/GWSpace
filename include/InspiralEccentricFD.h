// Created by Han Wang on 2022/3/23.

#ifndef _INSPIRALECCENTRICFD_H
#define _INSPIRALECCENTRICFD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

// From LALConstants.h
#define PI        3.141592653589793238462643383279502884
#define TWOPI     6.283185307179586476925286766559005768
#define GAMMA     0.577215664901532860606512090082402431

/* Solar mass, kg */
#define MSUN_SI 1.988546954961461467461011951140572744e30
/* Geometrized solar mass, s */
#define MTSUN_SI 4.925491025543575903411922162094833998e-6
/* Speed of light in vacuum, m s^-1 */
#define C_SI 299792458e0
/* Parsec, m */
#define PC_SI 3.085677581491367278913937957796471611e16

///////////////////////////////////////////////////////////////////////////////

typedef struct tagComplex16FDWaveform {
    double complex * data_p;
    double complex * data_c;
    double deltaF;
    size_t length;
} Complex16FDWaveform;

Complex16FDWaveform* CreateComplex16FDWaveform(
        double deltaF,
        size_t length
);

void DestroyComplex16FDWaveform(Complex16FDWaveform* wf);

typedef struct tagAmpPhaseFDWaveform {
    double complex * amp_p;
    double complex * amp_c;
    double * phase;
    double deltaF;
    size_t length;
    unsigned int harmonic;  // : 4
} AmpPhaseFDWaveform;

AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(
        double deltaF,
        size_t length,
        unsigned int harmonic
);

void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform* wf);

typedef enum {
    PD_SUCCESS = 0,      /**< PD_SUCCESS return value (not an error number) */
    PD_FAILURE = -1,     /**< Failure return value (not an error number) */
    PD_EDOM = 33,        /**< Input domain error */
    PD_EFAULT = 14,      /**< Invalid pointer */
    PD_EFUNC = 1024,     /**< Internal function call failed bit: "or" this with existing error number */
    PD_ENOMEM = 12,      /**< Memory allocation error */
} ERROR_type;

const char *ErrorString(int code);
void ERROR(ERROR_type e, char *errstr);

///////////////////////////////////////////////////////////////////////////////

int SimInspiralEccentricFD(Complex16FDWaveform **htilde,
                           double phiRef,
                           double deltaF,
                           double m1_SI,
                           double m2_SI,
                           double fStart,
                           double fEnd,
                           double i,
                           double r,
                           double inclination_azimuth,
                           double e_min,
                           bool space_cutoff);

int SimInspiralEccentricFDAmpPhase(AmpPhaseFDWaveform ***h_amp_phase,
                                   double phiRef,
                                   double deltaF,
                                   double m1_SI,
                                   double m2_SI,
                                   double fStart,
                                   double fEnd,
                                   double i,
                                   double r,
                                   double inclination_azimuth,
                                   double e_min,
                                   bool space_cutoff);

int SimInspiralEccentricFDAndPhase(AmpPhaseFDWaveform ***h_and_phase,
                                   double phiRef,
                                   double deltaF,
                                   double m1_SI,
                                   double m2_SI,
                                   double fStart,
                                   double fEnd,
                                   double i,
                                   double r,
                                   double inclination_azimuth,
                                   double e_min,
                                   bool space_cutoff);

#endif //_INSPIRALECCENTRICFD_H
