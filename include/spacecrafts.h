// Copyright (C) 2023 En-Kun Li (GWSpace modifications)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SPACECRAFTS_H
#define SPACECRAFTS_H

#include <math.h>
#include "constants.h"


/* Photon shot noise power */
#define Sps 8.321000e-23

/* Acceleration noise power */
#define Sacc 9.000000e-30


/* LISA modulation frequency */
// #define fm 3.168753575e-8


// parameters for LISA spacecrafts
#define Omega_lisa 1.99098659277e-7  // 2Pi/YRSID_SI
/* Mean arm length of constellation (m) */
#define armLength_lisa 2.5e9  
// #define Radius_lisa  AU
/* LISA orbital eccentricity */
#define ecc_lisa 0.004824185218078991 // L/(2 R sqrt{3})
/* Initial azimuthal position of the guiding center */
#define kappa_lisa -0.3490658503988659 // 20 deg behind earth (consider the Perihelion_Ang according earth)
/* Initial orientation of the LISA constellation */
#define lambda_lisa 0.0
/* transfer frequency (Hz) */ // C_SI/(2PI armL)
#define fstar_lisa 0.01908538063694777

// parameters for TaiJi spacecrafts
#define Omega_tj 1.99098659277e-7 // 2Pi/YRSID_SI
#define armLength_tj 3.e9
#define ecc_tj 0.0057890222616947895 // L/(2R sqrt{3})
#define kappa_tj 0.3490658503988659 // 20 deg before earth (consider the Perihelion_Ang according earth)
#define lambda_tj 0.0
#define fstar_tj 0.01590448386412314

// parameters for tianqin spacecrafts
// Kepler angular frequency, using the same physical inputs as TianQinOrbit.f_0.
#define Omega_tq sqrt(G_SI * EarthMass / (Radius_tq * Radius_tq * Radius_tq))
#define armLength_tq 1.7320508075688772e8
#define Radius_tq TianQinOrbitRadius_SI
#define kappa_tq 0.0
#define lambda_tq 0.00
#define fstar_tq 0.27547374120820667 // c/(2pi L_tq)

/* MLDC sampling rate */
//#define dt 15.000000

/* Observation period */
//#define Tobs 31457280.000000// 125829120.00000//31457280.000000

void spacecraft(char detector[], int N, double *t, double *x, double *y, double *z, double *Larm);
void spacecraft_LISA(double t, double *x, double *y, double *z);
void spacecraft_TaiJi(double t, double *x, double *y, double *z);
void spacecraft_TianQin(double t, double *x, double *y, double *z);


#endif /* SPACECRAFTS_H */
