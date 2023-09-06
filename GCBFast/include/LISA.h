#ifndef LISA_h
#define LISA_h


/* Photon shot noise power */
#define Sps 8.321000e-23

/* Acceleration noise power */
#define Sacc 9.000000e-30

/* Mean arm length of constellation (m) */
#define Larm 2.5e9

 /* LISA orbital eccentricity */
#define ec 0.0048241852

 /* Initial azimuthal position of the guiding center */
#define kappa 0.000000

 /* Initial orientation of the LISA constellation */
#define lambda 0.000000

 /* LISA modulation frequency */
#define fm 3.168753575e-8

/* transfer frequency (Hz) */
#define fstar 0.01908538063694777

/* MLDC sampling rate */
//#define dt 15.000000

/* Observation period */
//#define Tobs 31457280.000000// 125829120.00000//31457280.000000


void instrument_noise(double f, double *SAE, double *SXYZ);

void spacecraft(double t, double *x, double *y, double *z);



#endif /* LISA_h */
