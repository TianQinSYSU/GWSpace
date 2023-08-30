#ifndef _STRUCT_H
#define _STRUCT_H

/*
 * Copyright (C) 2019 Sylvain Marsat
 *
 */

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <string.h>

//Looks like these aren't used...
#include <gsl/gsl_errno.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_complex.h>

/******************************************************************************/
/* Handling errors */
/******************************************************************************/

/* Note: values inspired from unix errno.h, except EFUNC */
typedef enum {
    ERROR_EINVAL = 22,      /**< Invalid argument */
    ERROR_EDOM   = 33,      /**< Input domain error */
    ERROR_EFAULT = 14,      /**< Invalid pointer */
    ERROR_EFUNC  = 1024,    /**< Internal function call failed */
    ERROR_ENOMEM = 12       /**< Memory allocation error */
} ERROR_type;

#define ERROR(...) \
	RaiseError(__func__, __FILE__, __LINE__, __VA_ARGS__)
#define ERRORCODE(errorcode, ...) \
	{RaiseErrorCode(__func__, __FILE__, __LINE__, errorcode, __VA_ARGS__); \
  return errorcode;}
#define WARNING(...) \
	RaiseWarning(__func__, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(...) \
	RaiseInfo(__func__, __FILE__, __LINE__, __VA_ARGS__)
#define CHECK(...) \
	RaiseCheck(__func__, __FILE__, __LINE__, __VA_ARGS__)
#define CHECKP(...) \
	PrintCheck(__func__, __FILE__, __LINE__, __VA_ARGS__)
const char *ErrorString(int code);
void RaiseError(const char *funcstr, const char *filestr, const int line,
                ERROR_type e, char *errstr, ...);
void RaiseErrorCode(const char *funcstr, const char *filestr, const int line,
                ERROR_type e, char *errstr, ...);
void RaiseWarning(const char *funcstr, const char *filestr, const int line,
                  char *warningstr, ...);
void RaiseInfo(const char *funcstr, const char *filestr, const int line,
               char *infostr, ...);
void RaiseCheck(const char *funcstr, const char *filestr, const int line,
                bool assertion, ERROR_type e, char *errstr);
void PrintCheck(const char *funcstr, const char *filestr, const int line,
                bool assertion, char *errstr);

/* GSL error handler */
void GSL_Err_Handler(
  const char *reason,
  const char *file,
  int line,
  int gsl_errno);

/******************************************************************************/
/* Types for functions */
/******************************************************************************/

typedef double (*RealFunction)(double);
typedef double (*RealObjectFunction)(const void *, double);

/******************************************************************************/
/* I/O functions for GSL structures */
/******************************************************************************/

/* Functions to read/write binary data from files */
int Read_GSL_Vector(const char dir[], const char fname[], gsl_vector *v);
int Read_GSL_Matrix(const char dir[], const char fname[], gsl_matrix *m);
int Write_GSL_Vector(const char dir[], const char fname[], gsl_vector *v);
int Write_GSL_Matrix(const char dir[], const char fname[], gsl_matrix *m);

/* Functions to read/write ascii data from files */
int Read_GSL_Text_Vector(const char dir[], const char fname[], gsl_vector *v);
int Read_GSL_Text_Matrix(const char dir[], const char fname[], gsl_matrix *m);
int Write_GSL_Text_Vector(const char dir[], const char fname[], gsl_vector *v);
int Write_GSL_Text_Matrix(const char dir[], const char fname[], gsl_matrix *m);

/******************************************************************************/
/* Basic structures */
/******************************************************************************/

typedef struct tag_real_vector {
    double* data;
    size_t size;
} real_vector;

typedef struct tag_complex_vector {
    double complex* data;
    size_t size;
} complex_vector;

typedef struct tag_real_matrix {
    double* data;
    size_t size1;
    size_t size2;
} real_matrix;

typedef struct tag_complex_array_3d {
    double complex* data;
    size_t size1;
    size_t size2;
    size_t size3;
} complex_array_3d;

/******************************************************************************/
/* Waveform structures */
/******************************************************************************/

typedef struct tagAmpPhaseFDWaveform {
    double* freq;
    double* amp;
    double* phase;
    size_t length;
} AmpPhaseFDWaveform;

typedef struct tagAmpPhaseFDMode {
    real_vector* freq_amp;
    real_vector* amp_real;
    real_vector* amp_imag;
    real_vector* freq_phase;
    real_vector* phase;
    real_vector* tf;
} AmpPhaseFDMode;

typedef struct tagCAmpPhaseFDData {
    real_vector* freq;
    real_vector* amp_real;
    real_vector* amp_imag;
    real_vector* phase;
} CAmpPhaseFDData;

/* NOTE: splines here are implicitly cubic splines, so n*5 matrices */
typedef struct tagCAmpPhaseFDSpline {
    real_matrix* spline_amp_real;
    real_matrix* spline_amp_imag;
    real_matrix* spline_phase;
} CAmpPhaseFDSpline;

/******************************************************************************/
/* List structures for waveforms */
/******************************************************************************/

/* List structure, for a list of modes, each in amplitude and phase form */
typedef struct tagListAmpPhaseFDMode
{
  AmpPhaseFDMode*               hlm; /* The FD mode with amplitude and phase */
  int                           l; /* Mode number l  */
  int                           m; /* Mode number m  */
  struct tagListAmpPhaseFDMode* next; /* Pointer to next item in the list */
} ListAmpPhaseFDMode;

/*******************************************************************************/
/* class type of detector spacecraft orbits and vector                         */
/* ekli:                                                                       */
/*******************************************************************************/
typedef struct tagSpaceCraftOrbit
{
    char *detector;

} SpaceCraftOrbit;

/******************************************************************************/
/* Functions for basic structures */
/******************************************************************************/

real_vector* real_vector_alloc(size_t size);
void real_vector_free(real_vector* v);
real_vector* real_vector_view(double* data, size_t size);
real_vector* real_vector_view_subvector(
  real_vector* v,
  size_t istart,
  size_t iend);
void real_vector_view_free(real_vector* v);
real_vector* real_vector_resize(real_vector *v, size_t istart, size_t iend);
inline double real_vector_get(real_vector* v, size_t i);
inline void real_vector_set(real_vector* v, size_t i, double val);
void real_vector_set_zero(real_vector* vec);
void real_vector_scale_constant(real_vector* vec, const double s);
void real_vector_add_constant(real_vector* vec, const double a);
void real_vector_copy_gsl_vector(real_vector* vec, gsl_vector* gslvec);
void real_vector_from_gsl_vector(real_vector** vec, gsl_vector* gslvec);
void gsl_vector_from_real_vector(gsl_vector** gslvec, real_vector* vec);

complex_vector* complex_vector_alloc(size_t size);
void complex_vector_free(complex_vector* v);
complex_vector* complex_vector_view(double complex* data, size_t size);
complex_vector* complex_vector_view_subvector(
  complex_vector* v,
  size_t istart,
  size_t iend);
void complex_vector_view_free(complex_vector* v);
complex_vector* complex_vector_resize(complex_vector *v,
                                      size_t istart, size_t iend);
inline double complex complex_vector_get(complex_vector* v, size_t i);
inline void complex_vector_set(complex_vector* v, size_t i, double complex val);

real_matrix* real_matrix_alloc(size_t size1, size_t size2);
void real_matrix_free(real_matrix* m);
real_matrix* real_matrix_view(double* data, size_t size1, size_t size2);
void real_matrix_view_free(real_matrix *m);
inline double real_matrix_get(real_matrix* m, size_t i, size_t j);
inline double* real_matrix_line(real_matrix* m, size_t i);
inline void real_matrix_set(real_matrix* v, size_t i, size_t j, double val);

complex_array_3d* complex_array_3d_alloc(size_t size1, size_t size2, size_t size3);
void complex_array_3d_free(complex_array_3d* a);
complex_array_3d* complex_array_3d_view(double complex* data, size_t size1, size_t size2, size_t size3);
void complex_array_3d_view_free(complex_array_3d *a);
inline double complex complex_array_3d_get(complex_array_3d* a, size_t i, size_t j, size_t k);
inline double complex* complex_array_3d_line(complex_array_3d* a, size_t i);
inline void complex_array_3d_set(complex_array_3d* v, size_t i, size_t j, size_t k, double complex val);

/******************************************************************************/
/* Functions for waveform structures */
/******************************************************************************/

AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(
  size_t length
);
void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform* wf);

int AmpPhaseFDMode_Init(
  AmpPhaseFDMode** hlm,
  size_t length_amp,
  size_t length_phase
);
void AmpPhaseFDMode_Destroy(AmpPhaseFDMode* wf);

int CAmpPhaseFDData_Init(
  CAmpPhaseFDData** h,
  size_t length
);
void CAmpPhaseFDData_Destroy(CAmpPhaseFDData* h);
// CAmpPhaseFDData* CAmpPhaseFDData_view(
//   real_vector* freq,
//   real_vector* amp_real,
//   real_vector* amp_imag,
//   real_vector* phase
// );
// void CAmpPhaseFDData_view_free(CAmpPhaseFDData* h);

int CAmpPhaseFDSpline_Init(
  CAmpPhaseFDSpline** h,
  size_t length
);
void CAmpPhaseFDSpline_Destroy(CAmpPhaseFDSpline* h);

/******************************************************************************/
/* Functions for list structures */
/******************************************************************************/

/* Add mode to list, directly without copying the data */
ListAmpPhaseFDMode* ListAmpPhaseFDMode_AddMode(
	   ListAmpPhaseFDMode* list,  /* List structure to prepend to */
	   AmpPhaseFDMode* hlm,  /* Data for the mode */
	   int l, /*< Mode number l */
	   int m  /*< Mode number m */
);
/* Get a mode from list */
ListAmpPhaseFDMode* ListAmpPhaseFDMode_GetMode(
	   ListAmpPhaseFDMode* list,  /* List structure to get this mode from */
	   int l, /*< Mode number l */
	   int m  /*< Mode number m */
);
/* Destroy list of modes, all data contained is destroyed */
void ListAmpPhaseFDMode_Destroy(
	   ListAmpPhaseFDMode* list  /* List structure to destroy */
);

/******************************************************************************/
/* Inline functions */
/******************************************************************************/

inline double real_vector_get(real_vector* v, size_t i)
{
  return v->data[i];
}
inline void real_vector_set(real_vector* v, size_t i, double val)
{
  v->data[i] = val;
  return;
}

inline double complex complex_vector_get(complex_vector* v, size_t i)
{
  return v->data[i];
}
inline void complex_vector_set(complex_vector* v, size_t i, double complex val)
{
  v->data[i] = val;
  return;
}

inline double real_matrix_get(real_matrix* m, size_t i, size_t j)
{
  return m->data[i * m->size2 + j];
}
inline double* real_matrix_line(real_matrix* m, size_t i)
{
  return &(m->data[i * m->size2]);
}
inline void real_matrix_set(real_matrix* m, size_t i, size_t j, double val)
{
  m->data[i * m->size2 + j] = val;
  return;
}

inline double complex complex_array_3d_get(complex_array_3d* a, size_t i, size_t j, size_t k)
{
  return a->data[(i * a->size2 + j) * a->size3 + k];
}
inline double complex* complex_array_3d_line(complex_array_3d* a, size_t i)
{
  return &(a->data[i * a->size3]);
}
inline void complex_array_3d_set(complex_array_3d* a, size_t i, size_t j, size_t k, double complex val)
{
  a->data[(i * a->size3 + j) * a->size2 + k] = val;
  return;
}

/******************************************************************************/
/* Structure and function for generic function references */
/******************************************************************************/

typedef struct {
    const void* object;
    double (*function) (const void* object, double);
} object_function;

inline double call_object_function(object_function obfn, double x) {
  return obfn.function(obfn.object, x);
}

/******************************************************************************/
/* Structure and function for LAL-like FrequencySeries */
/******************************************************************************/

typedef struct tagCOMPLEX16FrequencySeries {
    double complex *data;
    char *name;
    long epoch;
    double f0;
    double deltaF;
    // Unit sampleUnits;
    size_t length;
} COMPLEX16FrequencySeries;

COMPLEX16FrequencySeries *CreateCOMPLEX16FrequencySeries(
    const char *name,
    long epoch,
    double f0,
    double deltaF,
    size_t length
);

COMPLEX16FrequencySeries *ResizeCOMPLEX16FrequencySeries(
                                   COMPLEX16FrequencySeries *fs, size_t length);
void DestroyCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs);


#endif	// of #ifndef _STRUCT_H
