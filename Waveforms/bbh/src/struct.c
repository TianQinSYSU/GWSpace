/*
 * Copyright (C) 2019 Sylvain Marsat
 *
 */

#include "constants.h"
#include "struct.h"

/******************************************************************************/
/* Handling errors */
/******************************************************************************/

/* Simplified code from lal/std/XLALError.c */
/* Return the error message associated with an error number or return value. */
const char *ErrorString(int code) {

    if (code <= 0) {    /* this is a return code, not an error number */
        if (code == SUCCESS)
            return "SUCCESS";
        else if (code == FAILURE)
            return "FAILURE";
        else
            return "Unknown return code";
    }

    /* check to see if an internal function call has failed, but the error
     * number was not "or"ed against the mask EFUNC */
    if (code == ERROR_EFUNC)
        return "Internal function call failed";

    /* use this to report error strings... deals with possible mask for
     * errors arising from internal function calls */
# define ERROR_STRING(s) \
    ( ( code & ERROR_EFUNC ) ? \
                        "Internal function call failed: " s : (const char *) s )
    switch (code & ~ERROR_EFUNC) {
        /* these are standard error numbers */
    case ERROR_EINVAL:
        return ERROR_STRING("Invalid input value error");
    case ERROR_EFAULT:
        return ERROR_STRING("Invalid pointer error");
    case ERROR_EDOM:
        return ERROR_STRING("Input domain error");
    case ERROR_ENOMEM:
        return ERROR_STRING("Memory allocation error");
        /* unrecognized error number */
    default:
        return "Unknown error";
    }
# undef ERROR_STRING
    return NULL;        /* impossible to get here */
}

void RaiseError(const char *funcstr, const char *filestr, const int line,
                ERROR_type e, char *errstr, ...)
{

  fprintf(stderr, "%s in %s (%s:%d):\n",
                   ErrorString(e), funcstr, filestr, line);

  va_list argptr;
  va_start(argptr, errstr);
  vfprintf(stderr, errstr, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);

  exit(-1);
}

void RaiseErrorCode(const char *funcstr, const char *filestr, const int line,
                ERROR_type e, char *errstr, ...)
{

  fprintf(stderr, "%s in %s (%s:%d):\n",
                   ErrorString(e), funcstr, filestr, line);

  va_list argptr;
  va_start(argptr, errstr);
  vfprintf(stderr, errstr, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
}

void RaiseWarning(const char *funcstr, const char *filestr, const int line,
                  char *warningstr, ...)
{

  fprintf(stderr, "Warning in %s (%s:%d):\n",
                   funcstr, filestr, line);

  va_list argptr;
  va_start(argptr, warningstr);
  vfprintf(stderr, warningstr, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
}

void RaiseInfo(const char *funcstr, const char *filestr, const int line,
               char *infostr, ...)
{

  fprintf(stderr, "Info in %s (%s:%d):\n",
                   funcstr, filestr, line);

  va_list argptr;
  va_start(argptr, infostr);
  vfprintf(stderr, infostr, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);

  exit(-1);
}

// FIXME: want file and line number where the error occurred
// void ERROR(ERROR_type e, char *errstr, ...) {
//
//   fprintf(stderr, "%s in %s:%d:\n", ErrorString(e), __FILE__, __LINE__);
//
//   va_list argptr;
//   va_start(argptr, errstr);
//   vfprintf(stderr, errstr, argptr);
//   va_end(argptr);
//
//   exit(-1);
// }

void RaiseCheck(const char *funcstr, const char *filestr, const int line,
                bool assertion, ERROR_type e, char *errstr)
{
  if (!(assertion)) ERROR(e, errstr);
}

void PrintCheck(const char *funcstr, const char *filestr, const int line,
                bool assertion, char *errstr)
{
  if (!(assertion)) WARNING(errstr);
}


/* GSL error handler */
void GSL_Err_Handler(
  const char *reason,
  const char *file,
  int line,
  int gsl_errno)
{
  printf("gsl: %s:%d: %s - %d\n", file, line, reason, gsl_errno);
  exit(1);
}

/******************************************************************************/
/* I/O functions for GSL structures */
/******************************************************************************/

/* Functions to read/write binary data from files */
int Read_GSL_Vector(const char dir[], const char fname[], gsl_vector *v) {
  char *path=malloc(strlen(dir)+64);
  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Error reading data from %s\n", path);
    free(path);
    return(FAILURE);
  }
  int ret = gsl_vector_fread(f, v);
  if (ret != 0) {
    fprintf(stderr, "Error reading data from %s.\n",path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}
int Read_GSL_Matrix(const char dir[], const char fname[], gsl_matrix *m) {
  char *path=malloc(strlen(dir)+64);
  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Error reading data from %s\n", path);
    free(path);
    return(FAILURE);
  }
  int ret = gsl_matrix_fread(f, m);
  if (ret != 0) {
    fprintf(stderr, "Error reading data from %s\n", path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}
int Write_GSL_Vector(const char dir[], const char fname[], gsl_vector *v) {
  char *path=malloc(strlen(dir)+64);
  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error writing data to %s\n", path);
    free(path);
    return(FAILURE);
  }
  int ret = gsl_vector_fwrite(f, v);
  if (ret != 0) {
    fprintf(stderr, "Error writing data to %s\n",path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}
int Write_GSL_Matrix(const char dir[], const char fname[], gsl_matrix *m) {
  char *path=malloc(strlen(dir)+64);

  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error writing data to %s\n", path);
    free(path);
    return(FAILURE);
  }
  int ret = gsl_matrix_fwrite(f, m);
  if (ret != 0) {
    fprintf(stderr, "Error writing data to %s\n", path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}

/* Functions to read/write ascii data from files */
int Read_GSL_Text_Vector(const char dir[], const char fname[], gsl_vector *v) {
  char *path=malloc(strlen(dir)+64);
  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Error reading data from %s\n", path);
    free(path);
    return(FAILURE);
  }
  int ret = gsl_vector_fscanf(f, v);
  if (ret != 0) {
    fprintf(stderr, "Error reading data from %s.\n",path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}
int Read_GSL_Text_Matrix(const char dir[], const char fname[], gsl_matrix *m) {
  char *path=malloc(strlen(dir)+64);
  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Error reading data from %s\n", path);
    free(path);
    return(FAILURE);
  }
  int ret = gsl_matrix_fscanf(f, m);
  if (ret != 0) {
    fprintf(stderr, "Error reading data from %s.\n",path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}
int Write_GSL_Text_Vector(const char dir[], const char fname[], gsl_vector *v) {
  char *path=malloc(strlen(dir)+64);
  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error writing data to %s\n", path);
    free(path);
    return(FAILURE);
  }
  int ret = gsl_vector_fprintf(f, v, "%.16e");
  if (ret != 0) {
    fprintf(stderr, "Error writing data to %s\n",path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}
int Write_GSL_Text_Matrix(const char dir[], const char fname[], gsl_matrix *m) {
  char *path=malloc(strlen(dir)+64);
  int ret = 0;

  sprintf(path,"%s/%s", dir, fname);
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error writing data to %s\n",path);
    free(path);
    return(FAILURE);
  }
  int N = (int) m->size1;
  int M = (int) m->size2;
  for(int i=0; i<N; i++){
    for(int j=0; j<M; j++){
      ret |= (fprintf(f, "%.16e ", gsl_matrix_get(m, i, j)) < 0);
    }
    if(i < N-1) ret |= (fprintf(f, "\n") < 0);
  }
  if (ret != 0) {
    fprintf(stderr, "Error writing data to %s\n",path);
    free(path);
    return(FAILURE);
  }
  fclose(f);
  free(path);
  return(SUCCESS);
}

/******************************************************************************/
/* Structure functions for real_vector */
/******************************************************************************/

real_vector* real_vector_alloc(size_t size)
{
  real_vector* v = (real_vector*) malloc(sizeof(real_vector));
  v->size = size;
  v->data = (double*) malloc(sizeof(double) * size);
  if (v->data == NULL)
    ERROR(ERROR_ENOMEM, "Failed to allocated data array.");

  memset(v->data, 0, sizeof(double) * size);
  return v;
}
void real_vector_free(real_vector *v) {
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(v->data);
  free(v);
}

real_vector* real_vector_view(double* data, size_t size)
{
  if (data == NULL)
    ERROR(ERROR_EINVAL, "Input data pointer double* is NULL.");
  real_vector* v = (real_vector*) malloc(sizeof(real_vector));
  v->size = size;
  v->data = data;
  return v;
}
real_vector* real_vector_view_subvector(
  real_vector* v,
  size_t istart,
  size_t iend)
{
  if (v == NULL)
    ERROR(ERROR_EINVAL, "Input vector pointer is NULL.");
  real_vector* subv = (real_vector*) malloc(sizeof(real_vector));
  if (istart<0 || istart>(v->size+1) || iend<0 || iend>(v->size+1)
      || istart>iend)
    ERROR(ERROR_EINVAL, "Incompatible indices for subvector.");
  subv->size = iend - istart + 1;
  subv->data = &(v->data)[istart];
  return subv;
}
void real_vector_view_free(real_vector *v)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(v);
}
real_vector* real_vector_resize(real_vector *v, size_t istart, size_t iend)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Input pointer is NULL.");
  if (istart>iend)
    ERROR(ERROR_EINVAL, "Incompatible indices, istart>iend.");
  if (iend>v->size-1)
    ERROR(ERROR_EINVAL, "Max index exceeds size of vector.");
  real_vector* v_r = real_vector_alloc(iend - istart + 1);
  memcpy(v_r->data, &(v->data[istart]), (iend - istart + 1) * sizeof(double));
  return v_r;
}
void real_vector_set_zero(real_vector* v)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Input pointer to real_vector is NULL.");
  memset(v->data, 0, v->size * sizeof(double));
}
void real_vector_scale_constant(real_vector* v, const double s)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Input pointer to real_vector is NULL.");
  double* data = v->data;
  for (size_t i=0; i<v->size; i++) data[i] *= s;
}
void real_vector_add_constant(real_vector* v, const double a)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Input pointer to real_vector is NULL.");
  double* data = v->data;
  for (size_t i=0; i<v->size; i++) data[i] += a;
}
/* NOTE: copy data, destination real_vector already allocated */
void real_vector_copy_gsl_vector(real_vector* v, gsl_vector* gslvec)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Input pointer to real_vector is NULL.");
  if (gslvec == NULL)
    ERROR(ERROR_EFAULT, "Input pointer to gsl_vector is NULL.");
  if (v->size != gslvec->size)
    ERROR(ERROR_EINVAL, "Incompatible size: %d, %d.", v->size, gslvec->size);
  memcpy(v->data, gslvec->data, v->size * sizeof(double));
}
/* Functions converting between gsl and local vectors, doing the allocation */
void real_vector_from_gsl_vector(real_vector** v, gsl_vector* gslvec)
{
  if ((*v) != NULL)
    ERROR(ERROR_EFAULT, "Output pointer to real_vector is not NULL.");
  if (gslvec == NULL)
    ERROR(ERROR_EFAULT, "Input pointer to gsl_vector is NULL.");
  (*v) = real_vector_alloc(gslvec->size);
  memcpy((*v)->data, gslvec->data, gslvec->size * sizeof(double));
}
void gsl_vector_from_real_vector(gsl_vector** gslvec, real_vector* vec)
{
  if ((*gslvec) != NULL)
    ERROR(ERROR_EFAULT, "Output pointer to gsl_vector is not NULL.");
  if (vec == NULL)
    ERROR(ERROR_EFAULT, "Input pointer to real_vector is NULL.");
  (*gslvec) = gsl_vector_alloc(vec->size);
  memcpy((*gslvec)->data, vec->data, vec->size * sizeof(double));
}

/******************************************************************************/
/* Structure functions for complex_vector */
/******************************************************************************/

complex_vector* complex_vector_alloc(size_t size)
{
  complex_vector* v = (complex_vector*) malloc(sizeof(complex_vector));
  v->size = size;
  v->data = (double complex*) malloc(sizeof(double complex) * size);
  if (v->data == NULL)
    ERROR(ERROR_ENOMEM, "Failed to allocated data array.");

  memset(v->data, 0, sizeof(double complex) * size);
  return v;
}
void complex_vector_free(complex_vector *v) {
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(v->data);
  free(v);
}

complex_vector* complex_vector_view(double complex* data, size_t size)
{
  if (data == NULL)
    ERROR(ERROR_EINVAL, "Input data pointer double complex* is NULL.");
  complex_vector* v = (complex_vector*) malloc(sizeof(complex_vector));
  v->size = size;
  v->data = data;
  return v;
}
complex_vector* complex_vector_view_subvector(
  complex_vector* v,
  size_t istart,
  size_t iend)
{
  if (v == NULL)
    ERROR(ERROR_EINVAL, "Input vector pointer is NULL.");
  complex_vector* subv = (complex_vector*) malloc(sizeof(complex_vector));
  if (istart<0 || istart>(v->size+1) || iend<0 || iend>(v->size+1)
      || istart>iend)
    ERROR(ERROR_EINVAL, "Incompatible indices for subvector.");
  subv->size = iend - istart + 1;
  subv->data = &(v->data)[istart];
  return subv;
}
void complex_vector_view_free(complex_vector *v)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(v);
}
complex_vector* complex_vector_resize(complex_vector *v,
                                      size_t istart, size_t iend)
{
  if (v == NULL)
    ERROR(ERROR_EFAULT, "Input pointer is NULL.");
  if (istart>iend)
    ERROR(ERROR_EINVAL, "Incompatible indices, istart>iend.");
  if (iend>v->size-1)
    ERROR(ERROR_EINVAL, "Max index exceeds size of vector.");
  complex_vector* v_r = complex_vector_alloc(iend - istart + 1);
  memcpy(v_r->data, &(v->data[istart]),
         (iend - istart + 1) * sizeof(double complex));
  return v_r;
}

/******************************************************************************/
/* Structure functions for real_matrix */
/******************************************************************************/

real_matrix* real_matrix_alloc(size_t size1, size_t size2)
{
  real_matrix* m = (real_matrix*) malloc(sizeof(real_matrix));
  m->size1 = size1;
  m->size2 = size2;
  m->data = (double*) malloc(sizeof(double) * size1 * size2);
  if (m->data == NULL)
    ERROR(ERROR_ENOMEM, "Failed to allocated data array.");

  memset(m->data, 0, sizeof(double) * size1 * size2);
  return m;
}
void real_matrix_free(real_matrix *m)
{
  if (m == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(m->data);
  free(m);
}
real_matrix* real_matrix_view(double* data, size_t size1, size_t size2)
{
  if (data == NULL)
    ERROR(ERROR_EINVAL, "Input data pointer double* is NULL.");
  real_matrix* m = (real_matrix*) malloc(sizeof(real_matrix));
  m->size1 = size1;
  m->size2 = size2;
  m->data = data;
  return m;
}
void real_matrix_view_free(real_matrix *m)
{
  if (m == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(m);
}

/******************************************************************************/
/* Structure functions for complex_array_3d */
/******************************************************************************/

complex_array_3d* complex_array_3d_alloc(size_t size1, size_t size2, size_t size3)
{
  complex_array_3d* a = (complex_array_3d*) malloc(sizeof(complex_array_3d));
  a->size1 = size1;
  a->size2 = size2;
  a->size3 = size3;
  a->data = (double complex*) malloc(sizeof(double complex) * size1 * size2 * size3);
  if (a->data == NULL)
    ERROR(ERROR_ENOMEM, "Failed to allocated data array.");

  memset(a->data, 0, sizeof(double complex) * size1 * size2 * size3);
  return a;
}
void complex_array_3d_free(complex_array_3d *a)
{
  if (a == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(a->data);
  free(a);
}
complex_array_3d* complex_array_3d_view(double complex* data, size_t size1, size_t size2, size_t size3)
{
  if (data == NULL)
    ERROR(ERROR_EINVAL, "Input data pointer double* is NULL.");
  complex_array_3d* a = (complex_array_3d*) malloc(sizeof(complex_array_3d));
  a->size1 = size1;
  a->size2 = size2;
  a->size3 = size3;
  a->data = data;
  return a;
}
void complex_array_3d_view_free(complex_array_3d *a)
{
  if (a == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(a);
}

/******************************************************************************/
/* Structure functions for AmpPhaseFDWaveform and CAmpPhaseFDWaveform */
/******************************************************************************/

AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(size_t length)
{
  AmpPhaseFDWaveform* wf =
                       (AmpPhaseFDWaveform*) malloc(sizeof(AmpPhaseFDWaveform));
  wf->length = length;
  wf->freq = (double*) malloc(sizeof(double) * length);
  wf->amp = (double*) malloc(sizeof(double) * length);
  wf->phase = (double*) malloc(sizeof(double) * length);
  if ((wf->freq == NULL) || (wf->amp == NULL) || (wf->phase == NULL) )
    ERROR(ERROR_ENOMEM, "Failed to allocated one of the data arrays.");

  memset(wf->freq, 0, sizeof(double) * length);
  memset(wf->amp, 0, sizeof(double) * length);
  memset(wf->phase, 0, sizeof(double) * length);
  return wf;
}

void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform* wf) {
  if (wf == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  free(wf->freq);
  free(wf->amp);
  free(wf->phase);
  free(wf);
}

/******************************************************************************/
/* Structure functions for AmpPhaseFDMode */
/******************************************************************************/

int AmpPhaseFDMode_Init(
  AmpPhaseFDMode** hlm,
  size_t length_amp,
  size_t length_phase)
{
  if ( !hlm )
    ERROR(ERROR_EFAULT, "Input double pointer is NULL.");
  if ( *hlm )
    ERROR(ERROR_EFAULT, "Input pointer is not NULL.");

  *hlm = (AmpPhaseFDMode*) malloc(sizeof(AmpPhaseFDMode));
  (*hlm)->freq_amp = real_vector_alloc(length_amp);
  (*hlm)->amp_real = real_vector_alloc(length_phase);
  (*hlm)->amp_imag = real_vector_alloc(length_phase);
  (*hlm)->freq_phase = real_vector_alloc(length_phase);
  (*hlm)->phase = real_vector_alloc(length_phase);
  (*hlm)->tf = real_vector_alloc(length_phase);

  return SUCCESS;
}

void AmpPhaseFDMode_Destroy(AmpPhaseFDMode* hlm) {
  if (hlm == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  if(hlm->freq_amp) real_vector_free(hlm->freq_amp);
  if(hlm->amp_real) real_vector_free(hlm->amp_real);
  if(hlm->amp_imag) real_vector_free(hlm->amp_imag);
  if(hlm->freq_phase) real_vector_free(hlm->freq_phase);
  if(hlm->phase) real_vector_free(hlm->phase);
  if(hlm->tf) real_vector_free(hlm->tf);
  free(hlm);
}

/******************************************************************************/
/* Structure functions for CAmpPhaseFDData */
/******************************************************************************/

int CAmpPhaseFDData_Init(
  CAmpPhaseFDData** h,
  size_t length)
{
  if ( !h )
    ERROR(ERROR_EFAULT, "Input double pointer is NULL.");
  if ( *h )
    ERROR(ERROR_EFAULT, "Input pointer is not NULL.");

  *h = (CAmpPhaseFDData*) malloc(sizeof(CAmpPhaseFDData));
  (*h)->freq = real_vector_alloc(length);
  (*h)->amp_real = real_vector_alloc(length);
  (*h)->amp_imag = real_vector_alloc(length);
  (*h)->phase = real_vector_alloc(length);

  return SUCCESS;
}

void CAmpPhaseFDData_Destroy(CAmpPhaseFDData* h) {
  if (h == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  if(h->freq) real_vector_free(h->freq);
  if(h->amp_real) real_vector_free(h->amp_real);
  if(h->amp_imag) real_vector_free(h->amp_imag);
  if(h->phase) real_vector_free(h->phase);
  free(h);
}

/******************************************************************************/
/* Structure functions for CAmpPhaseFDSpline */
/******************************************************************************/

int CAmpPhaseFDSpline_Init(
  CAmpPhaseFDSpline** h,
  size_t length)
{
  if ( !h )
    ERROR(ERROR_EFAULT, "Input double pointer is NULL.");
  if ( *h )
    ERROR(ERROR_EFAULT, "Input pointer is not NULL.");

  *h = (CAmpPhaseFDSpline*) malloc(sizeof(CAmpPhaseFDSpline));
  (*h)->spline_amp_real = real_matrix_alloc(length, 5);
  (*h)->spline_amp_imag = real_matrix_alloc(length, 5);
  (*h)->spline_phase = real_matrix_alloc(length, 5);

  return SUCCESS;
}

void CAmpPhaseFDSpline_Destroy(CAmpPhaseFDSpline* h) {
  if (h == NULL)
    ERROR(ERROR_EFAULT, "Trying to free NULL pointer.");
  if(h->spline_amp_real) real_matrix_free(h->spline_amp_real);
  if(h->spline_amp_imag) real_matrix_free(h->spline_amp_imag);
  if(h->spline_phase) real_matrix_free(h->spline_phase);
  free(h);
}

/******************************************************************************/
/* Structure functions for ListAmpPhaseFDMode */
/******************************************************************************/

/* Add mode to list, directly without copying the data */
/* We don't allow for the case where the mode already exists in the list */
ListAmpPhaseFDMode* ListAmpPhaseFDMode_AddMode(
	   ListAmpPhaseFDMode* list,  /* List structure to prepend to */
	   AmpPhaseFDMode* hlm,  /* Data for the mode */
	   int l,  /*< Mode number l */
	   int m)  /*< Mode number m */
{
  ListAmpPhaseFDMode* listp;
  listp = list;
  /* Go at the end of the list, or error at (l,m) if it already exists */
  while ( listp ) {
    if( l == listp->l && m == listp->m )
      ERROR(ERROR_EFAULT, "Mode (%d, %d) already present in ", l, m);
    listp = listp->next;
  }
  /* Set list element to be prepended */
  listp = malloc( sizeof(ListAmpPhaseFDMode) );
  listp->l = l;
  listp->m = m;
  if ( hlm )
    listp->hlm = hlm;
  else
    listp->hlm = NULL;
  /* Set next pointer to old list */
  if ( list )
    listp->next = list;
  else
    listp->next = NULL;
  return listp;
}

/* Get pointer to mode in list */
/* If input list is NULL (empty list) or if mode not present, return NULL */
ListAmpPhaseFDMode* ListAmpPhaseFDMode_GetMode(
	   ListAmpPhaseFDMode* list,  /* List structure to get this mode from */
	   int l,  /*< Mode number l */
	   int m)  /*< Mode number m */
{
  if( !list ) return NULL;

  ListAmpPhaseFDMode* listp = list;
  while ( listp->l != l || listp->m != m ) {
      listp = listp->next;
      if( !listp ) return NULL;
  }
  return listp;
}
/* Destroy list of modes, all data contained is destroyed */
void ListAmpPhaseFDMode_Destroy(
	   ListAmpPhaseFDMode* list)  /* List structure to destroy */
{
  ListAmpPhaseFDMode* listp;
  while ( (listp = list) ) {
    if ( listp->hlm ) AmpPhaseFDMode_Destroy( listp->hlm );
    list = listp->next;
    free( listp );
  }
}

/******************************************************************************/
/* Functions for LAL-like FrequencySeries */
/******************************************************************************/

COMPLEX16FrequencySeries *CreateCOMPLEX16FrequencySeries(
    const char *name,
    long epoch,
    double f0,
    double deltaF,
    size_t length)
{
    COMPLEX16FrequencySeries* fs = (COMPLEX16FrequencySeries*) malloc(
                                              sizeof(COMPLEX16FrequencySeries));

    size_t name_length = strlen(name);
    fs->name = (char*)malloc((name_length+1) * sizeof(char));
    strcpy(fs->name, name);

    fs->epoch = epoch;
    fs->f0 = f0;
    fs->deltaF = deltaF;

    fs->length = length;
    fs->data = (double complex *)malloc(sizeof(double complex) * length);
    if (fs->data == NULL)
        ERROR(ERROR_ENOMEM, "Failed to allocated data array.");

    memset(fs->data, 0, sizeof(double complex) * length);
    return fs;
}

COMPLEX16FrequencySeries *ResizeCOMPLEX16FrequencySeries(
                                    COMPLEX16FrequencySeries *fs, size_t length)
{
    // assume that we start at index 0
    free(fs->data);

    fs->length = length;
    fs->data = (double complex *)malloc(sizeof(double complex) * length);
    if (fs->data == NULL)
        ERROR(ERROR_ENOMEM, "Failed to allocated data array.");

    return fs;
}

void DestroyCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs) {
    free(fs->data);
    free(fs->name);
    free(fs);
}
