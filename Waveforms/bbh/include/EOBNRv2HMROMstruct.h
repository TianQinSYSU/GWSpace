/**
 * \author Sylvain Marsat, University of Maryland - NASA GSFC
 *
 * \brief C header for structures for EOBNRv2HM reduced order model (non-spinning version).
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

#ifndef _EOBNRV2HMROMSTRUCT_H
#define _EOBNRV2HMROMSTRUCT_H

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


#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

/***************************************************/
/*************** Type definitions ******************/

typedef struct tagSplineList {
    gsl_spline*            spline; /* The gsl spline */
    gsl_interp_accel*      accel; /* The gsl accelerator */
    int                    i; /* Index in the list  */
    struct tagSplineList*  next; /* Pointer to the next list element */
} SplineList;

typedef struct tagEOBNRHMROMdata
{
  gsl_vector* q;
  gsl_vector* freq;
  gsl_matrix* Camp;
  gsl_matrix* Cphi;
  gsl_matrix* Bamp;
  gsl_matrix* Bphi;
  gsl_vector* shifttime;
  gsl_vector* shiftphase;
} EOBNRHMROMdata;

typedef struct tagEOBNRHMROMdata_interp
{
  SplineList* Camp_interp; /* List of splines for the amp coefficients - SplineList, index of reduced basis */
  SplineList* Cphi_interp; /* List of splines for the phase coefficients - SplineList, index of reduced basis */
  SplineList* shifttime_interp; /* interpolated shift in time - SplineList with one element */
  SplineList* shiftphase_interp; /* interpolated shift in phase - SplineList with one element */
} EOBNRHMROMdata_interp;

typedef struct tagEOBNRHMROMdata_coeff
{
  gsl_vector* Camp_coeff;
  gsl_vector* Cphi_coeff;
  double*     shifttime_coeff;
  double*     shiftphase_coeff;
} EOBNRHMROMdata_coeff;

typedef struct tagListmodesEOBNRHMROMdata
{
    EOBNRHMROMdata*                     data; /* The ROM data. */
    int                                 l; /* Node mode l  */
    int                                 m; /* Node submode m  */
    struct tagListmodesEOBNRHMROMdata*  next; /* next pointer */
} ListmodesEOBNRHMROMdata;

typedef struct tagListmodesEOBNRHMROMdata_interp
{
    EOBNRHMROMdata_interp*                     data_interp; /* The splines built from the coefficients. */
    int                                        l; /* Node mode l  */
    int                                        m; /* Node submode m  */
    struct tagListmodesEOBNRHMROMdata_interp*  next; /* next pointer */
} ListmodesEOBNRHMROMdata_interp;

typedef struct tagListmodesEOBNRHMROMdata_coeff
{
    EOBNRHMROMdata_coeff*                     data_coeff; /* The data of coefficients. */
    int                                       l; /* Node mode l  */
    int                                       m; /* Node submode m  */
    struct tagListmodesEOBNRHMROMdata_coeff*  next; /* next pointer */
} ListmodesEOBNRHMROMdata_coeff;


/**********************************************************/
/**************** Internal functions **********************/

/* Functions associated to list manipulations */
SplineList* SplineList_AddElementNoCopy(
	   SplineList* appended,  /* List structure to prepend to */
	   gsl_spline* spline,  /* spline to contain */
           gsl_interp_accel* accel,  /* accelerator to contain */
	   int  i /* index in the list */
);
SplineList* SplineList_GetElement( 
	   SplineList* const splinelist,  /* List structure to get a particular mode from */
	   const int i /* index in the list */    
);
void SplineList_Destroy( 
	   SplineList* list  /* List structure to destroy; notice that the content is destroyed too */
);
ListmodesEOBNRHMROMdata* ListmodesEOBNRHMROMdata_AddModeNoCopy(
	   ListmodesEOBNRHMROMdata* appended,  /* List structure to prepend to */
	   EOBNRHMROMdata* indata,  /* data to contain */
	   int l, /*< major mode number */
	   int m  /*< minor mode number */
);
ListmodesEOBNRHMROMdata* ListmodesEOBNRHMROMdata_GetMode( 
	   ListmodesEOBNRHMROMdata* const list,  /* List structure to get a particular mode from */
	   int l, /*< major mode number */
	   int m  /*< minor mode number */    
);
void ListmodesEOBNRHMROMdata_Destroy( 
	   ListmodesEOBNRHMROMdata* list  /* List structure to destroy; notice that the data is destroyed too */
);
ListmodesEOBNRHMROMdata_interp* ListmodesEOBNRHMROMdata_interp_AddModeNoCopy(
	   ListmodesEOBNRHMROMdata_interp* appended,  /* List structure to prepend to */
	   EOBNRHMROMdata_interp* data,  /* data to contain */
	   int l, /* major mode number */
	   int m  /* minor mode number */
);
ListmodesEOBNRHMROMdata_interp* ListmodesEOBNRHMROMdata_interp_GetMode( 
	   ListmodesEOBNRHMROMdata_interp* const list,  /* List structure to get a particular mode from */
	   int l, /*< major mode number */
	   int m  /*< minor mode number */    
);
void ListmodesEOBNRHMROMdata_interp_Destroy( 
	   ListmodesEOBNRHMROMdata_interp* list  /* List structure to destroy; notice that the data is destroyed too */
);
ListmodesEOBNRHMROMdata_coeff* ListmodesEOBNRHMROMdata_coeff_AddModeNoCopy(
	   ListmodesEOBNRHMROMdata_coeff* appended,  /* List structure to prepend to */
	   EOBNRHMROMdata_coeff* data,  /* data to contain */
	   int l, /* major mode number */
	   int m  /* minor mode number */
);
ListmodesEOBNRHMROMdata_coeff* ListmodesEOBNRHMROMdata_coeff_GetMode( 
	   ListmodesEOBNRHMROMdata_coeff* const list,  /* List structure to get a particular mode from */
	   int l, /*< major mode number */
	   int m  /*< minor mode number */    
);
void ListmodesEOBNRHMROMdata_coeff_Destroy( 
	   ListmodesEOBNRHMROMdata_coeff* list  /* List structure to destroy; notice that the data is destroyed too */
);

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _EOBNRV2HMROMSTRUCT_H */
