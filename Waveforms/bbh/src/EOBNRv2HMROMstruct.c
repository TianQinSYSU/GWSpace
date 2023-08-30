/**
 * \author Sylvain Marsat, University of Maryland - NASA GSFC
 *
 * \brief C code for structures EOBNRv2HM reduced order model (non-spinning version).
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
#include "EOBNRv2HMROMstruct.h"
#include "EOBNRv2HMROM.h"


/************************************************************************/
/********************* Functions for list structures ********************/

/***************** Functions for the SplineList structure ****************/

/* Prepend a node to a linked list of splines, or create a new head */
SplineList* SplineList_AddElementNoCopy(
	   SplineList* appended,  /* List structure to prepend to */
	   gsl_spline* spline,  /* spline to contain */
           gsl_interp_accel* accel,  /* accelerator to contain */
	   int i /* index in the list */)
{
    SplineList* splinelist;
    /* Check if the node with this index already exists */
    splinelist = appended;
    while( splinelist ){
      if( i == splinelist->i ){
	break;
      }
      splinelist = splinelist->next;
    }
    if( splinelist ){ /* We don't allow for the case where the index already exists*/
      printf("Error: Tried to add an already existing index to a SplineList");
      return(NULL);
    } else { /* In that case, we do NOT COPY the input spline, which therefore can't be
                 used anywhere else; this will be acceptable as these operations will only be done
                 when initializing the data */
      splinelist = malloc( sizeof(SplineList) );
    }
    splinelist->i = i;
    if( spline ){
      splinelist->spline = spline;
    } else {
      splinelist->spline = NULL;
    }
    if( accel ){
      splinelist->accel = accel;
    } else {
      splinelist->accel = NULL;
    }
    if( appended ){
      splinelist->next = appended;
    } else {
        splinelist->next = NULL;
    }
    return splinelist;
}
/* Get the element of a SplineList with a given index */
SplineList* SplineList_GetElement(
	      SplineList* const splinelist,  /* List structure to get element from */
              const int i ) /* Index looked for */
{
    if( !splinelist ) return NULL;

    SplineList* itr = splinelist;
    while( itr->i != i ){
        itr = itr->next;
        if( !itr ) return NULL;
    }
    return itr; /* The element returned is itself a pointer to a SplineList */
}
/* Delete list from given pointer to the end of the list */
void SplineList_Destroy( SplineList* splinelist ) /* Head of linked list to destroy */
{
  SplineList* pop;
  while( (pop = splinelist) ){
    if( pop->spline ){ /* Internal spline and accelerator are freed */
      gsl_spline_free( pop->spline );
    }
    if( pop->accel ){
      gsl_interp_accel_free( pop->accel );
    }
    /* Notice that the index i is not freed, like in SphHarmTimeSeries struct indices l and m */
    splinelist = pop->next;
    free( pop );
  }
}

/***************** Functions for the EOBNRHMROMdata structure ****************/
ListmodesEOBNRHMROMdata* ListmodesEOBNRHMROMdata_AddModeNoCopy(
	   ListmodesEOBNRHMROMdata* appended,  /* List structure to prepend to */
	   EOBNRHMROMdata* data,  /* data to contain */
	   const int l, /* major mode number */
	   const int m  /* minor mode number */)
{
    ListmodesEOBNRHMROMdata* list;
    /* Check if the node with this mode already exists */
    list = appended;
    while( list ){
      if( l == list->l && m == list->m ){
	break;
      }
      list = list->next;
    }
    if( list ){ /* We don't allow for the case where the mode already exists in the list*/
      printf("Error: Tried to add an already existing mode to a ListmodesEOBNRHMROMdata ");
      return(NULL);
    } else { /* In that case, we do NOT COPY the input interpolated data, which therefore can't be
		used anywhere else; this will be acceptable as these operations will only be done
		when interpolating the initialization data */
      list = malloc( sizeof(ListmodesEOBNRHMROMdata) );
    }
    list->l = l;
    list->m = m;
    if( data ){
      list->data = data;
    } else {
      list->data = NULL;
    }
    if( appended ){
      list->next = appended;
    } else {
        list->next = NULL;
    }
    return list;
}
/* Get the element of a ListmodesEOBNRHMROMdata with a given index */
ListmodesEOBNRHMROMdata* ListmodesEOBNRHMROMdata_GetMode( 
	   ListmodesEOBNRHMROMdata* const list,  /* List structure to get a particular mode from */
	   int l, /*< major mode number */
	   int m  /*< minor mode number */ )
{
    if( !list ) return NULL;

    ListmodesEOBNRHMROMdata *itr = list;
    while( itr->l != l || itr->m != m ){
        itr = itr->next;
        if( !itr ) return NULL;
    }
    return itr; /* The element returned is itself a pointer to a ListmodesEOBNRHMROMdata */
}
void ListmodesEOBNRHMROMdata_Destroy( 
	   ListmodesEOBNRHMROMdata* list  /* List structure to destroy; notice that the data is destroyed too */
)
{
  ListmodesEOBNRHMROMdata* pop;
  while( (pop = list) ){
    if( pop->data ){ /* Destroying the EOBNRHMROMdata data */
      EOBNRHMROMdata_Cleanup( pop->data );
    }
    /* Notice that the mode indices l and m are not freed, like in SphHarmTimeSeries struct indices l and m */
    list = pop->next;
    free( pop );
  }
}

/***************** Functions for the EOBNRHMROMdata_interp structure ****************/
ListmodesEOBNRHMROMdata_interp* ListmodesEOBNRHMROMdata_interp_AddModeNoCopy(
	   ListmodesEOBNRHMROMdata_interp* appended,  /* List structure to prepend to */
	   EOBNRHMROMdata_interp* data_interp,  /* data to contain */
	   int l, /* major mode number */
	   int m  /* minor mode number */)
{
    ListmodesEOBNRHMROMdata_interp* list;
    /* Check if the node with this mode already exists */
    list = appended;
    while( list ){
      if( l == list->l && m == list->m ){
	break;
      }
      list = list->next;
    }
    if( list ){ /* We don't allow for the case where the mode already exists in the list*/
      printf("Error: Tried to add an already existing mode to a ListmodesEOBNRHMROMdata_interp ");
      return(NULL);
    } else { /* In that case, we do NOT COPY the input interpolated data, which therefore can't be
		used anywhere else; this will be acceptable as these operations will only be done
		when interpolating the initialization data */
      list = malloc( sizeof(ListmodesEOBNRHMROMdata_interp) );
    }
    list->l = l;
    list->m = m;
    if( data_interp ){
      list->data_interp = data_interp;
    } else {
      list->data_interp = NULL;
    }
    if( appended ){
      list->next = appended;
    } else {
        list->next = NULL;
    }
    return list;
}
/* Get the element of a ListmodesEOBNRHMROMdata with a given index */
ListmodesEOBNRHMROMdata_interp* ListmodesEOBNRHMROMdata_interp_GetMode( 
	   ListmodesEOBNRHMROMdata_interp* const list,  /* List structure to get a particular mode from */
	   int l, /*< major mode number */
	   int m  /*< minor mode number */ )
{
    if( !list ) return NULL;

    ListmodesEOBNRHMROMdata_interp *itr = list;
    while( itr->l != l || itr->m != m ){
        itr = itr->next;
        if( !itr ) return NULL;
    }
    return itr; /* The element returned is itself a pointer to a ListmodesEOBNRHMROMdata_interp */
}
void ListmodesEOBNRHMROMdata_interp_Destroy( 
	   ListmodesEOBNRHMROMdata_interp* list  /* List structure to destroy; notice that the data is destroyed too */
)
{
  ListmodesEOBNRHMROMdata_interp* pop;
  while( (pop = list) ){
    if( pop->data_interp ){ /* Destroying the EOBNRHMROMdata_interp data */
      EOBNRHMROMdata_interp_Cleanup( pop->data_interp );
    }
    /* Notice that the mode indices l and m are not freed, like in SphHarmTimeSeries struct indices l and m */
    list = pop->next;
    free( pop );
  }
}

/***************** Functions for the EOBNRHMROMdata_coeff structure ****************/
ListmodesEOBNRHMROMdata_coeff* ListmodesEOBNRHMROMdata_coeff_AddModeNoCopy(
	   ListmodesEOBNRHMROMdata_coeff* appended,  /* List structure to prepend to */
	   EOBNRHMROMdata_coeff* data_coeff,  /* data to contain */
	   int l, /* major mode number */
	   int m  /* minor mode number */)
{
    ListmodesEOBNRHMROMdata_coeff* list;
    /* Check if the node with this mode already exists */
    list = appended;
    while( list ){
      if( l == list->l && m == list->m ){
	break;
      }
      list = list->next;
    }
    if( list ){ /* We don't allow for the case where the mode already exists in the list*/
      printf("Error: Tried to add an already existing mode to a ListmodesEOBNRHMROMdata_coeff ");
      return(NULL);
    } else { /* In that case, we do NOT COPY the input interpolated data, which therefore can't be
		used anywhere else; this will be acceptable as these operations will only be done
		when interpolating the initialization data */
      list = malloc( sizeof(ListmodesEOBNRHMROMdata_coeff) );
    }
    list->l = l;
    list->m = m;
    if( data_coeff ){
      list->data_coeff = data_coeff;
    } else {
      list->data_coeff = NULL;
    }
    if( appended ){
      list->next = appended;
    } else {
        list->next = NULL;
    }
    return list;
}
/* Get the element of a ListmodesEOBNRHMROMdata_coeff with a given index */
ListmodesEOBNRHMROMdata_coeff* ListmodesEOBNRHMROMdata_coeff_GetMode( 
	   ListmodesEOBNRHMROMdata_coeff* const list,  /* List structure to get a particular mode from */
	   int l, /*< major mode number */
	   int m  /*< minor mode number */ )
{
    if( !list ) return NULL;

    ListmodesEOBNRHMROMdata_coeff *itr = list;
    while( itr->l != l || itr->m != m ){
        itr = itr->next;
        if( !itr ) return NULL;
    }
    return itr; /* The element returned is itself a pointer to a ListmodesEOBNRHMROMdata_coeff */
}
void ListmodesEOBNRHMROMdata_coeff_Destroy( 
	   ListmodesEOBNRHMROMdata_coeff* list  /* List structure to destroy; notice that the data is destroyed too */
)
{
  ListmodesEOBNRHMROMdata_coeff* pop;
  while( (pop = list) ){
    if( pop->data_coeff ){ /* Destroying the EOBNRHMROMdata_coeff data */
      EOBNRHMROMdata_coeff_Cleanup( pop->data_coeff );
    }
    /* Notice that the mode indices l and m are not freed, like in SphHarmTimeSeries struct indices l and m */
    list = pop->next;
    free( pop );
  }
}
