/**
 * \author Sylvain Marsat, University of Maryland - NASA GSFC
 *
 * \brief C code for EOBNRv2HM reduced order model (non-spinning version).
 * See CQG 31 195010, 2014, arXiv:1402.4146 for details on the reduced order method.
 * See arXiv:1106.1021 for the EOBNRv2HM model.
 *
 * Borrows from the SEOBNR ROM LAL code written by Michael Puerrer and John Veitch.
 *
 * Put the untared data in the directory designated by the environment variable ROM_DATA_PATH.
 *
 * Parameter ranges:
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
#include "struct.h"
#include "EOBNRv2HMROMstruct.h"
#include "EOBNRv2HMROM.h"

/*************************************************/
/********* Some general definitions **************/

/* Number and list of modes to generate - to be modified ultimately to allow for a selection of the desired mode(s) */
/* By convention the first mode of the list is used to set phiRef */
/* nbmodemax = 5 has been defined in the header */
const int listmode[nbmodemax][2] = { {2,2}, {2,1}, {3,3}, {4,4}, {5,5} };

/* Maximal mass ratio covered by the model - q=12 (almost) for now */
static const double q_max = 11.9894197212;
/* Minimal geometric frequency covered by the model - f=8Hz for M=10Msol for now */
static const double Mf_ROM_min = 0.0003940393857519091;
/* Maximal geometric frequency covered by the model - Mf=0.285 for the 55 mode - used as default */
//static const double Mf_ROM_max = 0.285;
/* Total mass (in units of solar mass) used to generate the ROM - used to restore the correct amplitude (global factor M) when coming back to physical units */
static const double M_ROM = 10.;

/* Define the number of points in frequency used by the SVD, identical for all modes */
static const int nbfreq = 300;
/* Define the number of training waveforms used by the SVD, identical for all modes */
static const int nbwf = 301;

/******************************************************************/
/********* Definitions for the persistent structures **************/

/* SEOBNR-ROM structures are generalized to lists */
ListmodesEOBNRHMROMdata* __EOBNRv2HMROM_data_init = NULL; /* for initialization only */
ListmodesEOBNRHMROMdata** const __EOBNRv2HMROM_data = &__EOBNRv2HMROM_data_init;
ListmodesEOBNRHMROMdata_interp* __EOBNRv2HMROM_interp_init = NULL; /* for initialization only */
ListmodesEOBNRHMROMdata_interp** const __EOBNRv2HMROM_interp = &__EOBNRv2HMROM_interp_init;
/* Tag indicating whether the data has been loaded and interpolated */
int __EOBNRv2HMROM_setup = FAILURE; /* To be set to SUCCESS after initialization*/

/********************* Miscellaneous ********************/

/* Return the closest higher power of 2 */
// static size_t NextPow2(const size_t n) {
//   return 1 << (size_t) ceil(log2(n));
// }

/* Arbitrary tuned q-dependent functions by which the frequencies for the 44 and 55 modes have been multiplied (to put the ringdowns at the same Mf). The frequencies stored in the data for the 44 and 55 modes are the rescaled ones, not the original ones. */
static double Scaling44(const double q) {
  return 1.-1./4.*exp(-(q-1.)/5.);
}
static double Scaling55(const double q) {
  return 1.-1./3.*exp(-(q-1.)/5.);
}

/* Function evaluating eta as a function of q */
static double EtaOfq(const double q) {
  return q/(1.+q)/(1.+q);
}
/* Function evaluating delta m/m = (m1-m2)/(m1+m2) as a function of q */
static double DeltaOfq(const double q) {
  return( (q-1.)/(q+1.) );
}

/* Fit of the frequency of the 22 mode at the peak amplitude - from table III in the EOBNRv2HM paper, Pan&al 1106 */
static double omega22peakOfq(const double q) {
  double eta = EtaOfq(q);
  return 0.2733 + 0.2316*eta + 0.4463*eta*eta;
}

/* Amplitude factors scaled out for each mode; this does not include the global amplitude factor scaled out from all modes. */
static double ModeAmpFactor(const int l, const int m, const double q) {
  double eta = EtaOfq(q);
  if( l==2 && m==2 ) return(sqrt(eta));
  else if( l==2 && m==1 ) return( sqrt(eta)*DeltaOfq(q) );
  else if( l==3 && m==3 ) return( sqrt(eta)*DeltaOfq(q) );
  else if( l==4 && m==4 ) return( sqrt(Scaling44(q))*sqrt(eta)*(1.-3.*eta) );
  else if( l==5 && m==5 ) return( pow(Scaling55(q), 1./6)*sqrt(eta)*DeltaOfq(q)*(1.-2.*eta) );
  else {
    fprintf(stderr, "Unknown mode (%d,%d) for the amplitude factor.\n", l, m);
    return(FAILURE);
  }
}

/********************* Functions to initialize and cleanup contents of data structures ********************/
void EOBNRHMROMdata_Init(EOBNRHMROMdata **data) {
  if(!data) exit(1);
  /* Create storage for structures */
  if(!*data) *data=malloc(sizeof(EOBNRHMROMdata));
  else
  {
    EOBNRHMROMdata_Cleanup(*data);
  }
  gsl_set_error_handler(&GSL_Err_Handler);
  (*data)->q = gsl_vector_alloc(nbwf);
  (*data)->freq = gsl_vector_alloc(nbfreq);
  (*data)->Camp = gsl_matrix_alloc(nk_amp,nbwf);
  (*data)->Cphi = gsl_matrix_alloc(nk_phi,nbwf);
  (*data)->Bamp = gsl_matrix_alloc(nbfreq,nk_amp);
  (*data)->Bphi = gsl_matrix_alloc(nbfreq,nk_phi);
  (*data)->shifttime = gsl_vector_alloc(nbwf);
  (*data)->shiftphase = gsl_vector_alloc(nbwf);
}
void EOBNRHMROMdata_interp_Init(EOBNRHMROMdata_interp **data_interp) {
  if(!data_interp) exit(1);
  /* Create storage for structures */
  if(!*data_interp) *data_interp=malloc(sizeof(EOBNRHMROMdata_interp));
  else
  {
    EOBNRHMROMdata_interp_Cleanup(*data_interp);
  }
  (*data_interp)->Camp_interp = NULL;
  (*data_interp)->Cphi_interp = NULL;
  (*data_interp)->shifttime_interp = NULL;
  (*data_interp)->shiftphase_interp = NULL;
}
void EOBNRHMROMdata_coeff_Init(EOBNRHMROMdata_coeff **data_coeff) {
  if(!data_coeff) exit(1);
  /* Create storage for structures */
  if(!*data_coeff) *data_coeff=malloc(sizeof(EOBNRHMROMdata_coeff));
  else
  {
    EOBNRHMROMdata_coeff_Cleanup(*data_coeff);
  }
  gsl_set_error_handler(&GSL_Err_Handler);
  (*data_coeff)->Camp_coeff = gsl_vector_alloc(nk_amp);
  (*data_coeff)->Cphi_coeff = gsl_vector_alloc(nk_phi);
  (*data_coeff)->shifttime_coeff = malloc(sizeof(double));
  (*data_coeff)->shiftphase_coeff = malloc(sizeof(double));
}
void EOBNRHMROMdata_Cleanup(EOBNRHMROMdata *data /* data to destroy */) {
  if(data->q) gsl_vector_free(data->q);
  if(data->freq) gsl_vector_free(data->freq);
  if(data->Camp) gsl_matrix_free(data->Camp);
  if(data->Cphi) gsl_matrix_free(data->Cphi);
  if(data->Bamp) gsl_matrix_free(data->Bamp);
  if(data->Bphi) gsl_matrix_free(data->Bphi);
  if(data->shifttime) gsl_vector_free(data->shifttime);
  if(data->shiftphase) gsl_vector_free(data->shiftphase);
  free(data);
}
void EOBNRHMROMdata_coeff_Cleanup(EOBNRHMROMdata_coeff *data_coeff) {
  if(data_coeff->Camp_coeff) gsl_vector_free(data_coeff->Camp_coeff);
  if(data_coeff->Cphi_coeff) gsl_vector_free(data_coeff->Cphi_coeff);
  if(data_coeff->shifttime_coeff) free(data_coeff->shifttime_coeff);
  if(data_coeff->shiftphase_coeff) free(data_coeff->shiftphase_coeff);
  free(data_coeff);
}
void EOBNRHMROMdata_interp_Cleanup(EOBNRHMROMdata_interp *data_interp) {
  if(data_interp->Camp_interp) SplineList_Destroy(data_interp->Camp_interp);
  if(data_interp->Cphi_interp) SplineList_Destroy(data_interp->Cphi_interp);
  if(data_interp->shifttime_interp) SplineList_Destroy(data_interp->shifttime_interp);
  if(data_interp->shiftphase_interp) SplineList_Destroy(data_interp->shiftphase_interp);
  free(data_interp);
}

/* Read binary ROM data for frequency vectors, coefficients matrices, basis functions matrices, and shiftvectors in time and phase */
int Read_Data_Mode(const char dir[], const int mode[2], EOBNRHMROMdata *data) {
  /* Load binary data for amplitude and phase spline coefficients as computed in Mathematica */
  int ret = SUCCESS;
  char* file_q = malloc(strlen(dir)+64);
  char* file_freq = malloc(strlen(dir)+64);
  char* file_Camp = malloc(strlen(dir)+64);
  char* file_Cphi = malloc(strlen(dir)+64);
  char* file_Bamp = malloc(strlen(dir)+64);
  char* file_Bphi = malloc(strlen(dir)+64);
  char* file_shifttime = malloc(strlen(dir)+64);
  char* file_shiftphase = malloc(strlen(dir)+64);
  sprintf(file_q, "%s", "EOBNRv2HMROM_q.dat"); /* The q vector is the same for all modes */
  sprintf(file_freq, "%s%d%d%s", "EOBNRv2HMROM_freq_", mode[0], mode[1], ".dat");
  sprintf(file_Camp, "%s%d%d%s", "EOBNRv2HMROM_Camp_", mode[0], mode[1], ".dat");
  sprintf(file_Cphi, "%s%d%d%s", "EOBNRv2HMROM_Cphi_", mode[0], mode[1], ".dat");
  sprintf(file_Bamp, "%s%d%d%s", "EOBNRv2HMROM_Bamp_", mode[0], mode[1], ".dat");
  sprintf(file_Bphi, "%s%d%d%s", "EOBNRv2HMROM_Bphi_", mode[0], mode[1], ".dat");
  sprintf(file_shifttime, "%s%d%d%s", "EOBNRv2HMROM_shifttime_", mode[0], mode[1], ".dat");
  sprintf(file_shiftphase, "%s%d%d%s", "EOBNRv2HMROM_shiftphase_", mode[0], mode[1], ".dat");
  ret |= Read_GSL_Vector(dir, file_q, data->q);
  ret |= Read_GSL_Vector(dir, file_freq, data->freq);
  ret |= Read_GSL_Matrix(dir, file_Camp, data->Camp);
  ret |= Read_GSL_Matrix(dir, file_Cphi, data->Cphi);
  ret |= Read_GSL_Matrix(dir, file_Bamp, data->Bamp);
  ret |= Read_GSL_Matrix(dir, file_Bphi, data->Bphi);
  ret |= Read_GSL_Vector(dir, file_shifttime, data->shifttime);
  ret |= Read_GSL_Vector(dir, file_shiftphase, data->shiftphase);
  free(file_q);
  free(file_freq);
  free(file_Camp);
  free(file_Cphi);
  free(file_Bamp);
  free(file_Bphi);
  free(file_shifttime);
  free(file_shiftphase);
  return(ret);
}

/* Function interpolating the data in matrix/vector form produces an interpolated data in the form of SplineLists */
int Interpolate_Spline_Data(const EOBNRHMROMdata *data, EOBNRHMROMdata_interp *data_interp) {

  gsl_set_error_handler(&GSL_Err_Handler);
  SplineList* splinelist;
  gsl_spline* spline;
  gsl_interp_accel* accel;
  gsl_vector* matrixline;
  gsl_vector* vector;
  int j;

  /* Interpolating Camp */
  splinelist = data_interp->Camp_interp;
  for (j = 0; j<nk_amp; j++) {
    matrixline = gsl_vector_alloc(nbwf);
    gsl_matrix_get_row(matrixline, data->Camp, j);

    accel = gsl_interp_accel_alloc();
    spline = gsl_spline_alloc(gsl_interp_cspline, nbwf);
    gsl_spline_init(spline, gsl_vector_const_ptr(data->q, 0), gsl_vector_const_ptr(matrixline, 0), nbwf);

    splinelist = SplineList_AddElementNoCopy(splinelist, spline,  accel, j);
    gsl_vector_free(matrixline);
  }
  data_interp->Camp_interp = splinelist;

  /* Interpolating Cphi */
  splinelist = data_interp->Cphi_interp;
  for (j = 0; j<nk_phi; j++) {
    matrixline = gsl_vector_alloc(nbwf);
    gsl_matrix_get_row(matrixline, data->Cphi, j);

    accel = gsl_interp_accel_alloc();
    spline = gsl_spline_alloc(gsl_interp_cspline, nbwf);
    gsl_spline_init(spline, gsl_vector_const_ptr(data->q, 0), gsl_vector_const_ptr(matrixline, 0), nbwf);

    splinelist = SplineList_AddElementNoCopy(splinelist, spline,  accel, j);
    gsl_vector_free(matrixline);
  }
  data_interp->Cphi_interp = splinelist;

  /* Interpolating shifttime */
  splinelist = data_interp->shifttime_interp;
  vector = data->shifttime;

  accel = gsl_interp_accel_alloc();
  spline = gsl_spline_alloc(gsl_interp_cspline, nbwf);
  gsl_spline_init(spline, gsl_vector_const_ptr(data->q, 0), gsl_vector_const_ptr(vector, 0), nbwf);

  splinelist = SplineList_AddElementNoCopy(NULL, spline,  accel, 0); /* This SplineList has only 1 element */
  data_interp->shifttime_interp = splinelist;

  /* Interpolating shiftphase */
  splinelist = data_interp->shiftphase_interp;
  vector = data->shiftphase;

  accel = gsl_interp_accel_alloc();
  spline = gsl_spline_alloc(gsl_interp_cspline, nbwf);
  gsl_spline_init(spline, gsl_vector_const_ptr(data->q, 0), gsl_vector_const_ptr(vector, 0), nbwf);

  splinelist = SplineList_AddElementNoCopy(NULL, spline,  accel, 0); /* This SplineList has only 1 element */
  data_interp->shiftphase_interp = splinelist;

  return SUCCESS;
}

/* Function taking as input interpolated data in the form of SplineLists
 * evaluates for a given q the projection coefficients and shifts in time and phase
*/
int Evaluate_Spline_Data(const double q, const EOBNRHMROMdata_interp* data_interp, EOBNRHMROMdata_coeff* data_coeff){

  SplineList* splinelist;
  /* Evaluating the vector of projection coefficients for the amplitude */
  for (int j=0; j<nk_amp; j++) {
    splinelist = SplineList_GetElement(data_interp->Camp_interp, j);
    gsl_vector_set(data_coeff->Camp_coeff, j, gsl_spline_eval(splinelist->spline, q, splinelist->accel));
  }
  /* Evaluating the vector of projection coefficients for the phase */
  for (int j=0; j<nk_phi; j++) {
    splinelist = SplineList_GetElement(data_interp->Cphi_interp, j);
    gsl_vector_set(data_coeff->Cphi_coeff, j, gsl_spline_eval(splinelist->spline, q, splinelist->accel));
  }
  /* Evaluating the shift in time */
  splinelist = SplineList_GetElement(data_interp->shifttime_interp, 0); /* This SplineList has only one element */
  *(data_coeff->shifttime_coeff) = gsl_spline_eval(splinelist->spline, q, splinelist->accel);
  /* Evaluating the shift in phase */
  splinelist = SplineList_GetElement(data_interp->shiftphase_interp, 0); /* This SplineList has only one element */
  *(data_coeff->shiftphase_coeff) = gsl_spline_eval(splinelist->spline, q, splinelist->accel);

  return SUCCESS;
}

/*
 * Core function for computing the ROM waveform.
 * Evaluates projection coefficients and shifts in time and phase at desired q.
 * Construct 1D splines for amplitude and phase.
 * Compute strain waveform from amplitude and phase.
*/
int EOBNRv2HMROMCore(
  ListAmpPhaseFDMode** listhlm,
  int nbmode,
  double deltatRef,
  double phiRef,
  double fRef,
  double Mtot_sec,
  double q,
  double distance,
  int setphiRefatfRef)
{
  int ret = SUCCESS;
  //int j;
  double tpeak22estimate = 0;
  /* Check output arrays */
  if(!listhlm) exit(1);
  if(*listhlm)
  {
    printf("Error: (*listhlm) is supposed to be NULL, but got %p\n",(*listhlm));
    exit(1);
  }
  /* Check number of modes */
  if(nbmode<1 || nbmode>nbmodemax) {
    printf("Error: incorrect number of modes: %d", nbmode);
    exit(1);
  }

  /* Check if the data has been set up */
  if(__EOBNRv2HMROM_setup) {
    printf("Error: the ROM data has not been set up\n");
    exit(1);
  }
  /* Set the global pointers to data */
  ListmodesEOBNRHMROMdata* listdata = *__EOBNRv2HMROM_data;
  ListmodesEOBNRHMROMdata_interp* listdata_interp = *__EOBNRv2HMROM_interp;

  /* Global amplitude prefactor - includes total mass scaling, Fourier scaling, distance scaling, and undoing an additional arbitrary scaling */
  double Mtot_msol = Mtot_sec / MTSUN_SI; /* Mtot_msol and M_ROM in units of solar mass */
  double amp0 = (Mtot_msol/M_ROM) * Mtot_sec * 1.E-16 * 1.E6 * PC_SI / distance;

  /* Highest allowed geometric frequency for the first mode of listmode in the ROM - used for fRef
   * by convention, we use the first mode of listmode (presumably the 22) for phiref */
  ListmodesEOBNRHMROMdata* listdata_ref = ListmodesEOBNRHMROMdata_GetMode(listdata, listmode[0][0], listmode[0][1]);
  EOBNRHMROMdata* data_ref = listdata_ref->data;
  double Mf_ROM_max_ref = gsl_vector_get(data_ref->freq, nbfreq-1);
  /* Convert to geometric units the reference time and frequency */
  double deltatRef_geom = deltatRef / Mtot_sec;
  double fRef_geom = fRef * Mtot_sec;

  /* Enforce allowed geometric frequency range for fRef */
  /* In case the user asks for a reference frequency higher than covered by the ROM, we keep it that way as we will just 0-pad the waveform (and do it anyway for some modes) */
  if (fRef_geom > Mf_ROM_max_ref || fRef_geom == 0)
    fRef_geom = Mf_ROM_max_ref; /* If fRef > fhigh or 0 we reset fRef to default value of cutoff frequency for the first mode of the list (presumably the 22 mode) */
  if (0 < fRef_geom && fRef_geom < Mf_ROM_min) {
    //printf("WARNING: Reference frequency Mf_ref=%g is smaller than lowest frequency in ROM Mf=%g. Setting it to the lowest frequency in ROM.\n", fRef_geom, Mf_ROM_min);
    fRef_geom = Mf_ROM_min;
  }

  /* Internal storage for the projection coefficients and shifts in time and phase */
  /* Initialized only once, and reused for the different modes */
  EOBNRHMROMdata_coeff *data_coeff = NULL;
  EOBNRHMROMdata_coeff_Init(&data_coeff);

  /* The phase change imposed by phiref, from the phase of the first mode in the list - to be set in the first step of the loop on the modes */
  double phase_change_ref = 0;

  /* Main loop over the modes */
  for(int i=0; i<nbmode; i++ ){
    int l = listmode[i][0];
    int m = listmode[i][1];

    /* Getting the relevant modes in the lists of data */
    ListmodesEOBNRHMROMdata* listdata_mode = ListmodesEOBNRHMROMdata_GetMode(listdata, l, m);
    ListmodesEOBNRHMROMdata_interp* listdata_interp_mode = ListmodesEOBNRHMROMdata_interp_GetMode(listdata_interp, l, m);

    /* Evaluating the projection coefficients and shift in time and phase */
    ret |= Evaluate_Spline_Data(q, listdata_interp_mode->data_interp, data_coeff);

    /* Evaluating the unnormalized amplitude and unshifted phase vectors for the mode */
    /* Notice a change in convention: B matrices are transposed with respect to the B matrices in SEOBNRROM */
    /* amp_pts = Bamp . Camp_coeff */
    /* phi_pts = Bphi . Cphi_coeff */
    gsl_vector* amp_f = gsl_vector_alloc(nbfreq);
    gsl_vector* phi_f = gsl_vector_alloc(nbfreq);
    //clock_t begblas = clock();
    gsl_blas_dgemv(CblasNoTrans, 1.0, listdata_mode->data->Bamp, data_coeff->Camp_coeff, 0.0, amp_f);
    gsl_blas_dgemv(CblasNoTrans, 1.0, listdata_mode->data->Bphi, data_coeff->Cphi_coeff, 0.0, phi_f);
    //clock_t endblas = clock();
    //printf("Mode (%d,%d) Blas time: %g s\n", l, m, (double)(endblas - begblas) / CLOCKS_PER_SEC);

     /* The downsampled frequencies for the mode - we undo the rescaling of the frequency for the 44 and 55 modes */
    gsl_vector* freq_ds = gsl_vector_alloc(nbfreq);
    gsl_vector_memcpy(freq_ds, listdata_mode->data->freq);
    if ( l==4 && m==4) gsl_vector_scale( freq_ds, 1./Scaling44(q));
    if ( l==5 && m==5) gsl_vector_scale( freq_ds, 1./Scaling55(q));

    /* Evaluating the shifts in time and phase - conditional scaling for the 44 and 55 modes */
    /* Note: the stored values of 'shifttime' correspond actually to 2pi*Deltat */
    SplineList* shifttime_splinelist = listdata_interp_mode->data_interp->shifttime_interp;
    SplineList* shiftphase_splinelist = listdata_interp_mode->data_interp->shiftphase_interp;
    double twopishifttime;
    if( l==4 && m==4) {
      twopishifttime = gsl_spline_eval(shifttime_splinelist->spline, q, shifttime_splinelist->accel) * Scaling44(q);
    }
    else if( l==5 && m==5) {
      twopishifttime = gsl_spline_eval(shifttime_splinelist->spline, q, shifttime_splinelist->accel) * Scaling55(q);
    }
    else {
      twopishifttime = gsl_spline_eval(shifttime_splinelist->spline, q, shifttime_splinelist->accel);
    }
    double shiftphase = gsl_spline_eval(shiftphase_splinelist->spline, q, shiftphase_splinelist->accel);

    /* If first mode in the list, assumed to be the 22 mode, set totalshifttime and phase_change_ref */
    if( i==0 ) {
      if(l==2 && m==2) {
      /* Setup 1d cubic spline for the phase of the 22 mode */
      gsl_interp_accel* accel_phi22 = gsl_interp_accel_alloc();
      gsl_spline* spline_phi22 = gsl_spline_alloc(gsl_interp_cspline, nbfreq);
      gsl_spline_init(spline_phi22, gsl_vector_const_ptr(freq_ds,0), gsl_vector_const_ptr(phi_f,0), nbfreq);
      /* Compute the shift in time needed to set the peak of the 22 mode roughly at deltatRef */
      /* We use the SPA formula tf = -(1/2pi)*dPsi/df to estimate the correspondence between frequency and time */
      /* The frequency corresponding to the 22 peak is omega22peak/2pi, with omega22peak taken from the fit to NR in Pan&al 1106 EOBNRv2HM paper */
      double f22peak = fmin(omega22peakOfq(q)/(2*PI), Mf_ROM_max_ref); /* We ensure we evaluate the spline within its range */
      /* Note : twopishifttime is almost 0 (order 1e-8) by construction for the 22 mode, so it does not intervene here */
      tpeak22estimate = -1./(2*PI) * gsl_spline_eval_deriv(spline_phi22, f22peak, accel_phi22);
      /* Determine the change in phase (to be propagated to all modes) required to have phi22(fRef) = 2*phiRef */
      if(setphiRefatfRef) {
        phase_change_ref = 2*phiRef + (gsl_spline_eval(spline_phi22, fRef_geom, accel_phi22) - (twopishifttime - 2*PI*tpeak22estimate + 2*PI*deltatRef_geom) * fRef_geom - shiftphase);
      }
      else {
        phase_change_ref = 2*phiRef;
      }
      gsl_spline_free(spline_phi22);
      gsl_interp_accel_free(accel_phi22);
      }
      else {
      	printf("Error: the first mode in listmode must be the 22 mode to set the changes in phase and time \n");
      	return FAILURE;
      }
    }
    /* Total shift in time, and total change in phase for this mode */
    double totaltwopishifttime = twopishifttime - 2*PI*tpeak22estimate + 2*PI*deltatRef_geom;
    double constphaseshift = m/2. * phase_change_ref + shiftphase;

    //
    //printf("deltatRef_geom: %g\n", deltatRef_geom);
    //printf("freq_ds 0: %g\n", gsl_vector_get(freq_ds, 0));

    //printf("2*PI*deltatRef_geom * gsl_vector_get(freq_ds, 0), phase_change_ref: %g, %g\n", 2*PI*deltatRef_geom * gsl_vector_get(freq_ds, 0), phase_change_ref);

    /* Initialize the complex series for the mode */
    // CAmpPhaseFrequencySeries *modefreqseries = NULL;
    // int len = (int) freq_ds->size;
    // CAmpPhaseFrequencySeries_Init(&modefreqseries, len);
    AmpPhaseFDMode* hlm = NULL;
    size_t len = freq_ds->size;
    AmpPhaseFDMode_Init(&hlm, len, len);

    /* Mode-dependent complete amplitude prefactor */
    double amp_pre = amp0 * ModeAmpFactor( l, m, q);

    /* Final result for the mode */
    /* Scale and set the amplitudes (amplitudes are real at this stage)*/
    gsl_vector_scale(amp_f, amp_pre);
    //gsl_vector_memcpy(modefreqseries->amp_real, amp_f);
    //gsl_vector_set_zero(modefreqseries->amp_imag); /* Amplitudes are real at this stage */
    /* Add the linear term and the constant (including the shift to phiRef), and set the phases */
    gsl_vector_scale(phi_f, -1.); /* Change the sign of the phases: ROM convention Psi=-phase */
    gsl_blas_daxpy(totaltwopishifttime, freq_ds, phi_f); /*Beware: here freq_ds must still be in geometric units*/
    gsl_vector_add_constant(phi_f, constphaseshift);
    //gsl_vector_memcpy(modefreqseries->phase, phi_f);

//
//printf("first phi_f: %g\n", gsl_vector_get(phi_f, 0 ));
//printf("last phi_f: %g\n", gsl_vector_get(phi_f, phi_f->size -1 ));

    /* Scale (to physical units) and set the frequencies */
    gsl_vector_scale(freq_ds, 1./Mtot_sec);
    //gsl_vector_memcpy(modefreqseries->freq, freq_ds);

    /* Append the computed mode to the ListmodesCAmpPhaseFrequencySeries structure */
    //*listhlm = ListmodesCAmpPhaseFrequencySeries_AddModeNoCopy(*listhlm, modefreqseries, l, m);
    real_vector_copy_gsl_vector(hlm->freq_amp, freq_ds);
    real_vector_copy_gsl_vector(hlm->amp_real, amp_f);
    real_vector_set_zero(hlm->amp_imag);
    real_vector_copy_gsl_vector(hlm->freq_phase, freq_ds);
    real_vector_copy_gsl_vector(hlm->phase, phi_f);
    *listhlm = ListAmpPhaseFDMode_AddMode(*listhlm, hlm, l, m);

    //
    //printf("Mode (%d,%d) generated:\n", l, m);
    //for( int j=0; j<len; j++ ){
    //  printf("%g %g %g %g\n", gsl_vector_get(modefreqseries->freq, j), gsl_vector_get(modefreqseries->amp_real, j), gsl_vector_get(modefreqseries->amp_imag, j), gsl_vector_get(modefreqseries->phase, j));
    //}

    /* Cleanup for the mode */
    gsl_vector_free(freq_ds);
    gsl_vector_free(amp_f);
    gsl_vector_free(phi_f);
  }

  /* Cleanup of the coefficients data structure */
  EOBNRHMROMdata_coeff_Cleanup(data_coeff);

  return(SUCCESS);
}

/* Compute waveform in downsampled frequency-amplitude-phase format */
int SimEOBNRv2HMROM(
  ListAmpPhaseFDMode** listhlm,                  /* Output: list of modes in Frequency-domain amplitude and phase form */
  int nbmode,                                    /* Number of modes to generate (starting with the 22) */
  double deltatRef,                              /* Time shift so that the peak of the 22 mode occurs at deltatRef */
  double phiRef,                                 /* Phase at reference frequency */
  double fRef,                                   /* Reference frequency (Hz); 0 defaults to fLow */
  double m1SI,                                   /* Mass of companion 1 (kg) */
  double m2SI,                                   /* Mass of companion 2 (kg) */
  double distance,                               /* Distance of source (m) */
  int setphiRefatfRef)                           /* Flag for adjusting the FD phase at phiRef at the given fRef, which depends also on tRef - if false, treat phiRef simply as an orbital phase shift (minus an observer phase shift) */
{
  /* Get masses in terms of solar mass */
  double mass1 = m1SI / MSUN_SI;
  double mass2 = m2SI / MSUN_SI;
  double Mtot = mass1 + mass2;
  double q = fmax(mass1/mass2, mass2/mass1);    /* Mass-ratio >1 by convention*/
  double Mtot_sec = Mtot * MTSUN_SI; /* Total mass in seconds */

  if ( q > q_max ) {
    //printf( "Error - %s: q out of range!\nEOBNRv2HMROM is only available for a mass ratio in the range q <= %g.\n", __func__, q_max);
    return FAILURE;
  }

  /* Set up (load and build interpolation) ROM data if not setup already */
  //clock_t beg = clock();
  EOBNRv2HMROM_Init_DATA();
  //clock_t end = clock();
  //printf("Initialization time: %g s\n", (double)(end - beg) / CLOCKS_PER_SEC);

  //beg = clock();
  int retcode = EOBNRv2HMROMCore(listhlm, nbmode, deltatRef, phiRef, fRef, Mtot_sec, q, distance, setphiRefatfRef);
  //end = clock();
  //printf("ROM evaluation time: %g s\n", (double)(end - beg) / CLOCKS_PER_SEC);

  return(retcode);
}

/* Setup EOBNRv2HMROM model using data files installed in $ROM_DATA_PATH */
int EOBNRv2HMROM_Init_DATA(void) {
  if (!__EOBNRv2HMROM_setup) return SUCCESS;

  int ret=FAILURE;
  char *envpath=NULL;
  char path[32768];
  char *brkt,*word;
  envpath=getenv("ROM_DATA_PATH");
  if(!envpath) {
    printf("Error: the environment variable ROM_DATA_PATH, giving the path to the ROM data, seems undefined\n");
    return(FAILURE);
  }
  strncpy(path,envpath,sizeof(path));

#pragma omp critical
  {
    for(word=strtok_r(path,":",&brkt); word; word=strtok_r(NULL,":",&brkt))
      {
	ret = EOBNRv2HMROM_Init(word);
	if(ret == SUCCESS) break;
      }
    if(ret!=SUCCESS) {
      printf("Error: unable to find EOBNRv2HMROM data files in $ROM_DATA_PATH\n");
      exit(FAILURE);
    }
    __EOBNRv2HMROM_setup = ret;
  }
  return(ret);
}

/* Setup EOBNRv2HMROM model using data files installed in dir */
int EOBNRv2HMROM_Init(const char dir[]) {
  if(!__EOBNRv2HMROM_setup) {
    printf("Error: EOBNRHMROMdata was already set up!");
    exit(1);
  }

  int ret = SUCCESS;
  ListmodesEOBNRHMROMdata* listdata = *__EOBNRv2HMROM_data;
  ListmodesEOBNRHMROMdata_interp* listdata_interp = *__EOBNRv2HMROM_interp;

  for (int j=0; j<nbmodemax; j++) { /* At setup, we initialize all available modes anyway */

    EOBNRHMROMdata* data = NULL;
    EOBNRHMROMdata_Init(&data);
    ret |= Read_Data_Mode( dir, listmode[j], data);
    if (!ret) {
      listdata = ListmodesEOBNRHMROMdata_AddModeNoCopy( listdata, data, listmode[j][0], listmode[j][1]);

      EOBNRHMROMdata_interp* data_interp = NULL;
      EOBNRHMROMdata_interp_Init(&data_interp);
      ret |= Interpolate_Spline_Data(data, data_interp);
      if (!ret) listdata_interp = ListmodesEOBNRHMROMdata_interp_AddModeNoCopy( listdata_interp, data_interp, listmode[j][0], listmode[j][1]);
    }
  }

  __EOBNRv2HMROM_setup = ret;
  if (!ret) {
    *__EOBNRv2HMROM_data = listdata;
    *__EOBNRv2HMROM_interp = listdata_interp;
  }
  return(ret);
}

/* Non-spinning merger TaylorF2 waveform, copied and condensed from LAL */
/* Used by SimEOBNRv2HMROMExtTF2 to extend the signal to arbitrarily low frequencies */
static void TaylorF2nonspin(
		double *amp,                            /**< FD waveform amplitude (modulus)*/
		double *phase,                          /**< FD waveform phase */
		const double *freqs,                    /**< frequency points at which to evaluate the waveform (Hz) */
		const int size,                         /** number of freq samples */
		const double m1_SI,                     /**< mass of companion 1 (kg) */
		const double m2_SI,                     /**< mass of companion 2 (kg) */
		const double distance,                  /** distance (m) */
		const double imatch                     /**< index at which to match phase;
							   assumes arrays are preloaded at imatch and imatch+1
							   with the required result */
		     )
{
  //The meat of this computation is copied from LAL: XLALSimInspiralPNPhasing_F2
  //We dont need the spin terms
  double m1 = m1_SI / MSUN_SI;
  double m2 = m2_SI / MSUN_SI;
  double mtot = m1 + m2;
  //double d = (m1 - m2) / (m1 + m2);
  double eta = m1*m2/mtot/mtot;
  //double m1M = m1/mtot;
  //double m2M = m2/mtot;
  double m_sec = mtot * MTSUN_SI;
  double piM = PI * m_sec;

  double pfaN = 3.L/(128.L * eta);

  /* Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18 */
  //double pfav0 = 1.L;
  double pfav2 = 5.L*(743.L/84.L + 11.L * eta)/9.L;
  double pfav3 = -16.L*PI;
  double pfav4 = 5.L*(3058.673L/7.056L + 5429.L/7.L * eta
		+ 617.L * eta*eta)/72.L;
  double pfav5 = 5.L/9.L * (7729.L/84.L - 13.L * eta) * PI;
  double pfalogv5 = 5.L/3.L * (7729.L/84.L - 13.L * eta) * PI;
  double pfav6 = (11583.231236531L/4.694215680L
	    - 640.L/3.L * PI * PI - 6848.L/21.L*GAMMA)
    + eta * (-15737.765635L/3.048192L
	     + 2255./12. * PI * PI)
    + eta*eta * 76055.L/1728.L
    - eta*eta*eta * 127825.L/1296.L;
  pfav6 += (-6848.L/21.L)*log(4.);
  double pfalogv6 = -6848.L/21.L;
  double pfav7 = PI * ( 77096675.L/254016.L
		      + 378515.L/1512.L * eta - 74045.L/756.L * eta*eta);

  /* Non-spin 2-2 amplitude terms (Blanchet LRR)*/
  double a2 = ( -107 + 55*eta ) / 42.;
  double a3 = 2*PI;
  double a4 = ( ( 2047.*eta - 7483. ) * eta - 2173. ) / 1512.;
  /* Blanchet has more terms, but there should diminishing returns:
     expect v^5 ~ 1e-5 and the higher terms are more complicated and, indeed, complex */


  //Lead coefficients
  //double amp0 = -4. * m1 * m2 / distance * C_SI * MTSUN_SI * MTSUN_SI * sqrt(PI/12.L); //(from LAL)
  double amp0B = 2. * m1 * m2 / distance * C_SI * MTSUN_SI * MTSUN_SI * sqrt(16*PI/5.L); //Based on Blanchet-LRR (327)
  //Note: amp0B = -4 * sqrt( 3/5) * amp0;
  double FTaN =  32.0 * eta*eta / 5.0;
  //printf("eta=%g\n",eta);
  //Compute raw TaylorF2
  int i;
  for (i = 0; i < size; i++) {
    double f = freqs[i];
    double v = cbrt(piM*f);
    double logv = log(v);
    double v2 = v*v;
    double v5 = v2*v2*v;
    double v10 = v5*v5;

    //printf("taylorf2: f=%g  v=%g\n",f,v);
    double phasing=0;
    phasing = pfav7 * v;
    phasing = (phasing + pfav6 + pfalogv6 * logv) * v;
    phasing = (phasing + pfav5 + pfalogv5 * logv) * v;
    phasing = (phasing + pfav4) * v;
    phasing = (phasing + pfav3) * v;
    phasing = (phasing + pfav2) * v2;
    phasing += 1;
    phasing *=pfaN;

    double amp22fac;
    amp22fac = a4*v;
    amp22fac = ( amp22fac + a3 ) * v;
    amp22fac = ( amp22fac + a2 ) * v2;
    amp22fac += 1.0;

    phasing /= v5;
    double flux = FTaN * v10;
    double dEnergy = -eta * v;
    phase[i]=phasing;
    //Notes for amplitude: Blanchet at leading order:
    /* mf=x^(3/2); fdot=3/2/m x^(1/2) xdot ~ 3/2/m x^(1/2) * (-1/16)*(4x)^5(-eta/5/m) = 96/5*eta/m^2 * x^(11/2) */
    /*-flux/dEnergy = 32.0 * eta*eta / 5.0 / eta *v^9 */
    /*--> -flux/dEnergy =  fdot / (3*v^2) [using x=v^2]*/
    //amp[i] = amp0 * sqrt(-dEnergy/flux) * v;  (Based on LAL)
    amp[i] = amp0B * amp22fac * v2 / sqrt(-flux/dEnergy * 3 * v2 );
    //printf("v=%g: a=%g ph=%g;  amp22fac=%g sqrt(v^-9)=%g\n",v,amp[i],phase[i],amp22fac,sqrt(v/v10));
    //Note ampB = - amp * 4*sqrt(3/5) * / sqrt(3) + higher-order = -4/sqrt(5)*( 1 + higher-order )
    //  ...possibly related to sph-harm normalization
    // HACK: Strangely it seems that an additional factor of 4/sqrt(5) is just right to nearly match the EOB wf FT
    amp[i] *= 4/sqrt(5);   //HACK

    //Here we depart from LAL, referencing phase and time-shift to two neighboring freq points
    //First we match the freq derivative
  }
}

/*Wrapper for waveform generation with possibly a combination of EOBNRv2HMROM and TaylorF2*/
/* Note: GenerateWaveform accepts masses and distances in SI units, whereas LISA params is in solar masses and Mpc */
/* Note: the extended waveform will now a different number of frequency points for each mode */
/* Note: phiRef is readjusted after the extension -- for the case where fRef is below the band covered by ROM, in which case the core function defaults fRef to the max geometric freq of the ROM */
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
  int setphiRefatfRef)                           /* Flag for adjusting the FD phase at phiRef at the given fRef, which depends also on tRef - if false, treat phiRef simply as an orbital phase shift (minus an observer phase shift) */
{//
  //printf("calling SimEOBNRv2HMROMExtTF2 with minf=%g\n", minf);
  //printf("params: %d %g %g %g %g %g %g %g %g\n", nbmode, Mf_match, minf, deltatRef, phiRef, fRef, m1SI, m2SI, distance);


  int ret;
  int i;
  ListAmpPhaseFDMode* listROM = NULL;
  //int lout=-1,mout=-1;
  int lout=-1, mout=-1;

  /* Generate the waveform with the ROM */
  ret = SimEOBNRv2HMROM(&listROM, nbmode, deltatRef, phiRef, fRef, m1SI, m2SI, distance, setphiRefatfRef);

  /* If the ROM waveform generation failed (e.g. parameters were out of bounds) return FAILURE */
  //if(ret==FAILURE)printf("SimEOBNRv2HMROMExtTF2: Generation of ROM for injection failed!\n");
  if(ret==FAILURE) return FAILURE;

  /* Main loop over the modes (as linked list) to perform the extension */
  /* The 2-2 mode will be extended by TaylorF2 model with the phase and time offset
  determined by matching conditions. All other modes will be extended as some
  sort of power-law fall-off in amplitude and power-law growth in phase.          */
  ListAmpPhaseFDMode* listelement = listROM;
  while (listelement) {    // For each l-m (ie each listelement)

    /* Definitions: l,m, frequency series and length */
    int l = listelement->l;
    int m = listelement->m;

    //NOTE: temporary hack to allow avoiding power-law extension of higher modes which seems problematic
    //    if((l==2&&m==2) || tagexthm) {

    /* First we must compute a new frequency grid including a possible extension to lower frequencies*/
    gsl_vector* freq_new;
    /* NOTE: freq_amp and freq_phase are identical here */
    real_vector* freq = listelement->hlm->freq_phase;
    int len = (int) freq->size;
    // Construct frequency grid extension on the geometric mean of the lowest few ROM frequencies after the matching point
    const int Navg=3;
    int imatch=-1;
    double f_match;
    if(Mf_match<=0) {
      f_match = Mf_ROM_min/(m1SI+m2SI)*MSUN_SI/MTSUN_SI;
    }
    else if(Mf_match>Mf_ROM_min) {
      f_match = Mf_match/(m1SI+m2SI)*MSUN_SI/MTSUN_SI;
    }
    else {
      printf("WARNING: Mf_match lower than the lowest geometric frequency of the ROM model - used instead\n");
      f_match = Mf_ROM_min/(m1SI+m2SI)*MSUN_SI/MTSUN_SI;
    }
    f_match = f_match*m/2; //Shift the matching frequency for non-22 modes to correspond to a comparable orbital freq.
    //printf("f_match=%g\n",f_match);
    for(i=0; i<len-Navg; i++){
      if(real_vector_get(freq,i)>f_match){
        imatch=i;
        break;
      }
    }
    if(imatch<0){
      printf("WARNING: f_match exceeds high-freq range of the ROM model\n");
      imatch=len-Navg-1;
    }
    double lnfmatch=log(real_vector_get(freq,imatch));
    double lnfmin=log(minf);
    double dlnf=(log(real_vector_get(freq,imatch+Navg))-lnfmatch)/Navg;
    double dffac=exp(dlnf);
    //The new grid will include the old grid, from imatch and after, plus adequate lower-freq
    int len_add = (lnfmatch-lnfmin)/dlnf;
    if(len_add<0) len_add=0;
    int len_new = len - imatch + len_add;
    //printf("extending waveform: len_add + len - imatch = len_new: %i + %i - %i = %i\n",len_add,len,imatch,len_new);
    /* construct the extended freq grid */
    freq_new=gsl_vector_alloc(len_new);
    for(i=len_add;i<len_new;i++)gsl_vector_set(freq_new,i,real_vector_get(freq,len-len_new+i));
    for(i=len_add;i>0;i--)gsl_vector_set(freq_new,i-1,gsl_vector_get(freq_new,i)/dffac);
    //for(i=0;i<len_new;i++)printf("%i: %g %g\n",i,(i>=len_new-len?freq->data[i-len_new+len]:0),freq_new->data[i]);


    //copy the old freqseries data to a new one and extend with power-law
    AmpPhaseFDMode* freqseries = listelement->hlm;
    if(l==lout&&m==mout){  //write to file for debugging
      printf("Writing waveform-before for (%i,%i)\n",l,m);
      FILE *f = fopen("waveform-before.dat", "w");
      for(i=0; i < (int) freqseries->freq_phase->size; i++){
        fprintf(f,"%i %i %g  %g  %g  %g %g %i\n",l,m,
        real_vector_get(freqseries->freq_amp,i),
        real_vector_get(freqseries->amp_real,i),
        real_vector_get(freqseries->amp_imag,i),
        real_vector_get(freqseries->freq_phase,i),
        real_vector_get(freqseries->phase,i),
        (int) i);
      }
      fclose(f);
    }
    AmpPhaseFDMode* freqseries_new = NULL;     //New result will be assembled here
    AmpPhaseFDMode_Init(&freqseries_new, len_new, len_new);
    //set the new freqs
    for(i=0;i<len_new;i++){
      real_vector_set(freqseries_new->freq_amp, i, gsl_vector_get(freq_new, i));
      real_vector_set(freqseries_new->freq_phase, i, gsl_vector_get(freq_new, i));
    }
    //copy in the high-freq ROM-model data
    //printf("l,m = %i,%i;  lenghts=%i,%i\n",l,m,freqseries->freq->size, freqseries_new->freq->size);
    for(i=len_add;i<len_new;i++){
      //printf("i, len-len_new+i: %i, %i\n",i,len-len_new+i);
      real_vector_set(freqseries_new->amp_real, i, real_vector_get(freqseries->amp_real,len-len_new+i));
      real_vector_set(freqseries_new->amp_imag, i, real_vector_get(freqseries->amp_imag,len-len_new+i));
      real_vector_set(freqseries_new->phase, i, real_vector_get(freqseries->phase,len-len_new+i));
      //printf("%i: copying %g  %g  %g  %g\n",i,freqseries_new->freq->data[i],freqseries_new->amp_real->data[i],freqseries_new->amp_imag->data[i],freqseries_new->phase->data[i]);
    }
    //extend
    if(l==2&&m==2&&len_add>0) {//extend 2-2 with TaylorF2
      //Assemble data for matching
      double f0=freq_new->data[len_add],f1=freq_new->data[len_add+1];
      double ph0=freqseries_new->phase->data[len_add],ph1=freqseries_new->phase->data[len_add+1];
      double amp=sqrt(freqseries_new->amp_real->data[len_add]*freqseries_new->amp_real->data[len_add]
        +freqseries_new->amp_imag->data[len_add]*freqseries_new->amp_imag->data[len_add]);
      double amp1=sqrt(freqseries_new->amp_real->data[len_add+1]*freqseries_new->amp_real->data[len_add+1]
        +freqseries_new->amp_imag->data[len_add+1]*freqseries_new->amp_imag->data[len_add+1]);
      double amprfac=freqseries_new->amp_real->data[len_add]/amp,ampifac=freqseries_new->amp_imag->data[len_add]/amp;
      //Compute raw TaylorF2
      TaylorF2nonspin(freqseries_new->amp_real->data,freqseries_new->phase->data,freq_new->data,len_add+2,m1SI,m2SI,distance,imatch);
      //Compute offsets in phase, first phase derivative, and amplitude argument
      double dphase0tf2=(freqseries_new->phase->data[len_add+1]-freqseries_new->phase->data[len_add])/(f1-f0);
      double phase0tf2=freqseries_new->phase->data[len_add];
      double dphase0eob=(ph1-ph0)/(f1-f0);
      double dphase0=dphase0eob-dphase0tf2;
      double phase0=ph0 - phase0tf2 - f0*dphase0;
      double amp0bcoeff = amp / freqseries_new->amp_real->data[len_add] - 1.0; //Compute correction for continuity matching.
      double amp0ccoeff = ((amp1/freqseries_new->amp_real->data[len_add+1]-1.0)/amp0bcoeff/(f1*f1/f0/f0)-1.0)/(f1/f0-1.0);
      //TESTING
      //printf("%g, %g\n", amp0bcoeff, amp0ccoeff);
      //Compute correction for continuity matching.
      /*
      printf("ph0eob,dph0eob= %g,  %g\n",ph0,dphase0eob);
      printf("ph0tf2,dph0tf2= %g,  %g\n",phase0tf2,dphase0tf2);
      printf("ph0,dph0= %g,  %g\n",phase0,dphase0);
      printf("f0,f0*dph0= %g,  %g\n",f0,dphase0*f0);
      printf("imatch=%i\n",imatch);
      */
      //Apply offsets
      for (i = 0; i < len_add+2; i++){
        /* NOTE: freq_amp and freq_phase are identical here */
        double f=freqseries_new->freq_phase->data[i];
        //printf("%i<%i,%g\n",i,len_add,len_add-i);
        //printf("f,ph0+f*dph0= %g, %g\n",freqseries_new->freq->data[i],phase0 + dphase0*f);
        freqseries_new->phase->data[i] += phase0 + dphase0*f;
        //First apply continuity matching
        // amp -> amp * ( 1 + b*f^2/f0^2 * ( 1 + c*(f/f0 -1) )
        //(starts at order f^2 since we only keep 2PN order ampl corrections in TaylorF2 code below; could change to f^3 if higher order terms are used)
        freqseries_new->amp_real->data[i] *= 1.0 + amp0bcoeff*f*f/f0/f0*(1+amp0ccoeff*(f/f0-1));
        freqseries_new->amp_imag->data[i] = freqseries_new->amp_real->data[i]*ampifac;
        freqseries_new->amp_real->data[i] *= amprfac;
        //printf("%i: extending TF2 %g  %g  %g  %g\n",i,freqseries_new->freq->data[i],freqseries_new->amp_real->data[i],freqseries_new->amp_imag->data[i],freqseries_new->phase->data[i]);
      }
    } else { //extend other modes with power-law
      //The results are many cycles out of phase almost immediately, so this definitely is not an accurate
      //waveform, but the results are reasonably smooth and of plausible structure.
      //Alternatively, we could also extend these with TaylorF2, btu we are mostly assuming this part of the WF is small
      double phref=freqseries->phase->data[len-1];//For phase we extend by a power-law referenced to zero phase at end of ROM
      for(i=0;i<len;i++)if(phref>freqseries->phase->data[i])phref=freqseries->phase->data[i];//get the smallest value of phi to use as ref.
      phref=phref+1.0;//add one more
      double ldArfac = 0;
      if(real_vector_get(freqseries->amp_real,imatch)>0) //avoid div0 in trivial cases
      //dArfac=exp(-log( gsl_vector_get(freqseries->amp_real,imatch+Navg)
      //		 /gsl_vector_get(freqseries->amp_real,imatch) ) / Navg);
      /* NOTE: freq_amp and freq_phase are identical here */
      ldArfac=(log( real_vector_get(freqseries->amp_real,imatch+Navg)
      /real_vector_get(freqseries->amp_real,imatch) ) /
      log( real_vector_get(freqseries->freq_phase,imatch+Navg)
      /real_vector_get(freqseries->freq_phase, imatch) )  );
      //double dphfac = exp(-log( (gsl_vector_get(freqseries->phase,imatch+Navg)-phref)
      //			/(gsl_vector_get(freqseries->phase,imatch) -phref)) / Navg);
      double ldphfac = (log( (real_vector_get(freqseries->phase,imatch+Navg)-phref)
      /(real_vector_get(freqseries->phase,imatch) -phref)) /
      log( real_vector_get(freqseries->freq_phase,imatch+Navg)
      /real_vector_get(freqseries->freq_phase,imatch) )  );
      if(1&&l==lout&&m==mout)printf("ldphfac(%i,%i)=%g\n",l,m,ldphfac);
      //double f0=gsl_vector_get(freqseries->freq,imatch);
      for(i=len_add;i>0;i--){
        /* NOTE: freq_amp and freq_phase are identical here */
        double fratio=real_vector_get(freqseries_new->freq_phase,i-1)/real_vector_get(freqseries_new->freq_phase,i);
        double dArfac=pow(fratio,ldArfac);
        real_vector_set(freqseries_new->amp_real,i-1,real_vector_get(freqseries_new->amp_real,i)*dArfac);
        real_vector_set(freqseries_new->amp_imag,i-1,real_vector_get(freqseries_new->amp_imag,i)*dArfac);
        double dphfac=pow(fratio,ldphfac);
        real_vector_set(freqseries_new->phase,i-1,(real_vector_get(freqseries_new->phase,i)-phref)*dphfac+phref);
        //printf("%i: extending %g  %g  %g  %g\n",i,freqseries_new->freq->data[i-1],freqseries_new->amp_real->data[i-1],freqseries_new->amp_imag->data[i-1],freqseries_new->phase->data[i-1]);
      }
      //printf("Extended (%i,%i) down to f=%g, ampR=%g, ampI=%g, phase=%g\n",l,m,freqseries_new->freq->data[0],freqseries_new->amp_real->data[0],freqseries_new->amp_imag->data[0],freqseries_new->phase->data[0]);
    }
    //delete the old content data and replace with the new
    AmpPhaseFDMode_Destroy(freqseries);
    listelement->hlm=freqseries_new;

    //TESTING
    //printf("In ext: len freqseries: %d\n", freqseries_new->freq->size);
    //for(int i=0; i<freqseries_new->freq->size; i++) {
    //printf("%g %g %g %g\n", gsl_vector_get(freqseries_new->freq, i), gsl_vector_get(freqseries_new->amp_real, i), gsl_vector_get(freqseries_new->amp_imag, i), gsl_vector_get(freqseries_new->phase, i));
    //}

    //debugging
    //   freqseries=listelement->freqseries;
    //   if(1&&l==lout&&m==mout){  //write to file for debugging
    //     FILE *f = fopen("waveform.dat", "w");
    //     for(i=0;i<freqseries->freq->size;i++){
    // fprintf(f,"%i %i %g  %g %g  %g %i\n",l,m,
    // 	freqseries->freq->data[i],
    // 	freqseries->amp_real->data[i],
    // 	freqseries->amp_imag->data[i],
    // 	freqseries->phase->data[i],
    // 	i);
    //     }
    //     fclose(f);
    //   }

    gsl_vector_free(freq_new);
    listelement=listelement->next;
  }

  /* Compute phase shift to set phiRef at fRef */
  /* Covers the case where input fRef is outside the range of the ROM, in which case the Core function defaulted to Mfmax_ROM */
  /* Not very clean and a bit redundant */
  if (setphiRefatfRef) {
    AmpPhaseFDMode* h22 = ListAmpPhaseFDMode_GetMode(listROM, 2, 2)->hlm;
    gsl_vector_view freq22 = gsl_vector_view_array(h22->freq_phase->data, h22->freq_phase->size);
    gsl_vector_view phase22 = gsl_vector_view_array(h22->phase->data, h22->phase->size);;
    int nbfreq = (&freq22.vector)->size;
    gsl_interp_accel* accel_phi22 = gsl_interp_accel_alloc();
    gsl_spline* spline_phi22 = gsl_spline_alloc(gsl_interp_cspline, nbfreq);
    gsl_spline_init(spline_phi22, gsl_vector_const_ptr((&freq22.vector),0), gsl_vector_const_ptr(&phase22.vector,0), nbfreq);
    /* If fRef was not set (fRef=0), use the last frequency generated for 22 mode -- as is done internally in Core */
    if (fRef==0.) fRef = (&freq22.vector)->data[(&freq22.vector)->size - 1];
    /* Compute 22 phase at fRef before adjustment, check the extended range */
    if ( (fRef<(&freq22.vector)->data[0]) || (fRef>(&freq22.vector)->data[(&freq22.vector)->size - 1]) ) {
      printf("Error: fRef is not covered by the frequency range of the waveform after extension.\n");
      return FAILURE;
    }
    double phi22atfRef = gsl_spline_eval(spline_phi22, fRef, accel_phi22);
    /* Phase shift, as an orbital or observer phase shift (change in 22 is 2*phaseshift) */
    double phaseshift = (2*phiRef - phi22atfRef)/2.;

    /* Propagate phase shift to full list of modes */
    listelement = listROM;
    while(listelement) {
      int m = listelement->m;

      double phaseshiftlm = m/2. * phaseshift;
      real_vector* philm = listelement->hlm->phase;
      real_vector_add_constant(philm, phaseshiftlm);

      listelement = listelement->next;
    }

    /* Cleanup */
    gsl_spline_free(spline_phi22);
    gsl_interp_accel_free(accel_phi22);
  }


  /* Output */
  *listhlm = listROM;

      /*
      printf("generated listROM: n=%i l=%i m=%i\n",(*listhlm)->freqseries->amp_real->size,listROM->l,listROM->m);
      for(i=0;i<(*listhlm)->freqseries->freq->size;i++){
      printf("%i:  %g  %g  %g  %g\n",i,(*listhlm)->freqseries->freq->data[i],(*listhlm)->freqseries->amp_real->data[i],(*listhlm)->freqseries->amp_imag->data[i],(*listhlm)->freqseries->phase->data[i]);
    }
    printf("listlhm=%x\n",listhlm);
    printf("*listlhm=%x\n",*listhlm);
    */
  return SUCCESS;
}
