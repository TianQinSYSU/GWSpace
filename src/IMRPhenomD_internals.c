/*
 * Copyright (C) 2015 Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

// LAL independent code (C) 2017 Michael Puerrer

/**
 * \author Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
 *
 * \file
 *
 * \brief Internal function for IMRPhenomD phenomenological waveform model.
 * See \ref LALSimIMRPhenom_c for more details.
 *
 * Note that the functions defined here in LALSimIMRPhenomD_internals.c / .h
 * must be used by BOTH LALSimIMRPhenomD.c AND LALSimIMRPhenomP.c.
 */

/*
This waveform uses the TaylorF2 coefficients for it's inspiral phase augmented
by higher order phenomenological terms tuned to SEOBv2-Hybrid waveforms.
Below are lines copied from LALSimInspiralPNCoefficients.c which are the TaylorF2
phase coefficients we have used.
We document them here in case changes to that file changes the behaviour
of this waveform.

    const double mtot = m1 + m2;
    const double d = (m1 - m2) / (m1 + m2);
    const double eta = m1*m2/mtot/mtot;
    const double m1M = m1/mtot;
    const double m2M = m2/mtot;
    // Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
    // We write dSigmaL for their (\delta m/m) * \Sigma_\ell
    // There's a division by mtotal^2 in both the energy and flux terms
    // We just absorb the division by mtotal^2 into SL and dSigmaL

    const double SL = m1M*m1M*chi1L + m2M*m2M*chi2L;
    const double dSigmaL = d*(m2M*chi2L - m1M*chi1L);

    const double pfaN = 3.L/(128.L * eta);
    //Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18
    pfa->v[0] = 1.L;
    pfa->v[2] = 5.L*(743.L/84.L + 11.L * eta)/9.L;
    pfa->v[3] = -16.L*PI;
    pfa->v[4] = 5.L*(3058.673L/7.056L + 5429.L/7.L * eta
                     + 617.L * eta*eta)/72.L;
    pfa->v[5] = 5.L/9.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->vlogv[5] = 5.L/3.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->v[6] = (11583.231236531L/4.694215680L
                     - 640.L/3.L * PI * PI - 6848.L/21.L*GAMMA)
                 + eta * (-15737.765635L/3.048192L
                     + 2255./12. * PI * PI)
                 + eta*eta * 76055.L/1728.L
                 - eta*eta*eta * 127825.L/1296.L;
    pfa->v[6] += (-6848.L/21.L)*log(4.);
    pfa->vlogv[6] = -6848.L/21.L;
    pfa->v[7] = PI * ( 77096675.L/254016.L
                     + 378515.L/1512.L * eta - 74045.L/756.L * eta*eta);

    // Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16
    const double pn_gamma = (554345.L/1134.L + 110.L*eta/9.L)*SL + (13915.L/84.L - 10.L*eta/3.)*dSigmaL;
    switch( spinO )
    {
        case LAL_SIM_INSPIRAL_SPIN_ORDER_ALL:
        case LAL_SIM_INSPIRAL_SPIN_ORDER_35PN:
            pfa->v[7] += (-8980424995.L/762048.L + 6586595.L*eta/756.L - 305.L*eta*eta/36.L)*SL - (170978035.L/48384.L - 2876425.L*eta/672.L - 4735.L*eta*eta/144.L) * dSigmaL;
*/

#include "IMRPhenomD_internals.h"

/*
 *
 * Internal function implementations
 *
 */

////////////////////////////// Miscellaneous functions //////////////////////////////

COMPLEX16FrequencySeries *CreateCOMPLEX16FrequencySeries(
    const char *name,
    long epoch,
    double f0,
    double deltaF,
    size_t length
){
    COMPLEX16FrequencySeries* fs = (COMPLEX16FrequencySeries*)malloc(sizeof(COMPLEX16FrequencySeries));

    size_t name_length = strlen(name);
    fs->name = (char*)malloc((name_length+1) * sizeof(char));
    strcpy(fs->name, name);

    fs->epoch = epoch;
    fs->f0 = f0;
    fs->deltaF = deltaF;

    fs->length = length;
    fs->data = (double complex *)malloc(sizeof(double complex) * length);
    if (fs->data == NULL)
        ERROR(PD_ENOMEM, "Failed to allocated data array.");

    memset(fs->data, 0, sizeof(double complex) * length);
    return fs;
}

COMPLEX16FrequencySeries *ResizeCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs, size_t length) {
    // assume that we start at index 0
    free(fs->data);

    fs->length = length;
    fs->data = (double complex *)malloc(sizeof(double complex) * length);
    if (fs->data == NULL)
        ERROR(PD_ENOMEM, "Failed to allocated data array.");

    return fs;
}

void DestroyCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs) {
    free(fs->data);
    free(fs->name);
    free(fs);
}

///////////////////////////////////////////////////////////////////////////////
// SM: structure for downsampled FD 22-mode Amp/Phase waveforms

AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(
    size_t length
){
    AmpPhaseFDWaveform* wf = (AmpPhaseFDWaveform*)malloc(sizeof(AmpPhaseFDWaveform));
    wf->length = length;
    wf->freq = (double*) malloc(sizeof(double) * length);
    wf->amp = (double*) malloc(sizeof(double) * length);
    wf->phase = (double*) malloc(sizeof(double) * length);
    if ((wf->freq == NULL) || (wf->amp == NULL) || (wf->phase == NULL) )
        ERROR(PD_ENOMEM, "Failed to allocated one of the data arrays.");

    memset(wf->freq, 0, sizeof(double) * length);
    memset(wf->amp, 0, sizeof(double) * length);
    memset(wf->phase, 0, sizeof(double) * length);
    return wf;
}

void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform *wf) {
    free(wf->freq);
    free(wf->amp);
    free(wf->phase);
    free(wf);
}

RealVector* CreateRealVector(
    size_t length
){
    RealVector* v = (RealVector*)malloc(sizeof(RealVector));
    v->length = length;
    v->data = (double*) malloc(sizeof(double) * length);
    if (v->data == NULL)
        ERROR(PD_ENOMEM, "Failed to allocated one of the data arrays.");

    memset(v->data, 0, sizeof(double) * length);
    return v;
}

void DestroyRealVector(RealVector *v) {
    free(v->data);
    free(v);
}

///////////////////////////////////////////////////////////////////////////////

// Simplified code from lal/std/XLALError.c
/* Return the error message associated with an error number or return value. */
const char *ErrorString(int code) {

    if (code <= 0) {    /* this is a return code, not an error number */
        if (code == 0)
            return "PD_SUCCESS";
        else if (code == -1)
            return "Failure";
        else
            return "Unknown return code";
    }

    /* check to see if an internal function call has failed, but the error
     * number was not "or"ed against the mask PD_EFUNC */
    if (code == PD_EFUNC)
        return "Internal function call failed";

    /* use this to report error strings... deals with possible mask for
     * errors arising from internal function calls */
# define ERROR_STRING(s) \
    ( ( code & PD_EFUNC ) ? "Internal function call failed: " s : (const char *) s )
    switch (code & ~PD_EFUNC) {
        /* these are standard error numbers */
    case PD_EFAULT:
        return ERROR_STRING("Invalid pointer");
    case PD_EDOM:
        return ERROR_STRING("Input domain error");
    case PD_ENOMEM:
        return ERROR_STRING("Memory allocation error");
        /* unrecognized error number */
    default:
        return "Unknown error";
    }
# undef ERROR_STRING
    return NULL;        /* impossible to get here */
}

// FIXME: want file and line number where the error occurred
void ERROR(ERROR_type e, char *errstr) {
    fprintf(stderr, "%s in %s:%d:\n %s\n", ErrorString(e), __FILE__, __LINE__, errstr);
    exit(-1);
}

void CHECK(bool assertion, ERROR_type e, char *errstr) {
    if (!(assertion))
        ERROR(e, errstr);
}

void PRINT_WARNING(char *warnstr) {
    fprintf(stderr, "Warning: %s\n", warnstr);
}

///////////////////////////////////////////////////////////////////////////////

// From LALSimInspiralTaylorF2.c
/** \brief Returns structure containing TaylorF2 phasing coefficients for given
 *  physical parameters.
 */
int TaylorF2AlignedPhasing(
        PNPhasingSeries **pn,   /**< phasing coefficients (output) */
        const double m1,         /**< mass of body 1 */
        const double m2,		/**< mass of body 2 */
        const double chi1,	/**< aligned spin parameter of body 1 */
        const double chi2	/**< aligned spin parameter of body 2 */
	)
{
    PNPhasingSeries *pfa;

    if (!pn) ERROR(PD_EFAULT, "");
    if (*pn) ERROR(PD_EFAULT, "");

    pfa = (PNPhasingSeries *) malloc(sizeof(PNPhasingSeries));

    PNPhasing_F2(pfa, m1, m2, chi1, chi2, chi1*chi1, chi2*chi2, chi1*chi2);

    *pn = pfa;

    return PD_SUCCESS;
}

// From LALSimInspiralPNCoefficients.c
/* The phasing function for TaylorF2 frequency-domain waveform.
 * This function is tested in ../test/PNCoefficients.c for consistency
 * with the energy and flux in this file.
 */
void PNPhasing_F2(
	PNPhasingSeries *pfa, /**< \todo UNDOCUMENTED */
	const double m1, /**< Mass of body 1, in Msol */
	const double m2, /**< Mass of body 2, in Msol */
	const double chi1L, /**< Component of dimensionless spin 1 along Lhat */
	const double chi2L, /**< Component of dimensionless spin 2 along Lhat */
	const double chi1sq,/**< Magnitude of dimensionless spin 1 */
	const double chi2sq, /**< Magnitude of dimensionless spin 2 */
	const double chi1dotchi2 /**< Dot product of dimensionles spin 1 and spin 2 */
	)
{
    const double mtot = m1 + m2;
    const double d = (m1 - m2) / (m1 + m2);
    const double eta = m1*m2/mtot/mtot;
    const double m1M = m1/mtot;
    const double m2M = m2/mtot;
    /* Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
     * We write dSigmaL for their (\delta m/m) * \Sigma_\ell
     * There's a division by mtotal^2 in both the energy and flux terms
     * We just absorb the division by mtotal^2 into SL and dSigmaL
     */
    const double SL = m1M*m1M*chi1L + m2M*m2M*chi2L;
    const double dSigmaL = d*(m2M*chi2L - m1M*chi1L);

    const double pfaN = 3.L/(128.L * eta);

    memset(pfa, 0, sizeof(PNPhasingSeries));

    /* Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18 */
    pfa->v[0] = 1.L;
    pfa->v[1] = 0.L;
    pfa->v[2] = 5.L*(743.L/84.L + 11.L * eta)/9.L;
    pfa->v[3] = -16.L*PI;
    pfa->v[4] = 5.L*(3058.673L/7.056L + 5429.L/7.L * eta
                     + 617.L * eta*eta)/72.L;
    pfa->v[5] = 5.L/9.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->vlogv[5] = 5.L/3.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->v[6] = (11583.231236531L/4.694215680L
                     - 640.L/3.L * PI * PI - 6848.L/21.L*GAMMA)
                 + eta * (-15737.765635L/3.048192L
                     + 2255./12. * PI * PI)
                 + eta*eta * 76055.L/1728.L
                 - eta*eta*eta * 127825.L/1296.L;
    pfa->v[6] += (-6848.L/21.L)*log(4.);
    pfa->vlogv[6] = -6848.L/21.L;
    pfa->v[7] = PI * ( 77096675.L/254016.L
                     + 378515.L/1512.L * eta - 74045.L/756.L * eta*eta);

    double qm_def1=1.0;
    double qm_def2=1.0;

    /* Compute 2.0PN SS, QM, and self-spin */
    // See Eq. (6.24) in arXiv:0810.5336
    // 9b,c,d in arXiv:astro-ph/0504538
    double pn_sigma = eta * (721.L/48.L*chi1L*chi2L - 247.L/48.L*chi1dotchi2);
    pn_sigma += (720.L*qm_def1 - 1.L)/96.0L * m1M * m1M * chi1L * chi1L;
    pn_sigma += (720.L*qm_def2 - 1.L)/96.0L * m2M * m2M * chi2L * chi2L;
    pn_sigma -= (240.L*qm_def1 - 7.L)/96.0L * m1M * m1M * chi1sq;
    pn_sigma -= (240.L*qm_def2 - 7.L)/96.0L * m2M * m2M * chi2sq;

    double pn_ss3 =  (326.75L/1.12L + 557.5L/1.8L*eta)*eta*chi1L*chi2L;
    pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m1M-120.L*m1M*m1M)*qm_def1 + (-4108.25L/6.72L-108.5L/1.2L*m1M+125.5L/3.6L*m1M*m1M)) *m1M*m1M * chi1sq;
    pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m2M-120.L*m2M*m2M)*qm_def2 + (-4108.25L/6.72L-108.5L/1.2L*m2M+125.5L/3.6L*m2M*m2M)) *m2M*m2M * chi2sq;

    /* Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16 */
    const double pn_gamma = (554345.L/1134.L + 110.L*eta/9.L)*SL + (13915.L/84.L - 10.L*eta/3.L)*dSigmaL;
    int spinorder = SPIN_ORDER_35PN; // Hardwired for simplicity
    switch( spinorder )
    {
        case SPIN_ORDER_ALL:
        case SPIN_ORDER_35PN:
            pfa->v[7] += (-8980424995.L/762048.L + 6586595.L*eta/756.L - 305.L*eta*eta/36.L)*SL - (170978035.L/48384.L - 2876425.L*eta/672.L - 4735.L*eta*eta/144.L) * dSigmaL;
        case SPIN_ORDER_3PN:
            pfa->v[6] += PI * (3760.L*SL + 1490.L*dSigmaL)/3.L + pn_ss3;
        case SPIN_ORDER_25PN:
            pfa->v[5] += -1.L * pn_gamma;
            pfa->vlogv[5] += -3.L * pn_gamma;
        case SPIN_ORDER_2PN:
            pfa->v[4] += -10.L * pn_sigma;
        case SPIN_ORDER_15PN:
            pfa->v[3] += 188.L*SL/3.L + 25.L*dSigmaL;
        case SPIN_ORDER_1PN:
        case SPIN_ORDER_05PN:
        case SPIN_ORDER_0PN:
            break;
    }

    /* At the very end, multiply everything in the series by pfaN */
    for(int ii = 0; ii <= PN_PHASING_SERIES_MAX_ORDER; ii++)
    {
        pfa->v[ii] *= pfaN;
        pfa->vlogv[ii] *= pfaN;
        pfa->vlogvsq[ii] *= pfaN;
    }
}

///////////////////////////////////////////////////////////////////////////////

/**
 * PN reduced spin parameter
 * See Eq 5.9 in http://arxiv.org/pdf/1107.1267v2.pdf
 */
double chiPN(double eta, double chi1, double chi2) {
  // Convention m1 >= m2 and chi1 is the spin on m1
  double delta = sqrt(1.0 - 4.0*eta);
  double chi_s = (chi1 + chi2) / 2.0;
  double chi_a = (chi1 - chi2) / 2.0;
  return chi_s * (1.0 - eta*76.0/113.0) + delta*chi_a;
}

/**
 * Return the closest higher power of 2
 */
size_t NextPow2(const size_t n)
{
  // use pow here, not bit-wise shift, as the latter seems to run against an upper cutoff long before SIZE_MAX, at least on some platforms
  return (size_t) pow(2,ceil(log2(n)));
}

///**
// * Step function
// */
//static double StepFunc(const double t, const double t1) {
//  if (t < t1)
//    return 0.0;
//  else
//    return 1.0;
//}

/**
 * Step function in boolean version
 */
bool StepFunc_boolean(const double t, const double t1) {
	return (t >= t1);
}

//////////////////////// Final spin, final mass, fring, fdamp ////////////////////////

// Final Spin and Radiated Energy formulas described in 1508.07250

/**
 * Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
 * s defined around Equation 3.6.
 */
double FinalSpin0815_s(double eta, double s) {
  double eta2 = eta*eta;
  double eta3 = eta2*eta;
  double eta4 = eta3*eta;
  double s2 = s*s;
  double s3 = s2*s;
  double s4 = s3*s;

return 3.4641016151377544*eta - 4.399247300629289*eta2 +
   9.397292189321194*eta3 - 13.180949901606242*eta4 +
   (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s +
   (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 +
   (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 +
   (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4;
}

/**
 * Wrapper function for FinalSpin0815_s.
 */
double FinalSpin0815(double eta, double chi1, double chi2) {
  // Convention m1 >= m2
  double Seta = sqrt(1.0 - 4.0*eta);
  double m1 = 0.5 * (1.0 + Seta);
  double m2 = 0.5 * (1.0 - Seta);
  double m1s = m1*m1;
  double m2s = m2*m2;
  // s defined around Equation 3.6 arXiv:1508.07250
  double s = (m1s * chi1 + m2s * chi2);
  return FinalSpin0815_s(eta, s);
}

/**
 * Formula to predict the total radiated energy. Equation 3.7 and 3.8 arXiv:1508.07250
 * Input parameter s defined around Equation 3.7 and 3.8.
 */
double EradRational0815_s(double eta, double s) {
  double eta2 = eta*eta;
  double eta3 = eta2*eta;
  double eta4 = eta3*eta;

  return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 - 1.4756929437702908*eta + 7.304676214885011*eta2)*s);
}

/**
 * Wrapper function for EradRational0815_s.
 */
double EradRational0815(double eta, double chi1, double chi2) {
  // Convention m1 >= m2
  double Seta = sqrt(1.0 - 4.0*eta);
  double m1 = 0.5 * (1.0 + Seta);
  double m2 = 0.5 * (1.0 - Seta);
  double m1s = m1*m1;
  double m2s = m2*m2;
  // arXiv:1508.07250
  double s = (m1s * chi1 + m2s * chi2) / (m1s + m2s);

  return EradRational0815_s(eta, s);
}

/**
 * fring is the real part of the ringdown frequency
 * 1508.07250 figure 9
 */
double fring(double eta, double chi1, double chi2, double finspin) {
  double return_val;

  if (finspin > 1.0) ERROR(PD_EDOM, "PhenomD fring function: final spin > 1.0 not supported\n");

  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_spline *iFring = gsl_spline_alloc(gsl_interp_cspline, QNMData_length);
  gsl_spline_init(iFring, QNMData_a, QNMData_fring, QNMData_length);

  return_val = gsl_spline_eval(iFring, finspin, acc) / (1.0 - EradRational0815(eta, chi1, chi2));

  gsl_spline_free(iFring);
  gsl_interp_accel_free(acc);
  return return_val;
}

/**
 * fdamp is the complex part of the ringdown frequency
 * 1508.07250 figure 9
 */
double fdamp(double eta, double chi1, double chi2, double finspin) {
  double return_val;

  if (finspin > 1.0) ERROR(PD_EDOM, "PhenomD fdamp function: final spin > 1.0 not supported\n");

  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_spline *iFdamp = gsl_spline_alloc(gsl_interp_cspline, QNMData_length);
  gsl_spline_init(iFdamp, QNMData_a, QNMData_fdamp, QNMData_length);

  return_val = gsl_spline_eval(iFdamp, finspin, acc) / (1.0 - EradRational0815(eta, chi1, chi2));

  gsl_spline_free(iFdamp);
  gsl_interp_accel_free(acc);
  return return_val;
}

int init_useful_powers(UsefulPowers *p, double number)
{
	CHECK(0 != p, PD_EFAULT, "p is NULL");
	CHECK(number >= 0, PD_EDOM, "number must be non-negative");

	// consider changing pow(x,1/6.0) to cbrt(x) and sqrt(x) - might be faster
	p->sixth = pow(number, 1/6.0);
	p->third = p->sixth * p->sixth;
	p->two_thirds = number / p->third;
	p->four_thirds = number * (p->third);
	p->five_thirds = p->four_thirds * (p->third);
	p->two = number * number;
	p->seven_thirds = p->third * p->two;
	p->eight_thirds = p->two_thirds * p->two;

	return PD_SUCCESS;
}

/******************************* Amplitude functions *******************************/

/**
 * amplitude scaling factor defined by eq. 17 in 1508.07253
 */
double amp0Func(double eta) {
  return (sqrt(2.0/3.0)*sqrt(eta))/powers_of_pi.sixth;
}

///////////////////////////// Amplitude: Inspiral functions /////////////////////////

// Phenom coefficients rho1, ..., rho3 from direct fit
// AmpInsDFFitCoeffChiPNFunc[eta, chiPN]

/**
 * rho_1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double rho1_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 3931.8979897196696 - 17395.758706812805*eta
  + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2)*xi
  + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi2
  + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi3;
}

/**
 * rho_2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double rho2_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -40105.47653771657 + 112253.0169706701*eta
  + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2)*xi
  + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi2
  + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi3;
}

/**
 * rho_3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double rho3_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 83208.35471266537 - 191237.7264145924*eta +
  (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2)*xi
  + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi2
  + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi3;
}

// The Newtonian term in LAL is fine and we should use exactly the same (either hardcoded or call).
// We just use the Mathematica expression for convenience.
/**
 * Inspiral amplitude plus rho phenom coefficents. rho coefficients computed
 * in rho1_fun, rho2_fun, rho3_fun functions.
 * Amplitude is a re-expansion. See 1508.07253 and Equation 29, 30 and Appendix B arXiv:1508.07253 for details
 */
double AmpInsAnsatz(double Mf, UsefulPowers * powers_of_Mf, AmpInsPrefactors * prefactors) {
  double Mf2 = powers_of_Mf->two;
  double Mf3 = Mf*Mf2;

  return 1 + powers_of_Mf->two_thirds * prefactors->two_thirds
			+ Mf * prefactors->one + powers_of_Mf->four_thirds * prefactors->four_thirds
			+ powers_of_Mf->five_thirds * prefactors->five_thirds + Mf2 * prefactors->two
			+ powers_of_Mf->seven_thirds * prefactors->seven_thirds + powers_of_Mf->eight_thirds * prefactors->eight_thirds
			+ Mf3 * prefactors->three;
}

int init_amp_ins_prefactors(AmpInsPrefactors * prefactors, IMRPhenomDAmplitudeCoefficients* p)
{
	CHECK(0 != p, PD_EFAULT, "p is NULL");
	CHECK(0 != prefactors, PD_EFAULT, "prefactors is NULL");

	double eta = p->eta;

	prefactors->amp0 = amp0Func(p->eta);

	double chi1 = p->chi1;
	double chi2 = p->chi2;
	double rho1 = p->rho1;
	double rho2 = p->rho2;
	double rho3 = p->rho3;

	double chi12 = chi1*chi1;
	double chi22 = chi2*chi2;
	double eta2 = eta*eta;
	double eta3 = eta*eta2;


	double Pi = PI;
	double Pi2 = powers_of_pi.two;
	double Seta = sqrt(1.0 - 4.0*eta);

	prefactors->two_thirds = ((-969 + 1804*eta)*powers_of_pi.two_thirds)/672.;
	prefactors->one = ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*Pi)/48.;
	prefactors->four_thirds = (	(-27312085.0 - 10287648*chi22 - 10287648*chi12*(1 + Seta) + 10287648*chi22*Seta
								 + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta
								 + 35371056*eta2
								 )
							* powers_of_pi.four_thirds) / 8.128512e6;
	prefactors->five_thirds = (powers_of_pi.five_thirds * (chi2*(-285197*(-1 + Seta) + 4*(-91902 + 1579*Seta)*eta - 35632*eta2)
															+ chi1*(285197*(1 + Seta) - 4*(91902 + 1579*Seta)*eta - 35632*eta2)
															+ 42840*(-1.0 + 4*eta)*Pi
															)
								) / 32256.;
	prefactors->two = - (Pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2
							  - 324322727232*eta3
							  - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*(1 + Seta)
									- 107414046432*chi22*Seta + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*Pi
									)
							  + 12*eta*(-545384828789 - 176491177632*chi1*chi2 + 202603761360*chi22
										+ 77616*chi12*(2610335 + 995766*Seta) - 77287373856*chi22*Seta
										+ 5841690624*(chi1 + chi2)*Pi + 21384760320*Pi2
										)
								)
						)/6.0085960704e10;
	prefactors->seven_thirds= rho1;
	prefactors->eight_thirds = rho2;
	prefactors->three = rho3;

	return PD_SUCCESS;
}

/**
 * Take the AmpInsAnsatz expression and compute the first derivative
 * with respect to frequency to get the expression below.
 */
double DAmpInsAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double eta = p->eta;
  double chi1 = p->chi1;
  double chi2 = p->chi2;
  double rho1 = p->rho1;
  double rho2 = p->rho2;
  double rho3 = p->rho3;

  double chi12 = chi1*chi1;
  double chi22 = chi2*chi2;
  double eta2 = eta*eta;
  double eta3 = eta*eta2;
  double Mf2 = Mf*Mf;
  double Pi = PI;
  double Pi2 = powers_of_pi.two;
  double Seta = sqrt(1.0 - 4.0*eta);

   return ((-969 + 1804*eta)*pow(Pi,2.0/3.0))/(1008.*pow(Mf,1.0/3.0))
   + ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*Pi)/48.
   + ((-27312085 - 10287648*chi22 - 10287648*chi12*(1 + Seta)
   + 10287648*chi22*Seta + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta
   + 35371056*eta2)*pow(Mf,1.0/3.0)*pow(Pi,4.0/3.0))/6.096384e6
   + (5*pow(Mf,2.0/3.0)*pow(Pi,5.0/3.0)*(chi2*(-285197*(-1 + Seta)
   + 4*(-91902 + 1579*Seta)*eta - 35632*eta2) + chi1*(285197*(1 + Seta)
   - 4*(91902 + 1579*Seta)*eta - 35632*eta2) + 42840*(-1 + 4*eta)*Pi))/96768.
   - (Mf*Pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2 - 324322727232*eta3
   - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*(1 + Seta) - 107414046432*chi22*Seta
   + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*Pi)
   + 12*eta*(-545384828789.0 - 176491177632*chi1*chi2 + 202603761360*chi22 + 77616*chi12*(2610335 + 995766*Seta)
   - 77287373856*chi22*Seta + 5841690624*(chi1 + chi2)*Pi + 21384760320*Pi2)))/3.0042980352e10
   + (7.0/3.0)*pow(Mf,4.0/3.0)*rho1 + (8.0/3.0)*pow(Mf,5.0/3.0)*rho2 + 3*Mf2*rho3;
}

/////////////////////////// Amplitude: Merger-Ringdown functions ///////////////////////

// Phenom coefficients gamma1, ..., gamma3
// AmpMRDAnsatzFunc[]

/**
 * gamma 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double gamma1_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.006927402739328343 + 0.03020474290328911*eta
  + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2)*xi
  + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi2
  + (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi3;
}

/**
 * gamma 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double gamma2_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 1.010344404799477 + 0.0008993122007234548*eta
  + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2)*xi
  + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi2
  + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi3;
}

/**
 * gamma 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double gamma3_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 1.3081615607036106 - 0.005537729694807678*eta
  + (-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2)*xi
  + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi2
  + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi3;
}

/**
 * Ansatz for the merger-ringdown amplitude. Equation 19 arXiv:1508.07253
 */
double AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;
  double fDMgamma3 = fDM*gamma3;
  double fminfRD = f - fRD;
  return exp( -(fminfRD)*gamma2 / (fDMgamma3) )
    * (fDMgamma3*gamma1) / (pow_2_of(fminfRD) + pow_2_of(fDMgamma3));
}

/**
 * first frequency derivative of AmpMRDAnsatz
 */
double DAmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;

  double fDMgamma3 = fDM * gamma3;
  double pow2_fDMgamma3 = pow_2_of(fDMgamma3);
  double fminfRD = f - fRD;
  double expfactor = exp(((fminfRD)*gamma2)/(fDMgamma3));
  double pow2pluspow2 = pow_2_of(fminfRD) + pow2_fDMgamma3;

   return (-2*fDM*(fminfRD)*gamma3*gamma1) / ( expfactor * pow_2_of(pow2pluspow2)) -
     (gamma2*gamma1) / ( expfactor * (pow2pluspow2)) ;
}

/**
 * Equation 20 arXiv:1508.07253 (called f_peak in paper)
 * analytic location of maximum of AmpMRDAnsatz
 */
double fmaxCalc(IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;

  // NOTE: There's a problem with this expression from the paper becoming imaginary if gamma2>=1
  // Fix: if gamma2 >= 1 then set the square root term to zero.
  if (gamma2 <= 1)
    return fabs(fRD + (fDM*(-1 + sqrt(1 - pow_2_of(gamma2)))*gamma3)/gamma2);
  else
    return fabs(fRD + (fDM*(-1)*gamma3)/gamma2);
}

///////////////////////////// Amplitude: Intermediate functions ////////////////////////

// Phenom coefficients delta0, ..., delta4 determined from collocation method
// (constraining 3 values and 2 derivatives)
// AmpIntAnsatzFunc[]

/**
 * Ansatz for the intermediate amplitude. Equation 21 arXiv:1508.07253
 */
double AmpIntAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double Mf2 = Mf*Mf;
  double Mf3 = Mf*Mf2;
  double Mf4 = Mf*Mf3;
  return p->delta0 + p->delta1*Mf + p->delta2*Mf2 + p->delta3*Mf3 + p->delta4*Mf4;
}

/**
 * The function name stands for 'Amplitude Intermediate Collocation Fit Coefficient'
 * This is the 'v2' value in Table 5 of arXiv:1508.07253
 */
double AmpIntColFitCoeff(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.8149838730507785 + 2.5747553517454658*eta
  + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2)*xi
  + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi2
  + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi3;
}

  /**
  * The following functions (delta{0,1,2,3,4}_fun) were derived
  * in mathematica according to
  * the constraints detailed in arXiv:1508.07253,
  * section 'Region IIa - intermediate'.
  * These are not given in the paper.
  * Can be rederived by solving Equation 21 for the constraints
  * given in Equations 22-26 in arXiv:1508.07253
  */
double delta0_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f22 = d->f22;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((d2*f15*f22*f3 - 2*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32
  - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33
  + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33
  - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35
  + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1
  + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2
  + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3
  + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3
  + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(f3-f2)));
}

double delta1_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f22 = d->f22;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((-(d2*f15*f22) + 2*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2*d1*f13*f23*f3
  + 2*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32
  - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32
  - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33
  + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35
  - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1
  + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3
  + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

double delta2_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2*d2*f12*f24 - d2*f15*f3
  + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32
  + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32
  - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33
  + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1
  - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2
  - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3
  + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

double delta3_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f22 = d->f22;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;

  return -((-2*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3
  - 2*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3
  + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32
  + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33
  - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1
  - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2
  - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3
  - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

double delta4_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f22 = d->f22;
  double f23 = d->f23;
  double f32 = d->f32;
  double f33 = d->f33;

  return -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2*d1*f12*f2*f3
  + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32
  - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1
  - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2
  + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

/**
 * Calculates delta_i's
 * Method described in arXiv:1508.07253 section 'Region IIa - intermediate'
 */
void ComputeDeltasFromCollocation(IMRPhenomDAmplitudeCoefficients* p) {
  // Three evenly spaced collocation points in the interval [f1,f3].
  double f1 = AMP_fJoin_INS;
  double f3 = p->fmaxCalc;
  double dfx = (f3 - f1)/2.0;
  double f2 = f1 + dfx;

  UsefulPowers powers_of_f1;
  int status = init_useful_powers(&powers_of_f1, f1);
  CHECK( status == PD_SUCCESS, PD_EFUNC, "Failed to initialize useful powers of f1.");

  AmpInsPrefactors prefactors;
  status = init_amp_ins_prefactors(&prefactors, p);
  CHECK( status == PD_SUCCESS, PD_EFUNC, "Failed to initialize amplitude prefactors for inspiral range.");


  // v1 is inspiral model evaluated at f1
  // d1 is derivative of inspiral model evaluated at f1
  double v1 = AmpInsAnsatz(f1, &powers_of_f1, &prefactors);
  double d1 = DAmpInsAnsatz(f1, p);

  // v3 is merger-ringdown model evaluated at f3
  // d2 is derivative of merger-ringdown model evaluated at f3
  double v3 = AmpMRDAnsatz(f3, p);
  double d2 = DAmpMRDAnsatz(f3, p);

  // v2 is the value of the amplitude evaluated at f2
  // they come from the fit of the collocation points in the intermediate region
  double v2 = AmpIntColFitCoeff(p->eta, p->chi);

  p->f1 = f1;
  p->f2 = f2;
  p->f3 = f3;
  p->v1 = v1;
  p->v2 = v2;
  p->v3 = v3;
  p->d1 = d1;
  p->d2 = d2;

  // Now compute the delta_i's from the collocation coefficients
  // Precompute common quantities here and pass along to delta functions.
  DeltaUtility d;
  d.f12 = f1*f1;
  d.f13 = f1*d.f12;
  d.f14 = f1*d.f13;
  d.f15 = f1*d.f14;
  d.f22 = f2*f2;
  d.f23 = f2*d.f22;
  d.f24 = f2*d.f23;
  d.f32 = f3*f3;
  d.f33 = f3*d.f32;
  d.f34 = f3*d.f33;
  d.f35 = f3*d.f34;
  p->delta0 = delta0_fun(p, &d);
  p->delta1 = delta1_fun(p, &d);
  p->delta2 = delta2_fun(p, &d);
  p->delta3 = delta3_fun(p, &d);
  p->delta4 = delta4_fun(p, &d);
}


///////////////////////////// Amplitude: glueing function ////////////////////////////

/**
 * A struct containing all the parameters that need to be calculated
 * to compute the phenomenological amplitude
 */
IMRPhenomDAmplitudeCoefficients* ComputeIMRPhenomDAmplitudeCoefficients(double eta, double chi1, double chi2, double finspin) {
  IMRPhenomDAmplitudeCoefficients *p = (IMRPhenomDAmplitudeCoefficients *) malloc(sizeof(IMRPhenomDAmplitudeCoefficients));

  p->eta = eta;
  p->chi1 = chi1;
  p->chi2 = chi2;

  p->q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta);
  p->chi = chiPN(eta, chi1, chi2);

  p->fRD = fring(eta, chi1, chi2, finspin);
  p->fDM = fdamp(eta, chi1, chi2, finspin);

  // Compute gamma_i's, rho_i's first then delta_i's
  p->gamma1 = gamma1_fun(eta, p->chi);
  p->gamma2 = gamma2_fun(eta, p->chi);
  p->gamma3 = gamma3_fun(eta, p->chi);

  p->fmaxCalc = fmaxCalc(p);

  p->rho1 = rho1_fun(eta, p->chi);
  p->rho2 = rho2_fun(eta, p->chi);
  p->rho3 = rho3_fun(eta, p->chi);

  // compute delta_i's
  ComputeDeltasFromCollocation(p);

  return p;
}

// Call ComputeIMRPhenomDAmplitudeCoefficients() first!
/**
 * This function computes the IMR amplitude given phenom coefficients.
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 */
double IMRPhenDAmplitude(double f, IMRPhenomDAmplitudeCoefficients *p, UsefulPowers *powers_of_f, AmpInsPrefactors * prefactors) {
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  // The inspiral, intermediate and merger-ringdown amplitude parts

  // Transition frequencies
  p->fInsJoin = AMP_fJoin_INS;
  p->fMRDJoin = p->fmaxCalc;

  double f_seven_sixths = f * powers_of_f->sixth;
  double AmpPreFac = prefactors->amp0 / f_seven_sixths;

  // split the calculation to just 1 of 3 possible mutually exclusive ranges

  if (!StepFunc_boolean(f, p->fInsJoin))	// Inspiral range
  {
	  double AmpIns = AmpPreFac * AmpInsAnsatz(f, powers_of_f, prefactors);
	  return AmpIns;
  }

  if (StepFunc_boolean(f, p->fMRDJoin))	// MRD range
  {
	  double AmpMRD = AmpPreFac * AmpMRDAnsatz(f, p);
	  return AmpMRD;
  }

  //	Intermediate range
  double AmpInt = AmpPreFac * AmpIntAnsatz(f, p);
  return AmpInt;
}

/********************************* Phase functions *********************************/

////////////////////////////// Phase: Ringdown functions ///////////////////////////

// alpha_i i=1,2,3,4,5 are the phenomenological intermediate coefficients depending on eta and chiPN
// PhiRingdownAnsatz is the ringdown phasing in terms of the alpha_i coefficients

/**
 * alpha 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double alpha1Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 43.31514709695348 + 638.6332679188081*eta
    + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2)*xi
    + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi2
    + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi3;
}

/**
 * alpha 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double alpha2Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -0.07020209449091723 - 0.16269798450687084*eta
  + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2)*xi
  + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi2
  + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi3;
}

/**
 * alpha 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double alpha3Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 9.5988072383479 - 397.05438595557433*eta
  + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2)*xi
  + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi2
  + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi3;
}

/**
 * alpha 4 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double alpha4Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -0.02989487384493607 + 1.4022106448583738*eta
  + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2)*xi
  + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi2
  + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi3;
}

/**
 * alpha 5 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double alpha5Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.9974408278363099 - 0.007884449714907203*eta
  + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2)*xi
  + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi2
  + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi3;
}

/**
 * Ansatz for the merger-ringdown phase Equation 14 arXiv:1508.07253
 */
double PhiMRDAnsatzInt(double f, IMRPhenomDPhaseCoefficients *p)
{
  double sqrootf = sqrt(f);
  double fpow1_5 = f * sqrootf;
  // check if this is any faster: 2 sqrts instead of one pow(x,0.75)
  double fpow0_75 = sqrt(fpow1_5); // pow(f,0.75);

  return -(p->alpha2/f)
		 + (4.0/3.0) * (p->alpha3 * fpow0_75)
		 + p->alpha1 * f
		 + p->alpha4 * atan((f - p->alpha5 * p->fRD) / p->fDM);
}

/**
 * First frequency derivative of PhiMRDAnsatzInt
 */
double DPhiMRD(double f, IMRPhenomDPhaseCoefficients *p) {
  return (p->alpha1 + p->alpha2/pow_2_of(f) + p->alpha3/pow(f,0.25) + p->alpha4/(p->fDM*(1 + pow_2_of(f - p->alpha5 * p->fRD)/pow_2_of(p->fDM)))) / p->eta;
}

///////////////////////////// Phase: Intermediate functions /////////////////////////////

// beta_i i=1,2,3 are the phenomenological intermediate coefficients depending on eta and chiPN
// PhiIntAnsatz is the intermediate phasing in terms of the beta_i coefficients


// \[Beta]1Fit = PhiIntFitCoeff\[Chi]PNFunc[\[Eta], \[Chi]PN][[1]]

/**
 * beta 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double beta1Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 97.89747327985583 - 42.659730877489224*eta
  + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2)*xi
  + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi2
  + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi3;
}

/**
 * beta 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double beta2Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -3.282701958759534 - 9.051384468245866*eta
  + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2)*xi
  + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi2
  + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi3;
}

/**
 * beta 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double beta3Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -0.000025156429818799565 + 0.000019750256942201327*eta
  + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2)*xi
  + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi2
  + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi3;
}

/**
 * ansatz for the intermediate phase defined by Equation 16 arXiv:1508.07253
 */
double PhiIntAnsatz(double Mf, IMRPhenomDPhaseCoefficients *p) {
  // 1./eta in paper omitted and put in when need in the functions:
  // ComputeIMRPhenDPhaseConnectionCoefficients
  // IMRPhenDPhase
  return  p->beta1*Mf - p->beta3/(3.*pow_3_of(Mf)) + p->beta2*log(Mf);
}

/**
 * First frequency derivative of PhiIntAnsatz
 * (this time with 1./eta explicitly factored in)
 */
double DPhiIntAnsatz(double Mf, IMRPhenomDPhaseCoefficients *p) {
  return (p->beta1 + p->beta3/pow_4_of(Mf) + p->beta2/Mf) / p->eta;
}

/**
 * temporary instance of DPhiIntAnsatz used when computing
 * coefficients to make the phase C(1) continuous between regions.
 */
double DPhiIntTemp(double ff, IMRPhenomDPhaseCoefficients *p) {
  double eta = p->eta;
  double beta1 = p->beta1;
  double beta2 = p->beta2;
  double beta3 = p->beta3;
  double C2Int = p->C2Int;

  return C2Int + (beta1 + beta3/pow_4_of(ff) + beta2/ff)/eta;
}

///////////////////////////// Phase: Inspiral functions /////////////////////////////

// sigma_i i=1,2,3,4 are the phenomenological inspiral coefficients depending on eta and chiPN
// PhiInsAnsatzInt is a souped up TF2 phasing which depends on the sigma_i coefficients

/**
 * sigma 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double sigma1Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 2096.551999295543 + 1463.7493168261553*eta
  + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2)*xi
  + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi2
  + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi3;
}

/**
 * sigma 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double sigma2Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -10114.056472621156 - 44631.01109458185*eta
  + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2)*xi
  + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi2
  + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi3;
}

/**
 * sigma 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double sigma3Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 22933.658273436497 + 230960.00814979506*eta
  + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2)*xi
  + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi2
  + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi3;
}

/**
 * sigma 4 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
double sigma4Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -14621.71522218357 - 377812.8579387104*eta
  + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2)*xi
  + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi2
  + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi3;
}

/**
 * Ansatz for the inspiral phase.
 * We call the LAL TF2 coefficients here.
 * The exact values of the coefficients used are given
 * as comments in the top of this file
 * Defined by Equation 27 and 28 arXiv:1508.07253
 */
double PhiInsAnsatzInt(double Mf, UsefulPowers *powers_of_Mf, PhiInsPrefactors *prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn)
{
	CHECK(0 != pn, PD_EFAULT, "pn is NULL");

  // Assemble PN phasing series
  const double v = powers_of_Mf->third * powers_of_pi.third;
  const double logv = log(v);

  double phasing = prefactors->initial_phasing;

  phasing += prefactors->two_thirds	* powers_of_Mf->two_thirds;
  phasing += prefactors->third * powers_of_Mf->third;
  phasing += prefactors->third_with_logv * logv * powers_of_Mf->third;
  phasing += prefactors->logv * logv;
  phasing += prefactors->minus_third / powers_of_Mf->third;
  phasing += prefactors->minus_two_thirds / powers_of_Mf->two_thirds;
  phasing += prefactors->minus_one / Mf;
  phasing += prefactors->minus_five_thirds / powers_of_Mf->five_thirds; // * v^0

  // Now add higher order terms that were calibrated for PhenomD
  phasing += ( prefactors->one * Mf + prefactors->four_thirds * powers_of_Mf->four_thirds
			   + prefactors->five_thirds * powers_of_Mf->five_thirds
			   + prefactors->two * powers_of_Mf->two
			 ) / p->eta;

  return phasing;
}

int init_phi_ins_prefactors(PhiInsPrefactors * prefactors, IMRPhenomDPhaseCoefficients* p, PNPhasingSeries *pn)
{
	CHECK(0 != p, PD_EFAULT, "p is NULL");
	CHECK(0 != prefactors, PD_EFAULT, "prefactors is NULL");

	double sigma1 = p->sigma1;
	double sigma2 = p->sigma2;
	double sigma3 = p->sigma3;
	double sigma4 = p->sigma4;
	double Pi = PI;

  // PN phasing series
	prefactors->initial_phasing = pn->v[5] - PI_4;
	prefactors->two_thirds = pn->v[7] * powers_of_pi.two_thirds;
	prefactors->third = pn->v[6] * powers_of_pi.third;
	prefactors->third_with_logv = pn->vlogv[6] * powers_of_pi.third;
	prefactors->logv = pn->vlogv[5];
	prefactors->minus_third = pn->v[4] / powers_of_pi.third;
	prefactors->minus_two_thirds = pn->v[3] / powers_of_pi.two_thirds;
	prefactors->minus_one = pn->v[2] / Pi;
	prefactors->minus_five_thirds = pn->v[0] / powers_of_pi.five_thirds; // * v^0

  // higher order terms that were calibrated for PhenomD
	prefactors->one = sigma1;
	prefactors->four_thirds = sigma2 * 3.0/4.0;
	prefactors->five_thirds = sigma3 * 3.0/5.0;
	prefactors->two = sigma4 / 2.0;

	return PD_SUCCESS;
}

/**
 * First frequency derivative of PhiInsAnsatzInt
 */
double DPhiInsAnsatzInt(double Mf, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn) {
  double sigma1 = p->sigma1;
  double sigma2 = p->sigma2;
  double sigma3 = p->sigma3;
  double sigma4 = p->sigma4;
  double Pi = PI;

  // Assemble PN phasing series
  const double v = cbrt(Pi*Mf);
  const double logv = log(v);
  const double v2 = v * v;
  const double v3 = v * v2;
  const double v4 = v * v3;
  const double v5 = v * v4;
  const double v6 = v * v5;
  const double v7 = v * v6;
  const double v8 = v * v7;

  // Apply the correct prefactors to LAL phase coefficients to get the
  // phase derivative dphi / dMf = dphi/dv * dv/dMf
  double Dphasing = 0.0;
  Dphasing += +2.0 * pn->v[7] * v7;
  Dphasing += (pn->v[6] + pn->vlogv[6] * (1.0 + logv)) * v6;
  Dphasing += pn->vlogv[5] * v5;
  Dphasing += -1.0 * pn->v[4] * v4;
  Dphasing += -2.0 * pn->v[3] * v3;
  Dphasing += -3.0 * pn->v[2] * v2;
  Dphasing += -4.0 * pn->v[1] * v;
  Dphasing += -5.0 * pn->v[0];
  Dphasing /= v8 * 3.0/Pi;

  // Now add higher order terms that were calibrated for PhenomD
  Dphasing += (
          sigma1
        + sigma2 * v / powers_of_pi.third
        + sigma3 * v2 / powers_of_pi.two_thirds
        + (sigma4/Pi) * v3
        ) / p->eta;

  return Dphasing;
}

///////////////////////////////// Phase: glueing function ////////////////////////////////

/**
 * A struct containing all the parameters that need to be calculated
 * to compute the phenomenological phase
 */
IMRPhenomDPhaseCoefficients* ComputeIMRPhenomDPhaseCoefficients(double eta, double chi1, double chi2, double finspin) {

  IMRPhenomDPhaseCoefficients *p = (IMRPhenomDPhaseCoefficients *) malloc(sizeof(IMRPhenomDPhaseCoefficients));

  // Convention m1 >= m2
  p->eta = eta;
  p->chi1 = chi1;
  p->chi2 = chi2;

  p->q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta);
  p->chi = chiPN(eta, chi1, chi2);

  p->sigma1 = sigma1Fit(eta, p->chi);
  p->sigma2 = sigma2Fit(eta, p->chi);
  p->sigma3 = sigma3Fit(eta, p->chi);
  p->sigma4 = sigma4Fit(eta, p->chi);

  p->beta1 = beta1Fit(eta, p->chi);
  p->beta2 = beta2Fit(eta, p->chi);
  p->beta3 = beta3Fit(eta, p->chi);

  p->alpha1 = alpha1Fit(eta, p->chi);
  p->alpha2 = alpha2Fit(eta, p->chi);
  p->alpha3 = alpha3Fit(eta, p->chi);
  p->alpha4 = alpha4Fit(eta, p->chi);
  p->alpha5 = alpha5Fit(eta, p->chi);

  p->fRD = fring(eta, chi1, chi2, finspin);
  p->fDM = fdamp(eta, chi1, chi2, finspin);

  return p;
}

/**
 * This function aligns the three phase parts (inspiral, intermediate and merger-rindown)
 * such that they are c^1 continuous at the transition frequencies
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 */
void ComputeIMRPhenDPhaseConnectionCoefficients(IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, PhiInsPrefactors * prefactors)
{
  double eta = p->eta;

  // Transition frequencies
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  p->fInsJoin=PHI_fJoin_INS;
  p->fMRDJoin=0.5*p->fRD;

  // Compute C1Int and C2Int coeffs
  // Equations to solve for to get C(1) continuous join
  // PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
  // Joining at fInsJoin
  // PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
  // PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
  double DPhiIns = DPhiInsAnsatzInt(p->fInsJoin, p, pn);
  double DPhiInt = DPhiIntAnsatz(p->fInsJoin, p);
  p->C2Int = DPhiIns - DPhiInt;

  UsefulPowers powers_of_fInsJoin;
  init_useful_powers(&powers_of_fInsJoin, p->fInsJoin);
  p->C1Int = PhiInsAnsatzInt(p->fInsJoin, &powers_of_fInsJoin, prefactors, p, pn)
    - 1.0/eta * PhiIntAnsatz(p->fInsJoin, p) - p->C2Int * p->fInsJoin;

  // Compute C1MRD and C2MRD coeffs
  // Equations to solve for to get C(1) continuous join
  // PhiInsInt (f)  =   PhiMRD (f) + C1MRD + C2MRD f
  // Joining at fMRDJoin
  // Where \[Phi]InsInt(f) is the \[Phi]Ins+\[Phi]Int joined function
  // PhiInsInt (fMRDJoin)  =   PhiMRD (fMRDJoin) + C1MRD + C2MRD fMRDJoin
  // PhiInsInt'(fMRDJoin)  =   PhiMRD'(fMRDJoin) + C2MRD
  // temporary Intermediate Phase function to Join up the Merger-Ringdown
  double PhiIntTempVal = 1.0/eta * PhiIntAnsatz(p->fMRDJoin, p) + p->C1Int + p->C2Int*p->fMRDJoin;
  double DPhiIntTempVal = DPhiIntTemp(p->fMRDJoin, p);
  double DPhiMRDVal = DPhiMRD(p->fMRDJoin, p);
  p->C2MRD = DPhiIntTempVal - DPhiMRDVal;
  p->C1MRD = PhiIntTempVal - 1.0/eta * PhiMRDAnsatzInt(p->fMRDJoin, p) - p->C2MRD*p->fMRDJoin;
}

/**
 * This function computes the IMR phase given phenom coefficients.
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 */
double IMRPhenDPhase(double f, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, UsefulPowers *powers_of_f, PhiInsPrefactors * prefactors)
{
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  // The inspiral, intermendiate and merger-ringdown phase parts

  // split the calculation to just 1 of 3 possible mutually exclusive ranges

  if (!StepFunc_boolean(f, p->fInsJoin))	// Inspiral range
  {
	  double PhiIns = PhiInsAnsatzInt(f, powers_of_f, prefactors, p, pn);
	  return PhiIns;
  }

  if (StepFunc_boolean(f, p->fMRDJoin))	// MRD range
  {
	  double PhiMRD = 1.0/p->eta * PhiMRDAnsatzInt(f, p) + p->C1MRD + p->C2MRD * f;
	  return PhiMRD;
  }

  //	Intermediate range
  double PhiInt = 1.0/p->eta * PhiIntAnsatz(f, p) + p->C1Int + p->C2Int * f;
  return PhiInt;
}

/**
 * Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
 * (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
 * was not available when PhenomD was tuned.
 */
double Subtract3PNSS(double m1, double m2, double M, double chi1, double chi2) {
  double eta = m1 * m2 / (M * M);
  double chi1sq = chi1 * chi1;
  double chi2sq = chi2 * chi2;
  double m1M = m1 / M;
  double m2M = m2 / M;
  double pn_ss3 =  (326.75L/1.12L + 557.5L/1.8L*eta)*eta*chi1*chi2;
  pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m1M-120.L*m1M*m1M) + (-4108.25L/6.72L-108.5L/1.2L*m1M+125.5L/3.6L*m1M*m1M)) *m1M*m1M * chi1sq;
  pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m2M-120.L*m2M*m2M) + (-4108.25L/6.72L-108.5L/1.2L*m2M+125.5L/3.6L*m2M*m2M)) *m2M*m2M * chi2sq;
  return pn_ss3;
}
