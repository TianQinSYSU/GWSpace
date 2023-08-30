/*
 *  Copyright (C) 2017 Sebastian Khan, Francesco Pannarale, Lionel London
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

// #include <lal/LALSimIMR.h>
// #include <lal/SphericalHarmonics.h>
// #include <lal/Date.h>
// #include <lal/Units.h>
//
// #include "LALSimIMRPhenomHM.h"
// #include "LALSimIMRPhenomInternalUtils.h"
// #include "LALSimIMRPhenomUtils.h"
// #include "LALSimRingdownCW.h"
// #include "LALSimIMRPhenomD_internals.c"

#include "IMRPhenomHM.h"
//#include "IMRPhenomD_internals.c"
#include "IMRPhenomD.c"


/*
 * Phase shift due to leading order complex amplitude
 * [L.Blancet, arXiv:1310.1528 (Sec. 9.5)]
 * "Spherical hrmonic modes for numerical relativity"
 */
/* List of phase shifts: the index is the azimuthal number m */
static const double cShift[7] = {0.0,
                                 PI_2  /* i shift */,
                                 0.0,
                                 -PI_2  /* -i shift */,
                                 PI  /* 1 shift */,
                                 PI_2  /* -1 shift */,
                                 0.0};

/**
 * read in a LALDict.
 * If ModeArray in LALDict is NULL then create a ModrArray
 * with the default modes in PhenomHM.
 * If ModeArray is not NULL then use the modes supplied by user.
 */
// LALDict *IMRPhenomHM_setup_mode_array(
//     LALDict *extraParams)
// {
//
//     /* setup ModeArray */
//     if (extraParams == NULL)
//         extraParams = XLALCreateDict();
//     LALValue *ModeArray = XLALSimInspiralWaveformParamsLookupModeArray(extraParams);
//     if (ModeArray == NULL)
//     { /* Default behaviour */
//         PRINT_INFO("Using default modes for PhenomHM.\n");
//         ModeArray = XLALSimInspiralCreateModeArray();
//         /* Only need to define the positive m modes/
//          * The negative m modes are automatically added.
//          */
//         XLALSimInspiralModeArrayActivateMode(ModeArray, 2, 2);
//         XLALSimInspiralModeArrayActivateMode(ModeArray, 2, 1);
//         XLALSimInspiralModeArrayActivateMode(ModeArray, 3, 3);
//         XLALSimInspiralModeArrayActivateMode(ModeArray, 3, 2);
//         XLALSimInspiralModeArrayActivateMode(ModeArray, 4, 4);
//         XLALSimInspiralModeArrayActivateMode(ModeArray, 4, 3);
//         // XLALSimInspiralModeArrayPrintModes(ModeArray);
//         /* Don't forget to insert ModeArray back into extraParams. */
//         XLALSimInspiralWaveformParamsInsertModeArray(extraParams, ModeArray);
//     }
//     else
//     {
//         PRINT_INFO("Using custom modes for PhenomHM.\n");
//     }
//
//     XLALDestroyValue(ModeArray);
//
//     return extraParams;
// }

/**
 *
 */
int PhenomHM_init_useful_mf_powers(PhenomHMUsefulMfPowers *p, double number)
{
    CHECK(0 != p, ERROR_EFAULT, "p is NULL");
    CHECK(!(number < 0), ERROR_EDOM, "number must be non-negative");

    // consider changing pow(x,1/6.0) to cbrt(x) and sqrt(x) - might be faster
    p->itself = number;
    p->sqrt = sqrt(number);
    p->sixth = cbrt(p->sqrt);
    p->m_sixth = 1.0 / p->sixth;
    p->third = p->sixth * p->sixth;
    p->two_thirds = p->third * p->third;
    p->four_thirds = number * p->third;
    p->five_thirds = number * p->two_thirds;
    p->two = number * number;
    p->seven_thirds = p->third * p->two;
    p->eight_thirds = p->two_thirds * p->two;
    p->m_seven_sixths = p->m_sixth / number;
    p->m_five_sixths = p->m_seven_sixths * p->third;
    p->m_sqrt = 1. / p->sqrt;

    return SUCCESS;
}

int PhenomHM_init_useful_powers(PhenomHMUsefulPowers *p, double number)
{
    CHECK(0 != p, ERROR_EFAULT, "p is NULL");
    CHECK(number >= 0, ERROR_EDOM, "number must be non-negative");

    // consider changing pow(x,1/6.0) to cbrt(x) and sqrt(x) - might be faster
    double sixth = pow(number, 1.0 / 6.0);
    p->third = sixth * sixth;
    p->two_thirds = p->third * p->third;
    p->four_thirds = number * p->third;
    p->five_thirds = p->four_thirds * p->third;
    p->two = number * number;
    p->seven_thirds = p->third * p->two;
    p->eight_thirds = p->two_thirds * p->two;
    p->inv = 1. / number;
    double m_sixth = 1.0 / sixth;
    p->m_seven_sixths = p->inv * m_sixth;
    p->m_third = m_sixth * m_sixth;
    p->m_two_thirds = p->m_third * p->m_third;
    p->m_five_thirds = p->inv * p->m_two_thirds;

    return SUCCESS;
}

// /**
//  * helper function to multiple hlm with Ylm.
//  * Adapted from LALSimIMREOBNRv2HMROMUtilities.c
//  */
// int IMRPhenomHMFDAddMode(
//     COMPLEX16FrequencySeries *hptilde,
//     COMPLEX16FrequencySeries *hctilde,
//     COMPLEX16FrequencySeries *hlmtilde,
//     double theta,
//     double phi,
//     INT4 l,
//     INT4 m,
//     INT4 sym)
// {
//     COMPLEX16 Y;
//     int j;
//     COMPLEX16 hlm; /* helper variable that contain a single point of hlmtilde */
//
//     INT4 minus1l; /* (-1)^l */
//     if (l % 2)
//         minus1l = -1;
//     else
//         minus1l = 1;
//     if (sym)
//     { /* Equatorial symmetry: add in -m mode */
//         Y = XLALSpinWeightedSphericalHarmonic(theta, phi, -2, l, m);
//         COMPLEX16 Ymstar = conj(XLALSpinWeightedSphericalHarmonic(theta, phi, -2, l, -m));
//         COMPLEX16 factorp = 0.5 * (Y + minus1l * Ymstar);
//         COMPLEX16 factorc = I * 0.5 * (Y - minus1l * Ymstar);
//         for (j = 0; j < hlmtilde->data->length; ++j)
//         {
//             hlm = (hlmtilde->data->data[j]);
//             hptilde->data->data[j] += factorp * hlm;
//             hctilde->data->data[j] += factorc * hlm;
//         }
//     }
//     else
//     { /* not adding in the -m mode */
//         Y = XLALSpinWeightedSphericalHarmonic(theta, phi, -2, l, m);
//         COMPLEX16 factorp = 0.5 * Y;
//         COMPLEX16 factorc = I * factorp;
//         for (j = 0; j < hlmtilde->data->length; ++j)
//         {
//             hlm = (hlmtilde->data->data[j]);
//             hptilde->data->data[j] += factorp * hlm;
//             hctilde->data->data[j] += factorc * hlm;
//         }
//     }
//
//     return SUCCESS;
// }

/**
 * returns the real and imag parts of the complex ringdown frequency
 * for the (l,m) mode.
 */
int IMRPhenomHMGetRingdownFrequency(
    double *fringdown,
    double *fdamp,
    int ell,
    int mm,
    double finalmass,
    double finalspin)
{
    const double inv2Pi = 0.5 / PI ;
    complex double ZZ;
    ZZ = RingdownCW_CW07102016(RingdownCW_KAPPA(finalspin, ell, mm), ell, mm, 0);
    const double Mf_RD_tmp = inv2Pi * creal(ZZ); /* GW ringdown frequency, converted from angular frequency */
    *fringdown = Mf_RD_tmp / finalmass;         /* scale by predicted final mass */
    /* lm mode ringdown damping time (imaginary part of ringdown), geometric units */
    const double f_DAMP_tmp = inv2Pi * cimag(ZZ); /* this is the 1./tau in the complex QNM */
    *fdamp = f_DAMP_tmp / finalmass;             /* scale by predicted final mass */

    return SUCCESS;
}

/**
 * helper function to easily check if the
 * input frequency sequence is uniformly space
 * or a user defined set of discrete frequencies.
 */
// int IMRPhenomHM_is_freq_uniform(
//     doubleSequence *freqs,
//     double deltaF)
// {
//     int freq_is_uniform = 0;
//     if ((freqs->length == 2) && (deltaF > 0.))
//     {
//         freq_is_uniform = 1;
//     }
//     else if ((freqs->length != 2) && (deltaF <= 0.))
//     {
//         freq_is_uniform = 0;
//     }
//
//     return freq_is_uniform;
// }

/**
 * derive frequency variables for PhenomHM based on input.
 * used to set the index on arrays where we have non-zero values.
 */
int init_IMRPhenomHMGet_FrequencyBounds_storage(
    PhenomHMFrequencyBoundsStorage *p, /**< [out] PhenomHMFrequencyBoundsStorage struct */
    real_vector* freqs,                /**< [in] list of GW frequencies [Hz] */
    double Mtot,                       /**< total mass in solar masses */
    //double deltaF,                   /**< frequency spacing */
    double f_ref_in                    /**< reference GW frequency */
)
{
    //p->deltaF = deltaF;
    /* determine how to populate frequency sequence */
    /* if len(freqs_in) == 2 and deltaF > 0. then
     * f_min = freqs_in[0]
     * f_max = freqs_in[1]
     * else if len(freqs_in) != 2 and deltaF <= 0. then
     * user has given an arbitrary set of frequencies to evaluate the model at.
     */

    //p->freq_is_uniform = IMRPhenomHM_is_freq_uniform(freqs, p->deltaF);

    // if (p->freq_is_uniform == 1)
    // { /* This case we use regularly spaced frequencies */
    //     p->f_min = freqs->data[0];
    //     p->f_max = freqs->data[1];
    //
    //     /* If p->f_max == 0. Then we default to the ending frequency
    //      * for PhenomHM
    //      */
    //     if (p->f_max == 0.)
    //     {
    //         p->f_max = PhenomUtilsMftoHz(
    //             PHENOMHM_DEFAULT_MF_MAX, Mtot);
    //     }
    //     /* we only need to evaluate the phase from
    //      * f_min to f_max with a spacing of deltaF
    //      */
    //     p->npts = PhenomInternal_NextPow2(p->f_max / p->deltaF) + 1;
    //     p->ind_min = (size_t)ceil(p->f_min / p->deltaF);
    //     p->ind_max = (size_t)ceil(p->f_max / p->deltaF);
    //     CHECK((p->ind_max <= p->npts) && (p->ind_min <= p->ind_max), ERROR_EDOM, "minimum freq index %zu and maximum freq index %zu do not fulfill 0<=ind_min<=ind_max<=npts=%zu.", p->ind_min, p->ind_max, p->npts);
    // }
    // else if (p->freq_is_uniform == 0)
    // { /* This case we possibly use irregularly spaced frequencies */
        /* Check that the frequencies are always increasing */
        for (int i = 0; i < (int)freqs->size - 1; i++)
        {
            CHECK(
                freqs->data[i] - freqs->data[i + 1] < 0.,
                ERROR_EFUNC,
                "custom frequencies must be increasing.");
        }

        //PRINT_INFO("Using custom frequency input.\n");
        p->f_min = freqs->data[0];
        p->f_max = freqs->data[freqs->size - 1]; /* Last element */

        p->npts = freqs->size;
        p->ind_min = 0;
        p->ind_max = p->npts;
    // }
    // else
    // { /* Throw an informative error. */
    //     XLAL_PRINT_ERROR("Input sequence of frequencies and deltaF is not \
    // compatible.\nSpecify a f_min and f_max by using a doubleSequence of length = 2 \
    // along with a deltaF > 0.\
    // \nIf you want to supply an arbitrary list of frequencies to evaluate the with \
    // then supply those frequencies using a doubleSequence and also set deltaF <= 0.");
    // }

    /* Fix default behaviour for f_ref */
    /* If f_ref = 0. then set f_ref = f_min */
    p->f_ref = f_ref_in;
    // if (p->f_ref == 0.)
    // {
    //     p->f_ref = p->f_min;
    // }

    return SUCCESS;
}

/**
 * Precompute a bunch of PhenomHM related quantities and store them filling in a
 * PhenomHMStorage variable
 */
static int init_PhenomHM_Storage(
    PhenomHMStorage* p,
    const double m1_SI,
    const double m2_SI,
    const double chi1z,
    const double chi2z,
    const double distance,
    real_vector* freqs,
    //const double deltaF,
    const double f_ref,
    const double phiRef)
{
    int retcode;
    CHECK(0 != p, ERROR_EFAULT, "p is NULL");

    p->m1 = m1_SI / MSUN_SI;
    p->m2 = m2_SI / MSUN_SI;
    p->m1_SI = m1_SI;
    p->m2_SI = m2_SI;
    p->Mtot = p->m1 + p->m2;
    p->eta = p->m1 * p->m2 / (p->Mtot * p->Mtot);
    p->chi1z = chi1z;
    p->chi2z = chi2z;
    p->distance = distance;
    p->phiRef = phiRef;
    //p->deltaF = deltaF;
    p->freqs = freqs;

    if (p->eta > 0.25)
        PhenomInternal_nudge(&(p->eta), 0.25, 1e-6);
    if (p->eta > 0.25 || p->eta < 0.0)
        ERROR(ERROR_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");
    // if (p->eta < MAX_ALLOWED_ETA)
    //     WARNING("The model is not calibrated for mass-ratios above 20\n");

    // retcode = 0;
    // retcode = PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
    //     &(p->m1),
    //     &(p->m2),
    //     &(p->chi1z),
    //     &(p->chi2z));
    // CHECK(
    //     SUCCESS == retcode,
    //     ERROR_EFUNC,
    //     "PhenomInternal_AlignedSpinEnforcePrimaryIsm1 failed");

    CHECK(
        m1_SI >= m2_SI,
        ERROR_EINVAL,
        "Need m1 >= m2.");

    /* Factors to go from geometric modes/frequencies to physical ones */
    p->Ms = p->Mtot * MTSUN_SI;
    double distance_SI = p->distance * 1e6*PC_SI;
    p->amp0 = PhenomUtilsFDamp0(p->Mtot, distance_SI);

    /* sanity checks on frequencies */
    PhenomHMFrequencyBoundsStorage pHMFS;
    retcode = 0;
    retcode = init_IMRPhenomHMGet_FrequencyBounds_storage(
        &pHMFS,
        p->freqs,
        p->Mtot,
        //p->deltaF,
        f_ref);
    CHECK(
        SUCCESS == retcode,
        ERROR_EFUNC,
        "init_IMRPhenomHMGet_FrequencyBounds_storage failed");

    /* redundent storage */
    p->f_min = pHMFS.f_min;
    p->f_max = pHMFS.f_max;
    p->f_ref = pHMFS.f_ref;
    p->freq_is_uniform = pHMFS.freq_is_uniform;
    p->npts = pHMFS.npts;
    p->ind_min = pHMFS.ind_min;
    p->ind_max = pHMFS.ind_max;

    p->Mf_ref = PhenomUtilsHztoMf(p->f_ref, p->Mtot);

    p->finmass = IMRPhenomDFinalMass(p->m1, p->m2, p->chi1z, p->chi2z);
    p->finspin = IMRPhenomDFinalSpin(p->m1, p->m2, p->chi1z, p->chi2z); /* dimensionless final spin */
    if (p->finspin > 1.0)
        ERROR(ERROR_EDOM, "PhenomD fring function: final spin > 1.0 not supported\n");

    /* populate the ringdown frequency array */
    /* If you want to model a new mode then you have to add it here. */
    /* (l,m) = (2,2) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[2][2],
        &p->PhenomHMfdamp[2][2],
        2, 2,
        p->finmass, p->finspin);

    /* (l,m) = (2,1) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[2][1],
        &p->PhenomHMfdamp[2][1],
        2, 1,
        p->finmass, p->finspin);

    /* (l,m) = (3,3) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[3][3],
        &p->PhenomHMfdamp[3][3],
        3, 3,
        p->finmass, p->finspin);

    /* (l,m) = (3,2) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[3][2],
        &p->PhenomHMfdamp[3][2],
        3, 2,
        p->finmass, p->finspin);

    /* (l,m) = (4,4) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[4][4],
        &p->PhenomHMfdamp[4][4],
        4, 4,
        p->finmass, p->finspin);

    /* (l,m) = (4,3) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[4][3],
        &p->PhenomHMfdamp[4][3],
        4, 3,
        p->finmass, p->finspin);

    p->Mf_RD_22 = p->PhenomHMfring[2][2];
    p->Mf_DM_22 = p->PhenomHMfdamp[2][2];

    /* (l,m) = (2,2) */
    int ell, mm;
    ell = 2;
    mm = 2;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (2,1) */
    ell = 2;
    mm = 1;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (3,3) */
    ell = 3;
    mm = 3;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (3,2) */
    ell = 3;
    mm = 2;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (4,4) */
    ell = 4;
    mm = 4;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (4,3) */
    ell = 4;
    mm = 3;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;

    return SUCCESS;
};

/**
 * domain mapping function - ringdown
 */
double IMRPhenomHMTrd(
    double Mf,
    double Mf_RD_22,
    double Mf_RD_lm,
    const int AmpFlag,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM)
{
    double ans = 0.0;
    if (AmpFlag == AmpFlagTrue)
    {
        /* For amplitude */
        ans = Mf - Mf_RD_lm + Mf_RD_22; /*Used for the Amplitude as an approx fix for post merger powerlaw slope */
    }
    else
    {
        /* For phase */
        double Rholm = pHM->Rholm[ell][mm];
        ans = Rholm * Mf; /* Used for the Phase */
    }

    return ans;
}

/**
 * mathematica function Ti
 * domain mapping function - inspiral
 */
double IMRPhenomHMTi(double Mf, const int mm)
{
    return 2.0 * Mf / mm;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
int IMRPhenomHMSlopeAmAndBm(
    double *Am,
    double *Bm,
    const int mm,
    double fi,
    double fr,
    double Mf_RD_22,
    double Mf_RD_lm,
    const int AmpFlag,
    const int ell,
    PhenomHMStorage *pHM)
{
    double Trd = IMRPhenomHMTrd(fr, Mf_RD_22, Mf_RD_lm, AmpFlag, ell, mm, pHM);
    double Ti = IMRPhenomHMTi(fi, mm);

    //Am = ( Trd[fr]-Ti[fi] )/( fr - fi );
    *Am = (Trd - Ti) / (fr - fi);

    //Bm = Ti[fi] - fi*Am;
    *Bm = Ti - fi * (*Am);

    return SUCCESS;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
int IMRPhenomHMMapParams(
    double *a,
    double *b,
    double flm,
    double fi,
    double fr,
    double Ai,
    double Bi,
    double Am,
    double Bm,
    double Ar,
    double Br)
{
    // Define function to output map params used depending on
    if (flm > fi)
    {
        if (flm > fr)
        {
            *a = Ar;
            *b = Br;
        }
        else
        {
            *a = Am;
            *b = Bm;
        }
    }
    else
    {
        *a = Ai;
        *b = Bi;
    };
    return SUCCESS;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
int IMRPhenomHMFreqDomainMapParams(
    double *a,             /**< [Out]  */
    double *b,             /**< [Out]  */
    double *fi,            /**< [Out]  */
    double *fr,            /**< [Out]  */
    double *f1,            /**< [Out]  */
    const double flm,      /**< input waveform frequency */
    const int ell,       /**< spherical harmonics ell mode */
    const int mm,        /**< spherical harmonics m mode */
    PhenomHMStorage *pHM, /**< Stores quantities in order to calculate them only once */
    const int AmpFlag     /**< is ==1 then computes for amplitude, if ==0 then computes for phase */
)
{

    /*check output points are NULL*/
    CHECK(a != NULL, ERROR_EFAULT, "");
    CHECK(b != NULL, ERROR_EFAULT, "");
    CHECK(fi != NULL, ERROR_EFAULT, "");
    CHECK(fr != NULL, ERROR_EFAULT, "");
    CHECK(f1 != NULL, ERROR_EFAULT, "");

    /* Account for different f1 definition between PhenomD Amplitude and Phase derivative models */
    double Mf_1_22 = 0.; /* initalise variable */
    if (AmpFlag == AmpFlagTrue)
    {
        /* For amplitude */
        Mf_1_22 = AMP_fJoin_INS; /* inspiral joining frequency from PhenomD [amplitude model], for the 22 mode */
    }
    else
    {
        /* For phase */
        Mf_1_22 = PHI_fJoin_INS; /* inspiral joining frequency from PhenomD [phase model], for the 22 mode */
    }

    *f1 = Mf_1_22;

    double Mf_RD_22 = pHM->Mf_RD_22;
    double Mf_RD_lm = pHM->PhenomHMfring[ell][mm];

    // Define a ratio of QNM frequencies to be used for scaling various quantities
    double Rholm = pHM->Rholm[ell][mm];

    // Given experiments with the l!=m modes, it appears that the QNM scaling rather than the PN scaling may be optimal for mapping f1
    double Mf_1_lm = Mf_1_22 / Rholm;

    /* Define transition frequencies */
    *fi = Mf_1_lm;
    *fr = Mf_RD_lm;

    /*Define the slope and intercepts of the linear transformation used*/
    double Ai = 2.0 / mm;
    double Bi = 0.0;
    double Am;
    double Bm;
    IMRPhenomHMSlopeAmAndBm(&Am, &Bm, mm, *fi, *fr, Mf_RD_22, Mf_RD_lm, AmpFlag, ell, pHM);

    double Ar = 1.0;
    double Br = 0.0;
    if (AmpFlag == AmpFlagTrue)
    {
        /* For amplitude */
        Br = -Mf_RD_lm + Mf_RD_22;
    }
    else
    {
        /* For phase */
        Ar = Rholm;
    }

    /* Define function to output map params used depending on */
    int ret = IMRPhenomHMMapParams(a, b, flm, *fi, *fr, Ai, Bi, Am, Bm, Ar, Br);
    if (ret != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMMapParams failed in IMRPhenomHMFreqDomainMapParams (1)");
    }

    return SUCCESS;
}

/**
 * IMRPhenomHMFreqDomainMap
 * Input waveform frequency in Geometric units (Mflm)
 * and computes what frequency this corresponds
 * to scaled to the 22 mode.
 */
double IMRPhenomHMFreqDomainMap(
    double Mflm,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM,
    const int AmpFlag)
{

    /* Mflm here has the same meaning as Mf_wf in XLALSimIMRPhenomHMFreqDomainMapHM (old deleted function). */
    double a = 0.;
    double b = 0.;
    /* Following variables not used in this funciton but are returned in IMRPhenomHMFreqDomainMapParams */
    double fi = 0.;
    double fr = 0.;
    double f1 = 0.;
    int ret = IMRPhenomHMFreqDomainMapParams(&a, &b, &fi, &fr, &f1, Mflm, ell, mm, pHM, AmpFlag);
    if (ret != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMap");
    }
    double Mf22 = a * Mflm + b;
    return Mf22;
}

int IMRPhenomHMPhasePreComp(
    HMPhasePreComp *q,          /**< [out] HMPhasePreComp struct */
    const int ell,             /**< ell spherical harmonic number */
    const int mm,              /**< m spherical harmonic number */
    PhenomHMStorage *pHM,      /**< PhenomHMStorage struct */
    PhenDAmpAndPhasePreComp* pDPreComp, /**< PhenDAmpAndPhasePreComp struct */
    const ModGRParams* modgrparams  /**< Modified GR parameters */
    //LALDict *extraParams /**< LALDict strcut */
)
{
    double ai = 0.0;
    double bi = 0.0;
    double am = 0.0;
    double bm = 0.0;
    double ar = 0.0;
    double br = 0.0;
    double fi = 0.0;
    double f1 = 0.0;
    double fr = 0.0;

    const int AmpFlag = 0;

    /* NOTE: As long as Mfshit isn't >= fr then the value of the shift is arbitrary. */
    const double Mfshift = 0.0001;

    int ret = IMRPhenomHMFreqDomainMapParams(&ai, &bi, &fi, &fr, &f1, Mfshift, ell, mm, pHM, AmpFlag);
    if (ret != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMapParams - inspiral");
    }
    q->ai = ai;
    q->bi = bi;

    ret = IMRPhenomHMFreqDomainMapParams(&am, &bm, &fi, &fr, &f1, fi + Mfshift, ell, mm, pHM, AmpFlag);
    if (ret != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMapParams - intermediate");
    }
    q->am = am;
    q->bm = bm;

    ret = IMRPhenomHMFreqDomainMapParams(&ar, &br, &fi, &fr, &f1, fr + Mfshift, ell, mm, pHM, AmpFlag);
    if (ret != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMapParams - merger-ringdown");
    }

    q->ar = ar;
    q->br = br;

    q->fi = fi;
    q->fr = fr;

    double Rholm = pHM->Rholm[ell][mm];
    double Taulm = pHM->Taulm[ell][mm];

    double PhDBMf = am * fi + bm;
    q->PhDBconst = IMRPhenomDPhase_OneFrequency(PhDBMf, *pDPreComp, Rholm, Taulm, modgrparams) / am;

    double PhDCMf = ar * fr + br;
    q->PhDCconst = IMRPhenomDPhase_OneFrequency(PhDCMf, *pDPreComp, Rholm, Taulm, modgrparams) / ar;

    double PhDBAMf = ai * fi + bi;
    q->PhDBAterm = IMRPhenomDPhase_OneFrequency(PhDBAMf, *pDPreComp, Rholm, Taulm, modgrparams) / ai;
    return SUCCESS;
}

/**
 * Define function for FD PN amplitudes
 */
double complex IMRPhenomHMOnePointFiveSpinPN(
    double fM,
    int l,
    int m,
    double M1,
    double M2,
    double X1z,
    double X2z)
{

    // LLondon 2017

    // Define effective intinsic parameters
    double complex Hlm = 0;
    double M_INPUT = M1 + M2;
    M1 = M1 / (M_INPUT);
    M2 = M2 / (M_INPUT);
    double M = M1 + M2;
    double eta = M1 * M2 / (M * M);
    double delta = sqrt(1.0 - 4 * eta);
    double Xs = 0.5 * (X1z + X2z);
    double Xa = 0.5 * (X1z - X2z);
    double complex ans = 0;

    // Define PN parameter and realed powers
    double v = pow(M * 2.0 * PI  * fM / m, 1.0 / 3.0);
    double v2 = v * v;
    double v3 = v * v2;

    // Define Leading Order Ampitude for each supported multipole
    if (l == 2 && m == 2)
    {
        // (l,m) = (2,2)
        // THIS IS LEADING ORDER
        Hlm = 1.0;
    }
    else if (l == 2 && m == 1)
    {
        // (l,m) = (2,1)
        // SPIN TERMS ADDED

        // UP TO 4PN
        double v4 = v * v3;
        Hlm = (sqrt(2.0) / 3.0) * \
            ( \
                v * delta - v2 * 1.5 * (Xa + delta * Xs) + \
                v3 * delta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)
            ) \
            + \
            v4 * \
                ( \
                Xa * (3427.0 / 1344 - eta * 2101.0 / 336) + \
                delta * Xs * (3427.0 / 1344 - eta * 965 / 336) + \
                delta * (-I * 0.5 - PI  - 2 * I * 0.69314718056) \
                )
            );
    }
    else if (l == 3 && m == 3)
    {
        // (l,m) = (3,3)
        // THIS IS LEADING ORDER
        Hlm = 0.75 * sqrt(5.0 / 7.0) * (v * delta);
    }
    else if (l == 3 && m == 2)
    {
        // (l,m) = (3,2)
        // NO SPIN TERMS to avoid roots
        Hlm = (1.0 / 3.0) * sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta));
    }
    else if (l == 4 && m == 4)
    {
        // (l,m) = (4,4)
        // THIS IS LEADING ORDER
        Hlm = (4.0 / 9.0) * sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta);
    }
    else if (l == 4 && m == 3)
    {
        // (l,m) = (4,3)
        // NO SPIN TERMS TO ADD AT DESIRED ORDER
        Hlm = 0.75 * sqrt(3.0 / 35.0) * v3 * delta * (1.0 - 2.0 * eta);
    }
    else
    {
        ERROR(ERROR_EDOM, "requested (l,m) mode not available, check documentation for available modes\n");
    }
    // Compute the final PN Amplitude at Leading Order in fM
    ans = M * M * PI  * sqrt(eta * 2.0 / 3) * pow(v, -3.5) * cabs(Hlm);

    return ans;
}

/**
 * PhenomHM in-place PN amplitude scaling -- based on IMRPhenomHMOnePointFiveSpinPN
 */
int IMRPhenomHMAmpPNScaling(
    real_vector* amps,
    real_vector* freqs_amp,
    real_vector* freqs_geom,
    double m1,
    double m2,
    double chi1z,
    double chi2z,
    int ell,
    int mm)
{
    // Define intinsic parameters
    double M = m1 + m2;
    double eta = m1 * m2 / (M * M);
    double delta = sqrt(1.0 - 4 * eta);
    double chis = 0.5 * (chi1z + chi2z);
    double chia = 0.5 * (chi1z - chi2z);

    /* Setting up coefficients of v */
    double complex coeffv0 = 0.;
    double complex coeffv1 = 0.;
    double complex coeffv2 = 0.;
    double complex coeffv3 = 0.;
    double complex coeffv4 = 0.;

    // Define Leading Order Ampitude for each supported multipole
    if (ell == 2 && mm == 2)
    {
        // (l,m) = (2,2)
        // THIS IS LEADING ORDER
        coeffv0 = 1.0;
    }
    else if (ell == 2 && mm == 1)
    {
        // (l,m) = (2,1)
        // SPIN TERMS ADDED

        // UP TO 4PN
        double sqrt2ov3 = sqrt(2.0) / 3.0;
        coeffv1 = sqrt2ov3 * delta;
        coeffv2 = sqrt2ov3 * (-1.5) * (chia + delta * chis);
        coeffv3 = sqrt2ov3 * delta * ((335.0 / 672.0) + (eta * 117.0 / 56.0));
        coeffv4 = sqrt2ov3 * (chia * (3427.0 / 1344 - eta * 2101.0 / 336) + delta * chis * (3427.0 / 1344 - eta * 965. / 336) + delta * (-I * 0.5 - PI - 2 * I * 0.69314718056));
    }
    else if (ell == 3 && mm == 3)
    {
        // (l,m) = (3,3)
        // THIS IS LEADING ORDER
        coeffv1 = 0.75 * sqrt(5.0 / 7.0) * delta;
    }
    else if (ell == 3 && mm == 2)
    {
        // (l,m) = (3,2)
        // NO SPIN TERMS to avoid roots
        coeffv2 = (1.0 / 3.0) * sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta);
    }
    else if (ell == 4 && mm == 4)
    {
        // (l,m) = (4,4)
        // THIS IS LEADING ORDER
        coeffv2 = (4.0 / 9.0) * sqrt(10.0 / 7.0) * (1.0 - 3.0 * eta);
    }
    else if (ell == 4 && mm == 3)
    {
        // (l,m) = (4,3)
        // NO SPIN TERMS TO ADD AT DESIRED ORDER
        coeffv3 = 0.75 * sqrt(3.0 / 35.0) * delta * (1.0 - 2.0 * eta);
    }
    else
    {
        ERROR(ERROR_EDOM, "requested (l,m) mode not available, check documentation for available modes\n");
    }

    /* We keep blindly the terminology surrounding the various calls to IMRPhenomHMOnePointFiveSpinPN... */
    /* Notation: in the original code, f is freqs_geom, fmapped is freqs_amp */
    /* NOTE: we removed a constant prefactor M * M * PI  * sqrt(eta * 2.0 / 3) in IMRPhenomHMOnePointFiveSpinPN since it scales out */
    double beta_term1 = 0.;
    double beta_term2 = 0.;
    double HMamp_term1 = 0.;
    double HMamp_term2 = 0.;
    double scaling = 0.;
    /* NOTE: v will represent different quantities in different terms */
    double v = 0.;
    double v2 = 0.;
    double v3 = 0.;
    double v4 = 0.;
    double Mf = 0.;
    for (int i=0; i<(int)freqs_geom->size; i++) {
      /* beta_term1 in original code: IMRPhenomHMOnePointFiveSpinPN(f) */
      Mf = freqs_geom->data[i];
      /* NOTE: the notation v here in the original code is actually already for the scaled frequency 2f/m */
      v = pow(2.0 * PI * Mf / mm, 1.0 / 3.0);
      v2 = v * v;
      v3 = v * v2;
      v4 = v * v3;
      beta_term1 = pow(v, -7./2) * cabs(coeffv0 + v * (coeffv1 + v * (coeffv2 + v * (coeffv3 + v * coeffv4))));

      /* beta_term2 in original code: IMRPhenomHMOnePointFiveSpinPN(2f/m) */
      /* Very weird that we re-scale yet again by 2/m... */
      Mf = 2. / mm * freqs_geom->data[i];
      /* NOTE: the notation v here in the original code is actually already for the scaled frequency 2f/m */
      v = pow(2.0 * PI * Mf / mm, 1.0 / 3.0);
      v2 = v * v;
      v3 = v * v2;
      v4 = v * v3;
      beta_term2 = pow(v, -7./2) * cabs(coeffv0 + v * (coeffv1 + v * (coeffv2 + v * (coeffv3 + v * coeffv4))));

      /* HMamp_term1 in original code: IMRPhenomHMOnePointFiveSpinPN(fmapped) */
      /* Very weird that we re-scale yet again by 2/m... */
      Mf = freqs_amp->data[i];
      /* NOTE: the notation v here in the original code is actually already for the scaled frequency 2f/m */
      v = pow(2.0 * PI * Mf / mm, 1.0 / 3.0);
      v2 = v * v;
      v3 = v * v2;
      v4 = v * v3;
      HMamp_term1 = pow(v, -7./2) * cabs(coeffv0 + v * (coeffv1 + v * (coeffv2 + v * (coeffv3 + v * coeffv4))));

      /* HMamp_term2 in original code: IMRPhenomHMOnePointFiveSpinPN(fmapped, 22, chi=0) */
      Mf = freqs_amp->data[i];
      /* NOTE: the notation v here in the original code is actually already for the scaled frequency 2f/m */
      v = pow(2.0 * PI * Mf / 2, 1.0 / 3.0);
      /* For 22, no PN corrections, and we scaled out the constant prefactor leaving only v^(-7/2) */
      HMamp_term2 = pow(v, -7./2);

      /* For equal masses, original comment: */
      //HACK to fix equal black hole case producing NaNs.
      //More elegant solution needed.
      if (beta_term1==0.) scaling = 0.;
      else scaling = beta_term1 / beta_term2 * HMamp_term1 / HMamp_term2;
      amps->data[i] *= scaling;
    }

    return SUCCESS;
}

/**
 * @addtogroup LALSimIMRPhenom_c
 * @{
 *
 * @name Routines for IMR Phenomenological Model "HM"
 * @{
 *
 * @author Sebastian Khan, Francesco Pannarale, Lionel London
 *
 * @brief C code for IMRPhenomHM phenomenological waveform model.
 *
 * Inspiral-merger and ringdown phenomenological, frequecny domain
 * waveform model for binary black holes systems.
 * Models not only the dominant (l,|m|) = (2,2) modes
 * but also some of the sub-domant modes too.
 * Model described in PhysRevLett.120.161102/1708.00404.
 * The model is based on IMRPhenomD (\cite Husa:2015iqa, \cite Khan:2015jqa)
 *
 * @note The higher mode information was not calibrated to Numerical Relativity
 * simulation therefore the calibration range is inherited from PhenomD.
 *
 * @attention The model is usable outside this parameter range,
 * and in tests to date gives sensible physical results,
 * but conclusive statements on the physical fidelity of
 * the model for these parameters await comparisons against further
 * numerical-relativity simulations. For more information, see the review wiki
 * under https://git.ligo.org/waveforms/reviews/phenomhm/wikis/home
 * Also a technical document in the DCC https://dcc.ligo.org/LIGO-T1800295
 */

/**
 * Returns h+ and hx in the frequency domain.
 *
 * This function can be called in the usual sense
 * where you supply a f_min, f_max and deltaF.
 * This is the case when deltaF > 0.
 * If f_max = 0. then the default ending frequnecy is used.
 * or you can also supply a custom set of discrete
 * frequency points with which to evaluate the waveform.
 * To do this you must call this function with
 * deltaF <= 0.
 *
 */
// UNUSED int IMRPhenomHM(
//     UNUSED COMPLEX16FrequencySeries **hptilde, /**< [out] Frequency-domain waveform h+ */
//     UNUSED COMPLEX16FrequencySeries **hctilde, /**< [out] Frequency-domain waveform hx */
//     UNUSED doubleSequence *freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
//     UNUSED double m1_SI,                        /**< mass of companion 1 (kg) */
//     UNUSED double m2_SI,                        /**< mass of companion 2 (kg) */
//     UNUSED double chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
//     UNUSED double chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
//     UNUSED const double distance,               /**< distance of source (m) */
//     UNUSED const double inclination,            /**< inclination of source (rad) */
//     UNUSED const double phiRef,                 /**< reference orbital phase (rad) */
//     UNUSED const double deltaF,                 /**< Sampling frequency (Hz). To use arbitrary frequency points set deltaF <= 0. */
//     UNUSED double f_ref,                        /**< Reference frequency */
//     UNUSED LALDict *extraParams                /**<linked list containing the extra testing GR parameters */
// )
// {
//     /* define and init return code for this function */
//     int retcode;
//
//     /* sanity checks on input parameters: check pointers, etc. */
//     /* NOTE: a lot of checks are done in the function
//      * XLALSimIMRPhenomHMGethlmModes because that can also be used
//      * as a standalone function. It gets called through IMRPhenomHMCore
//      * so to avoid doubling up on checks alot of the checks are done in
//      * XLALSimIMRPhenomHMGethlmModes.
//      */
//     CHECK(NULL != hptilde, ERROR_EFAULT);
//     CHECK(NULL != hctilde, ERROR_EFAULT);
//     CHECK(*hptilde == NULL, ERROR_EFAULT);
//     CHECK(*hctilde == NULL, ERROR_EFAULT);
//     CHECK(distance > 0, ERROR_EDOM, "distance must be positive.\n");
//
//     /* main: evaluate model at given frequencies */
//     retcode = 0;
//     retcode = IMRPhenomHMCore(
//         hptilde,
//         hctilde,
//         freqs,
//         m1_SI,
//         m2_SI,
//         chi1z,
//         chi2z,
//         distance,
//         inclination,
//         phiRef,
//         deltaF,
//         f_ref,
//         extraParams);
//     CHECK(retcode == SUCCESS,
//                ERROR_EFUNC, "IMRPhenomHMCore failed in XLALSimIMRPhenomHM.");
//
//     /* cleanup */
//     /* XLALDestroy and XLALFree any pointers. */
//
//     return SUCCESS;
// }

/** @} */
/** @} */

// /**
//  * internal function that returns h+ and hx.
//  * Inside this function the my bulk of the work is done
//  * like the loop over frequencies.
//  */
// int IMRPhenomHMCore(
//     UNUSED COMPLEX16FrequencySeries **hptilde, /**< [out] Frequency domain h+ GW strain */
//     UNUSED COMPLEX16FrequencySeries **hctilde, /**< [out] Frequency domain hx GW strain */
//     doubleSequence *freqs,                      /**< GW frequecny list [Hz] */
//     double m1_SI,                               /**< primary mass [kg] */
//     double m2_SI,                               /**< secondary mass [kg] */
//     double chi1z,                               /**< aligned spin of primary */
//     double chi2z,                               /**< aligned spin of secondary */
//     const double distance,                      /**< distance [m] */
//     const double inclination,                   /**< inclination angle */
//     const double phiRef,                        /**< orbital phase at f_ref */
//     const double deltaF,                        /**< frequency spacing */
//     double f_ref,                               /**< reference GW frequency */
//     LALDict *extraParams                       /**< LALDict struct */
// )
// {
//     int retcode;
//
//     // Make a pointer to LALDict to circumvent a memory leak
//     // At the end we will check if we created a LALDict in extraParams
//     // and destroy it if we did.
//     LALDict *extraParams_in = extraParams;
//
//     /* evaluate all hlm modes */
//     SphHarmFrequencySeries **hlms = XLALMalloc(sizeof(SphHarmFrequencySeries));
//     *hlms = NULL;
//     retcode = 0;
//     retcode = IMRPhenomHMGethlmModes(
//         hlms,
//         freqs,
//         m1_SI,
//         m2_SI,
//         chi1z,
//         chi2z,
//         phiRef,
//         deltaF,
//         f_ref,
//         extraParams);
//     CHECK(SUCCESS == retcode,
//                ERROR_EFUNC, "XLALSimIMRPhenomHMGethlmModes failed");
//
//     /* compute the frequency bounds */
//     const double Mtot = (m1_SI + m2_SI) / LAL_MSUN_SI;
//     PhenomHMFrequencyBoundsStorage *pHMFS;
//     pHMFS = XLALMalloc(sizeof(PhenomHMFrequencyBoundsStorage));
//     retcode = 0;
//     retcode = init_IMRPhenomHMGet_FrequencyBounds_storage(
//         pHMFS,
//         freqs,
//         Mtot,
//         deltaF,
//         f_ref);
//     CHECK(SUCCESS == retcode,
//                ERROR_EFUNC, "init_IMRPhenomHMGet_FrequencyBounds_storage failed");
//
//     /* now we have generated all hlm modes we need to
//      * multiply them with the Ylm's and sum them.
//      */
//
//     //LIGOTimeGPS tC = LIGOTIMEGPSZERO; // = {0, 0}
//     long ligotimegps_zero = 0;
//     if (pHMFS->freq_is_uniform == 1)
//     { /* 1. uniformly spaced */
//         PRINT_INFO("freq_is_uniform = True\n");
//         /* coalesce at t=0 */
//         /* Shift by overall length in time */
//         ligotimegps_zero += -1. / deltaF;
// //         CHECK(
// //             XLALGPSAdd(&tC, -1. / deltaF),
// //             ERROR_EFUNC,
// //             "Failed to shift coalescence time to t=0,\
// // tried to apply shift of -1.0/deltaF with deltaF=%g.",
// //             deltaF);
//     } /* else if 2. i.e. not uniformly spaced then we don't shift. */
//
//     /* Allocate hptilde and hctilde */
//     *hptilde = CreateCOMPLEX16FrequencySeries("hptilde: FD waveform", ligotimegps_zero, 0.0, deltaF, &lalStrainUnit, pHMFS->npts);
//     if (!(hptilde))
//         ERROR(ERROR_EFUNC);
//     memset((*hptilde)->data->data, 0, pHMFS->npts * sizeof(COMPLEX16));
//     //XLALUnitDivide(&(*hptilde)->sampleUnits, &(*hptilde)->sampleUnits, &lalSecondUnit);
//
//     *hctilde = CreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &tC, 0.0, deltaF, &lalStrainUnit, pHMFS->npts);
//     if (!(hctilde))
//         ERROR(ERROR_EFUNC);
//     memset((*hctilde)->data->data, 0, pHMFS->npts * sizeof(COMPLEX16));
//     //XLALUnitDivide(&(*hctilde)->sampleUnits, &(*hctilde)->sampleUnits, &lalSecondUnit);
//
//     /* Adding the modes to form hplus, hcross
//      * - use of a function that copies XLALSimAddMode but for Fourier domain structures */
//     INT4 sym; /* sym will decide whether to add the -m mode (when equatorial symmetry is present) */
//
//     /* setup ModeArray */
//     if (extraParams == NULL)
//     {
//         extraParams = XLALCreateDict();
//     }
//     extraParams = IMRPhenomHM_setup_mode_array(extraParams);
//     LALValue *ModeArray = XLALSimInspiralWaveformParamsLookupModeArray(extraParams);
//
//     /* loop over modes */
//     /* at this point ModeArray should contain the list of modes
//      * and therefore if NULL then something is wrong and abort.
//      */
//     if (ModeArray == NULL)
//     {
//         ERROR(ERROR_EDOM, "ModeArray is NULL when it shouldn't be. Aborting.\n");
//     }
//     for (int ell = 2; ell < L_MAX_PLUS_1; ell++)
//     {
//         for (INT4 mm = 1; mm < (INT4)ell + 1; mm++)
//         { /* loop over only positive m is intentional. negative m added automatically */
//             /* first check if (l,m) mode is 'activated' in the ModeArray */
//             /* if activated then generate the mode, else skip this mode. */
//             if (XLALSimInspiralModeArrayIsModeActive(ModeArray, ell, mm) != 1)
//             { /* skip mode */
//                 continue;
//             } /* else: generate mode */
//
//             COMPLEX16FrequencySeries *hlm = XLALSphHarmFrequencySeriesGetMode(*hlms, ell, mm);
//             if (!(hlm))
//                 ERROR(ERROR_EFUNC);
//
//             /* We test for hypothetical m=0 modes */
//             if (mm == 0)
//             {
//                 sym = 0;
//             }
//             else
//             {
//                 sym = 1;
//             }
//             IMRPhenomHMFDAddMode(*hptilde, *hctilde, hlm, inclination, 0., ell, mm, sym); /* The phase \Phi is set to 0 - assumes phiRef is defined as half the phase of the 22 mode h22 */
//         }
//     }
//
//     XLALDestroySphHarmFrequencySeries(*hlms);
//     XLALFree(hlms);
//
//     /* Compute the amplitude pre-factor */
//     const double amp0 = PhenomUtilsFDamp0(Mtot, distance);
//     #pragma omp parallel for
//     for (size_t i = pHMFS->ind_min; i < pHMFS->ind_max; i++)
//     {
//         ((*hptilde)->data->data)[i] = ((*hptilde)->data->data)[i] * amp0;
//         ((*hctilde)->data->data)[i] = -1 * ((*hctilde)->data->data)[i] * amp0;
//     }
//
//     /* cleanup */
//     XLALDestroyValue(ModeArray);
//     LALFree(pHMFS);
//
//   /* If extraParams was allocated in this function and not passed in
//    * we need to free it to prevent a leak */
//   if (extraParams && !extraParams_in) {
//     XLALDestroyDict(extraParams);
//   }
//
//     return SUCCESS;
// }

/**
 * XLAL function that returns
 * a SphHarmFrequencySeries object
 * containing all the hlm modes
 * requested.
 * These have the correct relative phases between modes.
 * Note this has a similar interface to XLALSimIMRPhenomHM
 * because it is designed so it can be used independently.
 */
 /* TODO: clunky freq_lm duplicated by hand, need to pass lists in cython */
int IMRPhenomHMGethlmModes(
    ListAmpPhaseFDMode** hlms,  /**< [out] list of modes, FD amp/phase */
    double* fpeak,              /**< [out] Approximate 22 peak frequency (Hz) */
    double* tpeak,              /**< [out] tf 22 at peak frequency (s) */
    double* phipeak,            /**< [out] phase 22 at peak frequency */
    double* fstart,             /**< [out] Starting frequency (Hz) */
    double* tstart,             /**< [out] tf 22 at starting frequency (s) */
    double* phistart,           /**< [out] phase 22 at starting frequency */
    real_vector* freq_22,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_21,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_33,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_32,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_44,        /**< [in] frequency vector for lm in Hz */
    real_vector* freq_43,        /**< [in] frequency vector for lm in Hz */
    double m1,                   /**< primary mass [solar masses] */
    double m2,                   /**< secondary mass [solar masses] */
    double chi1z,                   /**< aligned spin of primary */
    double chi2z,                   /**< aligned spin of secondary */
    double distance,                /**< luminosity distance (Mpc) */
    //const double deltaF,            /**< frequency spacing */
    const double phiRef,            /**< orbital phase at f_ref */
    const double fRef_in,                   /**< reference GW frequency */
    const double Deltat,             /**< Time shift (s) applied a posteriori */
    const int scale_freq_hm,         /**< Scale mode freq by m/2, ignored if freq_lm is not NULL */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
    //LALDict *extraParams           /**< LALDict struct */
)
{
    int retcode;

    // Make a pointer to LALDict to circumvent a memory leak
    // At the end we will check if we created a LALDict in extraParams
    // and destroy it if we did.
    //LALDict *extraParams_in = extraParams;

    /* sanity checks on input parameters: check pointers, etc. */

    /* Check input pointer for freq_22 */
    /* Pointers can be NULL for other modes, for instance when rescaling */
    if (!freq_22) ERROR(ERROR_EINVAL, "Input freq_22 pointer cannot be NULL.\n");

    /* Check inputs for sanity */
    CHECK(m1 > 0, ERROR_EDOM, "m1 must be positive.\n");
    CHECK(m2 > 0, ERROR_EDOM, "m2 must be positive.\n");
    CHECK(fabs(chi1z) <= 1.0, ERROR_EDOM, "Aligned spin chi1z \
must be <= 1 in magnitude!\n");
    CHECK(fabs(chi2z) <= 1.0, ERROR_EDOM, "Aligned spin chi2z \
must be <= 1 in magnitude!\n");
    CHECK(fRef_in >= 0, ERROR_EDOM, "Reference frequency must be \
positive or 0.\n");

    double m1_SI = m1 * MSUN_SI;
    double m2_SI = m2 * MSUN_SI;

    double M = m1 + m2;
    double Ms = M * MTSUN_SI;
    double MfRef_in = Ms * fRef_in;

    /* Time shif tin geometric units */
    double DeltatM = Deltat / Ms;

    /* setup ModeArray */
    // if (extraParams == NULL)
    //     extraParams = XLALCreateDict();
    // extraParams = IMRPhenomHM_setup_mode_array(extraParams);
    // LALValue *ModeArray = XLALSimInspiralWaveformParamsLookupModeArray(extraParams);

    // LIGOTimeGPS tC = LIGOTIMEGPSZERO; // = {0, 0}

    /* setup PhenomHM model storage struct / structs */
    /* Compute quantities/parameters related to PhenomD only once and store them */
    PhenomHMStorage *pHM;
    pHM = malloc(sizeof(PhenomHMStorage));
    retcode = 0;
    retcode = init_PhenomHM_Storage(
        pHM,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        freq_22,
        //deltaF,
        fRef_in,
        phiRef);
    CHECK(SUCCESS == retcode, ERROR_EFUNC, "init_PhenomHM_Storage \
failed");

    /* Two possibilities */
//     if (pHM->freq_is_uniform == 1)
//     { /* 1. uniformly spaced */
//         PRINT_INFO("freq_is_uniform = True\n");
//
//         freqs = XLALCreatedoubleSequence(pHM->npts);
//         phases = XLALCreatedoubleSequence(pHM->npts);
//         amps = XLALCreatedoubleSequence(pHM->npts);
//
//         for (size_t i = 0; i < pHM->npts; i++)
//         {                                     /* populate the frequency unitformly from zero - this is the standard
//              convention we use when generating waveforms in LAL. */
//             freqs->data[i] = i * pHM->deltaF; /* This is in Hz */
//             phases->data[i] = 0;              /* initalise all phases to zero. */
//             amps->data[i] = 0;                /* initalise all amps to zero. */
//         }
//         /* coalesce at t=0 */
//         CHECK(
//             XLALGPSAdd(&tC, -1. / pHM->deltaF),
//             ERROR_EFUNC,
//             "Failed to shift coalescence time to t=0,\
// tried to apply shift of -1.0/deltaF with deltaF=%g.",
//             pHM->deltaF);
//     }
    // else if (pHM->freq_is_uniform == 0)
    // { /* 2. arbitrarily space */
        // PRINT_INFO("freq_is_uniform = False\n");

    // }
    // else
    // {
    //     ERROR(ERROR_EDOM, "freq_is_uniform = %i and should be either 0 or 1.", pHM->freq_is_uniform);
    // }

    PhenDAmpAndPhasePreComp pDPreComp22;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreComp22,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm[2][2],
        pHM->Taulm[2][2]);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
    }

    /* Copy amp and phase coefficients for later HM update of pPhi */
    /* TODO: for now, just redo it; we should copy instead */
    PhenDAmpAndPhasePreComp pDPreCompHM;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreCompHM,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm[2][2],
        pHM->Taulm[2][2]);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
    }

    /* Reference frequency for the phase (and not time) -- if 0, use fpeak */
    /* Mfpeak: approximate 'peak' frequency for Phenom waveforms */
    double Mfpeak = pDPreComp22.pAmp.fmaxCalc;
    double MfRef = (MfRef_in == 0.0) ? Mfpeak : MfRef_in;
    pHM->Mf_ref = MfRef;

    /* compute the reference phase shift need to align the waveform so that
     the phase is equal to phiRef at the reference frequency f_ref. */
    /* the phase shift is computed by evaluating the phase of the
    (l,m)=(2,2) mode.
    phi0 is the correction we need to add to each mode. */
    double phi_22_at_Mfref = IMRPhenomDPhase_OneFrequency(pHM->Mf_ref, pDPreComp22,
                                                         1.0, 1.0, modgrparams);
    //double phi0 = 0.5 * phi_22_at_f_ref + phiRef;
    double phi_shift = - 1./2 * phi_22_at_Mfref - phiRef;

    /* Compute the reference time shift (t/M) to enforce tf=Deltat at fpeak */
    /* Option to reproduce buggy time alignment in LAL PhenomD/HM */
    double tM_22_at_Mfpeak = 0.;
    if ((extraparams==NULL) || (!(extraparams->use_buggy_LAL_tpeak))) {
      tM_22_at_Mfpeak = 1./(2*PI) * IMRPhenomDPhaseDerivative_OneFrequency(
                                    Mfpeak, pDPreComp22, 1.0, 1.0, modgrparams);
    }
    else {
      tM_22_at_Mfpeak = 1./(2*PI) * DPhiMRD(Mfpeak, &(pDPreComp22.pPhi), 1., 1.);
    }

    /* Note: this is a time shift applied as 2pitime_shift*(Mf-Mfref) */
    /* Does not affect phi=-2phiref at fref, and does not include Deltat */
    /* Deltat is to be added as 2piDeltat*f, regardless of alignment at fref */
    double time_shift = -tM_22_at_Mfpeak;

    /* Scaling of frequencies between mode, either 1 or m/2 */
    double mode_freq_scaling = 0.;

    /* loop over modes */
    /* at this point ModeArray should contain the list of modes
     * and therefore if NULL then something is wrong and abort.
     */
    // if (ModeArray == NULL)
    // {
    //     ERROR(ERROR_EDOM, "ModeArray is NULL when it shouldn't be. Aborting.\n");
    // }
    real_vector* freq_lm = NULL;
    for (int ell = 2; ell < L_MAX_PLUS_1; ell++)
    {
        for (int mm = 1; mm < (int)ell + 1; mm++)
        { /* loop over only positive m is intentional. negative m added automatically */
            /* first check if (l,m) mode is 'activated' in the ModeArray */
            /* if activated then generate the mode, else skip this mode. */
            // if (InspiralModeArrayIsModeActive(ModeArray, ell, mm) != 1)
            // { /* skip mode */
            //     PRINT_INFO("SKIPPING ell = %i mm = %i\n", ell, mm);
            //     continue;
            // } /* else: generate mode */
            // PRINT_INFO("generateing ell = %i mm = %i\n", ell, mm);

            if (!((ell==2 && mm==2) || (ell==2 && mm==1) || (ell==3 && mm==3) || (ell==3 && mm==2) || (ell==4 && mm==4) || (ell==4 && mm==3))) continue;

            /* NOTE: clunky, need to past list structures by cython */
            if (ell==2 && mm==2) freq_lm = freq_22;
            if (ell==2 && mm==1) freq_lm = freq_21;
            if (ell==3 && mm==3) freq_lm = freq_33;
            if (ell==3 && mm==2) freq_lm = freq_32;
            if (ell==4 && mm==4) freq_lm = freq_44;
            if (ell==4 && mm==3) freq_lm = freq_43;

            /* Intermediate vectors */
            real_vector* amps = NULL;
            real_vector* phases = NULL;
            real_vector* tfs = NULL;
            real_vector* freqs = NULL; /* freqs is in pysical units (Hz) */

            /* If freq_lm is NULL, use freq_22, possibly with rescaling */
            /* Else, use freq_lm as input and ignore rescaling */
            if (!freq_lm) {

              /* PhenomD functions take geometric frequencies */
              freqs = real_vector_alloc(freq_22->size);

              /* Geometric frequencies for mode (l,m): if flag scale_freq_hm, scaled by m/2 from 22 frequencies */
              if (scale_freq_hm) mode_freq_scaling = mm/2.;
              else mode_freq_scaling = 1.;
              for (size_t i = 0; i < freq_22->size; i++)
              {
                  freqs->data[i] = mode_freq_scaling * freq_22->data[i];
              }
            }
            else {
              /* PhenomD functions take geometric frequencies */
              /* No mode rescaling here, take freq_lm as input by the user */
              freqs = real_vector_alloc(freq_lm->size);
              for (size_t i = 0; i < freq_lm->size; i++)
              {
                  freqs->data[i] = freq_lm->data[i];
              }
            }

            amps = real_vector_alloc(freqs->size);
            phases = real_vector_alloc(freqs->size);
            tfs = real_vector_alloc(freqs->size);
            /* initalise all to zero. */
            for (size_t i = 0; i < freqs->size; i++)
            {
              amps->data[i] = 0;
              phases->data[i] = 0;
              tfs->data[i] = 0;
            }

            // COMPLEX16FrequencySeries *hlm = CreateCOMPLEX16FrequencySeries("hlm: FD waveform", &tC, 0.0, pHM->deltaF, &lalStrainUnit, pHM->npts);
            // memset(hlm->data->data, 0, pHM->npts * sizeof(COMPLEX16));
            // XLALUnitMultiply(&((*hlm)->sampleUnits), &((*hlm)->sampleUnits), &lalSecondUnit);
            AmpPhaseFDMode* hlm = NULL;
            AmpPhaseFDMode_Init(&hlm, freqs->size, freqs->size);
            retcode = 0;
            retcode = IMRPhenomHMEvaluateOnehlmMode(hlm,
                                                    amps, phases, tfs,
                                                    freqs,
                                                    pHM,
                                                    &pDPreCompHM,
                                                    ell, mm,
                                                    time_shift,
                                                    phi_shift,
                                                    modgrparams);
                                                    //extraParams);
            CHECK(SUCCESS == retcode,
                       ERROR_EFUNC, "IMRPhenomHMEvaluateOnehlmMode failed");

            /* Apply time shift by Deltat (*after* alignment at fref) */
            for (size_t i = 0; i < freqs->size; i++)
            {
              hlm->phase->data[i] += 2*PI * DeltatM * Ms * freqs->data[i];
              hlm->tf->data[i] += DeltatM * Ms;
            }

            // *hlms = XLALSphHarmFrequencySeriesAddMode(*hlms, hlm, ell, mm);
            *hlms = ListAmpPhaseFDMode_AddMode(*hlms, hlm, ell, mm);

            real_vector_free(freqs);
            real_vector_free(amps);
            real_vector_free(phases);
            real_vector_free(tfs);
        }
    }

    /* Additional output: 22-mode phi,f,t at peak and at starting frequency */
    double phi22peak = IMRPhenomDPhase_OneFrequency(Mfpeak, pDPreComp22, 1.0, 1.0, modgrparams) + 2*PI*time_shift*(Mfpeak - pHM->Mf_ref) + 2*phi_shift + 2*PI*DeltatM*Mfpeak;
    double tM22peak = 1/(2*PI) * IMRPhenomDPhaseDerivative_OneFrequency(Mfpeak, pDPreComp22, 1.0, 1.0, modgrparams) + time_shift + DeltatM;
    double Mfstart = pHM->freqs->data[0] * Ms;
    double phi22start = IMRPhenomDPhase_OneFrequency(Mfstart, pDPreComp22, 1.0, 1.0, modgrparams) + 2*PI*time_shift*(Mfstart - pHM->Mf_ref) + 2*phi_shift + 2*PI*DeltatM*Mfstart;
    double tM22start = 1/(2*PI) * IMRPhenomDPhaseDerivative_OneFrequency(Mfstart, pDPreComp22, 1.0, 1.0, modgrparams) + time_shift + DeltatM;
    *fpeak = Mfpeak / Ms;
    *tpeak = tM22peak * Ms;
    *phipeak = phi22peak;
    *fstart = Mfstart / Ms;
    *tstart = tM22start * Ms;
    *phistart = phi22start;

    /* cleanup */
    //XLALDestroydoubleSequence(freqs_geom);
    //XLALDestroyValue(ModeArray);

    // if (pHM->freq_is_uniform == 1)
    // { /* 1. uniformly spaced */
    //     XLALDestroydoubleSequence(phases);
    //     XLALDestroydoubleSequence(amps);
    //     XLALDestroydoubleSequence(freqs);
    // }
    // else if (pHM->freq_is_uniform == 0)
    // { /* 2. arbitrarily space */
        // XLALDestroydoubleSequence(phases);
        // XLALDestroydoubleSequence(amps);
    // }
    // else
    // {
    //     ERROR(ERROR_EDOM, "freq_is_uniform = %i and should be either 0 or 1.", pHM->freq_is_uniform);
    // }

    // LALFree(pHM);
    free(pHM);

  /* If extraParams was allocated in this function and not passed in
   * we need to free it to prevent a leak */
  // if (extraParams && !extraParams_in) {
  //   XLALDestroyDict(extraParams);
  // }

    return SUCCESS;
}

/**
 * Function to compute the one hlm mode.
 * Note this is not static so that IMRPhenomPv3HM
 * can also use this function
 */
int IMRPhenomHMEvaluateOnehlmMode(
    AmpPhaseFDMode* hlm,        /**< [out] One hlm mode */
    real_vector* amps,           /**< amplitude frequency sequence */
    real_vector* phases,         /**< phase frequency sequence */
    real_vector* tfs,            /**< tf frequency sequence */
    real_vector* freqs,          /**< frequency sequence (Hz) */
    PhenomHMStorage* pHM,        /**< PhenomHMStorage struct */
    PhenDAmpAndPhasePreComp* pDPreComp, /**< PhenDAmpAndPhasePreComp struct */
    int ell,                     /**< ell spherical harmonic number */
    int mm,                      /**< m spherical harmonic number */
    double time_shift,           /**< Time (t/M) shift needed to align tf to 0 at f_ref, does not include Deltat. */
    double phi_shift,            /**< Phase shift needed to align waveform to phiRef at f_ref. */
    //double phi0                  /**< phase shift needed to align waveform to phiRef at f_ref. */
    const ModGRParams* modgrparams  /**< Modified GR parameters */
    //LALDict *extraParams       /**< LALDict struct */
)
{
    int retcode;

    /* Geometric frequencies */
    real_vector* freqs_geom = real_vector_alloc(freqs->size);
    for (size_t i = 0; i < freqs->size; i++)
    {
        freqs_geom->data[i] = freqs->data[i] * pHM->Ms;
    }

    /* generate phase */
    retcode = 0;
    retcode = IMRPhenomHMPhase(
        phases,
        tfs,
        freqs_geom,
        pHM,
        pDPreComp,
        ell, mm,
        modgrparams);
        //extraParams);
    CHECK(SUCCESS == retcode,
               ERROR_EFUNC, "IMRPhenomHMPhase failed");

    /* generate amplitude */
    retcode = 0;
    retcode = IMRPhenomHMAmplitude(
        amps,
        freqs_geom,
        pHM,
        pDPreComp,
        ell, mm);
        //extraParams);
    CHECK(SUCCESS == retcode,
               ERROR_EFUNC, "IMRPhenomHMAmplitude failed");

    // /* compute time shift */
    // double t0 = IMRPhenomDComputet0(
    //     pHM->eta, pHM->chi1z, pHM->chi2z,
    //     pHM->finspin); // extraParams);

    double phase_term1 = 0.0;
    double phase_term2 = 0.0;
    double tf_val = 0.0;
    double Mf = 0.0; /* geometric frequency */
    /* combine together to make hlm */
    for (size_t i = 0; i < freqs_geom->size; i++)
    {
        hlm->freq_amp->data[i] = freqs->data[i];
        hlm->freq_phase->data[i] = freqs->data[i];
    }
    //loop over hlm COMPLEX16FrequencySeries
    for (size_t i = 0; i < freqs_geom->size; i++)
    {
        Mf = freqs_geom->data[i];
        phase_term1 = 2*PI * time_shift * (Mf - pHM->Mf_ref);
        phase_term2 = phases->data[i] + (mm * phi_shift);
        /* Note: tfs and time_shift is in t/M units, tf in physical units */
        tf_val = (tfs->data[i] + time_shift) * pHM->Ms;
        // ((*hlm)->data)[i] = amps->data[i] * cexp(-I * (phase_term1 + phase_term2));
        hlm->amp_real->data[i] = pHM->amp0 * amps->data[i];
        hlm->amp_imag->data[i] = 0.;
        /* NOTE: sign change of phase wrt LAL, related to Fourier convention */
        //hlm->phase->data[i] = -(phase_term1 + phase_term2);
        hlm->phase->data[i] = phase_term1 + phase_term2;
        /* tf values */
        hlm->tf->data[i] = tf_val;
    }

    /* cleanup */
    real_vector_free(freqs_geom);

    return SUCCESS;
}

/**
 * returns IMRPhenomHM amplitude evaluated at a set of input frequencies
 * for the l,m mode
 */
int IMRPhenomHMAmplitude(
    real_vector* amps,       /**< [out] amplitude frequency sequence */
    real_vector* freqs_geom, /**< dimensionless frequency sequence */
    PhenomHMStorage* pHM,    /**< PhenomHMStorage struct */
    PhenDAmpAndPhasePreComp* pDPreComp, /**< PhenDAmpAndPhasePreComp struct */
    int ell,                 /**< ell spherical harmonic number */
    int mm                   /**< m spherical harmonic number */
    //LALDict *extraParams   /**< LALDict struct */
)
{
    int retcode;

    /* scale input frequencies according to PhenomHM model */
    /* LL: Map the input domain (frequencies) for this ell mm multipole
    to those appropirate for the ell=|mm| multipole */
    //doubleSequence *freqs_amp = XLALCreatedoubleSequence(freqs_geom->length);
    real_vector* freqs_amp = real_vector_alloc(freqs_geom->size);
    for (int i = 0; i < (int)freqs_amp->size; i++)
    {
        freqs_amp->data[i] = IMRPhenomHMFreqDomainMap(
            freqs_geom->data[i], ell, mm, pHM, AmpFlagTrue);
    }

    /* LL: Compute the PhenomD Amplitude at the mapped l=m=2 fequencies */
    retcode = 0;
    retcode = IMRPhenomDAmpFrequencySequence(
        amps,
        freqs_amp,
        &(pDPreComp->pAmp),
        &(pDPreComp->amp_prefactors),
        // pHM->ind_min, pHM->ind_max,
        0, freqs_geom->size,
        pHM->m1, pHM->m2,
        pHM->chi1z, pHM->chi2z);
    CHECK(SUCCESS == retcode,
               ERROR_EFUNC, "IMRPhenomDAmpFrequencySequence failed");

    /*
    LL: Here we map the ampliude's range using two steps:
    (1) We divide by the leading order l=m=2 behavior, and then
    scale in the expected PN behavior for the multipole of interest.
    NOTE that this step is done at the mapped frequencies,
    which results in smooth behavior despite the sharp featured of the domain map.
    There are other (perhaps more intuitive) options for mapping the amplitudes,
    but these do not have the desired smooth features.
    (2) An additional scaling is needed to recover the desired PN ampitude.
    This is needed becuase only frequencies appropriate for the dominant
    quadrupole have been used thusly, so the current answer does not
    conform to PN expectations for inspiral.
    This is trikier than described here, so please give it a deeper think.
    */

    retcode = 0;
    retcode = IMRPhenomHMAmpPNScaling(
        amps,
        freqs_amp,
        freqs_geom,
        pHM->m1, pHM->m2,
        pHM->chi1z, pHM->chi2z,
        ell, mm);
    CHECK(SUCCESS == retcode,
               ERROR_EFUNC, "IMRPhenomHMAmpPNScaling failed");

    // // int status_in_for = SUCCESS;
    // for (int i = 0; i < (int)freqs_amp->size; i++)
    // {
    //     // PhenomHMUsefulPowers powers_of_freq_amp;
    //     // status_in_for = PhenomHM_init_useful_powers(
    //     //     &powers_of_freq_amp, freqs_amp->data[i]);
    //     // if (SUCCESS != status_in_for)
    //     // {
    //     //     ERROR(ERROR_EDOM, "PhenomHM_init_useful_powers failed for Mf\n");
    //     // }
    //     //new
    //
    //     /* LL: Calculate the corrective factor for step #2 */
    //
    //     double beta_term1 = IMRPhenomHMOnePointFiveSpinPN(
    //         freqs_geom->data[i],
    //         ell,
    //         mm,
    //         pHM->m1,
    //         pHM->m2,
    //         pHM->chi1z,
    //         pHM->chi2z);
    //
    //     double beta=0.;
    //     double beta_term2=0.;
    //     double HMamp_term1=1.;
    //     double HMamp_term2=1.;
    //     //HACK to fix equal black hole case producing NaNs.
    //     //More elegant solution needed.
    //     if (beta_term1 == 0.){
    //         beta = 0.;
    //     } else {
    //
    //         beta_term2 = IMRPhenomHMOnePointFiveSpinPN(2.0 * freqs_geom->data[i] / mm, ell, mm, pHM->m1, pHM->m2, pHM->chi1z, pHM->chi2z);
    //         beta = beta_term1 / beta_term2;
    //
    //         /* LL: Apply steps #1 and #2 */
    //         HMamp_term1 = IMRPhenomHMOnePointFiveSpinPN(
    //             freqs_amp->data[i],
    //             ell,
    //             mm,
    //             pHM->m1,
    //             pHM->m2,
    //             pHM->chi1z,
    //             pHM->chi2z);
    //         HMamp_term2 = IMRPhenomHMOnePointFiveSpinPN(freqs_amp->data[i], 2, 2, pHM->m1, pHM->m2, 0.0, 0.0);
    //
    //     }
    //
    //     //HMamp is computed here
    //     // amps->data[i] *= beta * HMamp_term1 / HMamp_term2;
    // }

    /* cleanup */
    //XLALDestroydoubleSequence(freqs_amp);
    real_vector_free(freqs_amp);

    return SUCCESS;
}
/**
 * returns IMRPhenomHM phase evaluated at a set of input frequencies
 * for the l,m mode
 */
int IMRPhenomHMPhase(
    real_vector* phases,     /**< [out] phase frequency sequence */
    real_vector* tfs,        /**< [out] tf frequency sequence (t/M) */
    real_vector* freqs_geom, /**< dimensionless frequency sequence */
    PhenomHMStorage* pHM,    /**< PhenomHMStorage struct */
    PhenDAmpAndPhasePreComp* pDPreComp, /**< PhenDAmpAndPhasePreComp struct */
    int ell,                 /**< ell spherical harmonic number */
    int mm,                  /**< m spherical harmonic number */
    const ModGRParams* modgrparams  /**< Modified GR parameters */
    //LALDict *extraParams   /**< LALDict struct */
)
{
    int retcode;

    double Rholm = pHM->Rholm[ell][mm];
    double Taulm = pHM->Taulm[ell][mm];

    retcode = IMRPhenomDUpdatePhaseCoefficients(
        pDPreComp,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        Rholm,
        Taulm);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDUpdatePhaseCoefficients failed");
    }

    HMPhasePreComp q;
    retcode = 0;
    retcode = IMRPhenomHMPhasePreComp(&q, ell, mm, pHM, pDPreComp, modgrparams); //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMPhasePreComp failed\n");
    }

    double inv2pi = 1./(2*PI);
    double Mf_wf = 0.0;
    double Mf = 0.0;
    double Mfr = 0.0;
    double tmpphaseC = 0.0;
    for (int i = 0; i < (int)freqs_geom->size; i++)
    {
        /* Add complex phase shift depending on 'm' mode */
        phases->data[i] = cShift[mm];
        Mf_wf = freqs_geom->data[i];
        // This if ladder is in the mathematica function HMPhase. PhenomHMDev.nb
        if (!(Mf_wf > q.fi))
        { /* in mathematica -> IMRPhenDPhaseA */
            Mf = q.ai * Mf_wf + q.bi;
            phases->data[i] += IMRPhenomDPhase_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams) / q.ai;
            tfs->data[i] = inv2pi * IMRPhenomDPhaseDerivative_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams);
        }
        else if (!(Mf_wf > q.fr))
        { /* in mathematica -> IMRPhenDPhaseB */
            Mf = q.am * Mf_wf + q.bm;
            phases->data[i] += IMRPhenomDPhase_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams) / q.am - q.PhDBconst + q.PhDBAterm;
            tfs->data[i] = inv2pi * IMRPhenomDPhaseDerivative_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams);
        }
        else if ((Mf_wf > q.fr))
        { /* in mathematica -> IMRPhenDPhaseC */
            Mfr = q.am * q.fr + q.bm;
            tmpphaseC = IMRPhenomDPhase_OneFrequency(Mfr, *pDPreComp, Rholm, Taulm, modgrparams) / q.am - q.PhDBconst + q.PhDBAterm;
            Mf = q.ar * Mf_wf + q.br;
            phases->data[i] += IMRPhenomDPhase_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams) / q.ar - q.PhDCconst + tmpphaseC;
            tfs->data[i] = inv2pi * IMRPhenomDPhaseDerivative_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams);
        }
        else
        {
            ERROR(ERROR_EDOM, "should not get here - in function IMRPhenomHMPhase");
        }
    }

    return SUCCESS;
}

static double PhenomHMtflm(
    double Mf_lm,
    HMPhasePreComp q,
    PhenDAmpAndPhasePreComp* pDPreComp,
    double Rholm,
    double Taulm,
    double time_shift,
    double DeltatM,
    const ModGRParams* modgrparams
)
{
    double inv2pi = 1./(2*PI);
    double Mf_wf = Mf_lm;
    double Mf = 0.0;
    double Mfr = 0.0;
    // double tmpphaseC = 0.0;
    double tfM_lm_val = 0.;

        /* Add complex phase shift depending on 'm' mode */
        // phases->data[i] = cShift[mm];
        // This if ladder is in the mathematica function HMPhase. PhenomHMDev.nb
        if (!(Mf_wf > q.fi))
        { /* in mathematica -> IMRPhenDPhaseA */
            Mf = q.ai * Mf_wf + q.bi;
            // phases->data[i] += IMRPhenomDPhase_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams) / q.ai;
            tfM_lm_val = inv2pi * IMRPhenomDPhaseDerivative_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams);
        }
        else if (!(Mf_wf > q.fr))
        { /* in mathematica -> IMRPhenDPhaseB */
            Mf = q.am * Mf_wf + q.bm;
            // phases->data[i] += IMRPhenomDPhase_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams) / q.am - q.PhDBconst + q.PhDBAterm;
            tfM_lm_val = inv2pi * IMRPhenomDPhaseDerivative_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams);
        }
        else if ((Mf_wf > q.fr))
        { /* in mathematica -> IMRPhenDPhaseC */
            Mfr = q.am * q.fr + q.bm;
            // tmpphaseC = IMRPhenomDPhase_OneFrequency(Mfr, *pDPreComp, Rholm, Taulm, modgrparams) / q.am - q.PhDBconst + q.PhDBAterm;
            Mf = q.ar * Mf_wf + q.br;
            // phases->data[i] += IMRPhenomDPhase_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams) / q.ar - q.PhDCconst + tmpphaseC;
            tfM_lm_val = inv2pi * IMRPhenomDPhaseDerivative_OneFrequency(Mf, *pDPreComp, Rholm, Taulm, modgrparams);
        }
        else
        {
            ERROR(ERROR_EDOM, "should not get here - in function PhenomHMtflm");
        }
    /* Add time shift coming from alignment, and extra time shift */
    tfM_lm_val += time_shift + DeltatM;

    return tfM_lm_val;
}

/**
 * returns IMRPhenomHM phase derivative over 2pi evaluated at an input frequency
 * for the l,m mode
 */
static int IMRPhenomHMTimeOfFrequencyMode(
    double* tfM_lm,             /**< [out] tf_lm time (t/M) */
    double Mf_lm,              /**< [in] f_lm dimensionless frequency  */
    PhenomHMStorage* pHM,    /**< PhenomHMStorage struct */
    PhenDAmpAndPhasePreComp* pDPreComp, /**< PhenDAmpAndPhasePreComp struct */
    int ell,                 /**< ell spherical harmonic number */
    int mm,                  /**< m spherical harmonic number */
    double time_shift,       /**< Time shift (t/M) coming from alignment at Mfref */
    double DeltatM,          /**< Extra time shift (t/M) after alignment */
    const ModGRParams* modgrparams  /**< Modified GR parameters */
    //LALDict *extraParams   /**< LALDict struct */
)
{
    int retcode;

    double Rholm = pHM->Rholm[ell][mm];
    double Taulm = pHM->Taulm[ell][mm];

    retcode = IMRPhenomDUpdatePhaseCoefficients(
        pDPreComp,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        Rholm,
        Taulm);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDUpdatePhaseCoefficients failed");
    }

    HMPhasePreComp q;
    retcode = 0;
    retcode = IMRPhenomHMPhasePreComp(&q, ell, mm, pHM, pDPreComp, modgrparams); //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMPhasePreComp failed\n");
    }

    double tfM_lm_val = PhenomHMtflm(Mf_lm, q, pDPreComp, Rholm, Taulm, time_shift, DeltatM, modgrparams);

    *tfM_lm = tfM_lm_val;

    return SUCCESS;
}

/**
 * returns IMRPhenomHM frequency corresponding to an input time by numerical inversion of t(f)
 * for the l,m mode
 */
static int IMRPhenomHMInverseFrequencyOfTimeMode(
    double* Mf_lm,             /**< [out] f_lm dimensionless frequency */
    double tfM_lm,              /**< [in] t_lm time (t/M)  */
    double Mf_lm_estimate,      /**< [in] esimate for f_lm dimensionless frequency  */
    const double tM_acc,        /**< Target accuracy of t(f) (t/M) where to stop refining f */
    PhenomHMStorage* pHM,    /**< PhenomHMStorage struct */
    PhenDAmpAndPhasePreComp* pDPreComp, /**< PhenDAmpAndPhasePreComp struct */
    int ell,                 /**< ell spherical harmonic number */
    int mm,                  /**< m spherical harmonic number */
    double time_shift,       /**< Time shift (t/M) coming from alignment at Mfref */
    double DeltatM,          /**< Extra time shift (t/M) after alignment */
    const int max_iter,      /**< Maximal number of iterations in bisection */
    const ModGRParams* modgrparams  /**< Modified GR parameters */
    //LALDict *extraParams   /**< LALDict struct */
)
{
    int retcode;

    double Rholm = pHM->Rholm[ell][mm];
    double Taulm = pHM->Taulm[ell][mm];

    retcode = IMRPhenomDUpdatePhaseCoefficients(
        pDPreComp,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        Rholm,
        Taulm);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDUpdatePhaseCoefficients failed");
    }

    HMPhasePreComp q;
    retcode = 0;
    retcode = IMRPhenomHMPhasePreComp(&q, ell, mm, pHM, pDPreComp, modgrparams); //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomHMPhasePreComp failed\n");
    }

    double tM = tfM_lm;
    double Mf_estimate = Mf_lm_estimate;
    double tM_computed = 0;

    int i = 0;

    double Mflow = Mf_estimate;
    double Mfhigh = Mf_estimate;
    double Mf = 0;

    tM_computed = PhenomHMtflm(Mf_estimate, q, pDPreComp, Rholm, Taulm, time_shift, DeltatM, modgrparams);
    i = 0;
    while ( (i<max_iter) && (tM_computed <= tM) ) {
      Mfhigh = 1.2*Mfhigh;
      tM_computed = PhenomHMtflm(Mfhigh, q, pDPreComp, Rholm, Taulm, time_shift, DeltatM, modgrparams);
      i++;
    }

    tM_computed = PhenomHMtflm(Mf_estimate, q, pDPreComp, Rholm, Taulm, time_shift, DeltatM, modgrparams);
    i = 0;
    while ( (i<max_iter) && (tM_computed >= tM) ) {
      Mflow = 0.8*Mflow;
      tM_computed = PhenomHMtflm(Mflow, q, pDPreComp, Rholm, Taulm, time_shift, DeltatM, modgrparams);
      i++;
    }

    Mf = sqrt(Mflow * Mfhigh);
    tM_computed = PhenomHMtflm(Mf, q, pDPreComp, Rholm, Taulm, time_shift, DeltatM, modgrparams);
    i = 0;
    while ( (i<max_iter) && (fabs(tM_computed - tM) > tM_acc) ) {
      if (tM_computed > tM) {
        Mfhigh = Mf;
      }
      else {
        Mflow = Mf;
      }
      Mf = sqrt(Mflow * Mfhigh);
      tM_computed = PhenomHMtflm(Mf, q, pDPreComp, Rholm, Taulm, time_shift, DeltatM, modgrparams);
      i++;
    }

    CHECKP(i < max_iter, "Reached maximal number of iterations.");
    /* TODO: check that the result does not exceed the merger frequency */

    *Mf_lm = Mf;

    return SUCCESS;
}

/* NOTE: for now ugly input/output by hand for each mode */
int IMRPhenomHMComputeTimeOfFrequencyModeByMode(
    double* tf22,             /**< [out] value of t_22 (s) */
    double* tf21,             /**< [out] value of t_21 (s) */
    double* tf33,             /**< [out] value of t_33 (s) */
    double* tf32,             /**< [out] value of t_32 (s) */
    double* tf44,             /**< [out] value of t_44 (s) */
    double* tf43,             /**< [out] value of t_43 (s) */
    double f22,              /**< [in] value of f_22 (Hz) */
    double f21,              /**< [in] value of f_21 (Hz) */
    double f33,              /**< [in] value of f_33 (Hz) */
    double f32,              /**< [in] value of f_32 (Hz) */
    double f44,              /**< [in] value of f_44 (Hz) */
    double f43,              /**< [in] value of f_43 (Hz) */
    double m1,                   /**< primary mass [solar masses] */
    double m2,                   /**< secondary mass [solar masses] */
    double chi1z,                   /**< aligned spin of primary */
    double chi2z,                   /**< aligned spin of secondary */
    double distance,                /**< luminosity distance (Mpc) */
    //const double deltaF,            /**< frequency spacing */
    const double phiRef,            /**< orbital phase at f_ref */
    const double fRef_in,                   /**< reference GW frequency */
    const double Deltat,             /**< Time shift (s) applied a posteriori */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
    //LALDict *extraParams           /**< LALDict struct */
)
{
    int retcode;

    // Make a pointer to LALDict to circumvent a memory leak
    // At the end we will check if we created a LALDict in extraParams
    // and destroy it if we did.
    //LALDict *extraParams_in = extraParams;

    /* sanity checks on input parameters: check pointers, etc. */

    /* Check inputs for sanity */
    CHECK(m1 > 0, ERROR_EDOM, "m1 must be positive.\n");
    CHECK(m2 > 0, ERROR_EDOM, "m2 must be positive.\n");
    CHECK(fabs(chi1z) <= 1.0, ERROR_EDOM, "Aligned spin chi1z \
must be <= 1 in magnitude!\n");
    CHECK(fabs(chi2z) <= 1.0, ERROR_EDOM, "Aligned spin chi2z \
must be <= 1 in magnitude!\n");
    CHECK(fRef_in >= 0, ERROR_EDOM, "Reference frequency must be \
positive or 0.\n");

    double m1_SI = m1 * MSUN_SI;
    double m2_SI = m2 * MSUN_SI;

    double M = m1 + m2;
    double Ms = M * MTSUN_SI;
    double MfRef_in = Ms * fRef_in;

    /* Time shif tin geometric units */
    double DeltatM = Deltat / Ms;

    /* setup PhenomHM model storage struct / structs */
    /* Compute quantities/parameters related to PhenomD only once and store them */
    PhenomHMStorage *pHM;
    pHM = malloc(sizeof(PhenomHMStorage));
    /* The interface of init_PhenomHM_Storage requires input frequencies */
    /* We create a mock vector with just 2 arbitrary fequencies */
    real_vector* mock_freqs = real_vector_alloc(2);
    mock_freqs->data[0] = 1e-3 / Ms;
    mock_freqs->data[1] = 1e-1 / Ms;
    retcode = 0;
    retcode = init_PhenomHM_Storage(
        pHM,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        mock_freqs,
        //deltaF,
        fRef_in,
        phiRef);
    CHECK(SUCCESS == retcode, ERROR_EFUNC, "init_PhenomHM_Storage \
failed");

    PhenDAmpAndPhasePreComp pDPreComp22;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreComp22,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm[2][2],
        pHM->Taulm[2][2]);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
    }

    /* Copy amp and phase coefficients for later HM update of pPhi */
    /* TODO: for now, just redo it; we should copy instead */
    PhenDAmpAndPhasePreComp pDPreCompHM;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreCompHM,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm[2][2],
        pHM->Taulm[2][2]);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
    }

    /* Reference frequency for the phase (and not time) -- if 0, use fpeak */
    /* Mfpeak: approximate 'peak' frequency for Phenom waveforms */
    double Mfpeak = pDPreComp22.pAmp.fmaxCalc;
    double MfRef = (MfRef_in == 0.0) ? Mfpeak : MfRef_in;
    pHM->Mf_ref = MfRef;

    // /* compute the reference phase shift need to align the waveform so that
    //  the phase is equal to phiRef at the reference frequency f_ref. */
    // /* the phase shift is computed by evaluating the phase of the
    // (l,m)=(2,2) mode.
    // phi0 is the correction we need to add to each mode. */
    // double phi_22_at_Mfref = IMRPhenomDPhase_OneFrequency(pHM->Mf_ref, pDPreComp22,
    //                                                      1.0, 1.0, modgrparams);
    // //double phi0 = 0.5 * phi_22_at_f_ref + phiRef;
    // double phi_shift = - 1./2 * phi_22_at_Mfref - phiRef;

    /* Compute the reference time shift (t/M) to enforce tf=Deltat at fpeak */
    /* Option to reproduce buggy time alignment in LAL PhenomD/HM */
    double tM_22_at_Mfpeak = 0.;
    if ((extraparams==NULL) || (!(extraparams->use_buggy_LAL_tpeak))) {
      tM_22_at_Mfpeak = 1./(2*PI) * IMRPhenomDPhaseDerivative_OneFrequency(
                                    Mfpeak, pDPreComp22, 1.0, 1.0, modgrparams);
    }
    else {
      tM_22_at_Mfpeak = 1./(2*PI) * DPhiMRD(Mfpeak, &(pDPreComp22.pPhi), 1., 1.);
    }

    /* Note: this is a time shift applied as 2pitime_shift*(Mf-Mfref) */
    /* Does not affect phi=-2phiref at fref, and does not include Deltat */
    /* Deltat is to be added as 2piDeltat*f, regardless of alignment at fref */
    double time_shift = -tM_22_at_Mfpeak;

    /* Vectors for tf_lm and f_lm */
    /* BEWARE: loop mode order: 21, 22, 32, 33, 43, 44 */
    real_vector* tf_lm = real_vector_alloc(6);
    real_vector* f_lm = real_vector_alloc(6);
    f_lm->data[0] = f21;
    f_lm->data[1] = f22;
    f_lm->data[2] = f32;
    f_lm->data[3] = f33;
    f_lm->data[4] = f43;
    f_lm->data[5] = f44;

    /* loop over modes */
    int mode_counter = 0;
    double tfM_lm_val = 0., Mf_lm_val = 0.;
    for (int ell = 2; ell < L_MAX_PLUS_1; ell++)
    {
        for (int mm = 1; mm < (int)ell + 1; mm++)
        { /* BEWARE: the loop goes through the modes in that order: [21, 22, 32, 33, 43, 44] */
          /* BEWARE: the output indices are harcoded for that order, be careful if you modify this ! */

            if (!((ell==2 && mm==2) || (ell==2 && mm==1) || (ell==3 && mm==3) || (ell==3 && mm==2) || (ell==4 && mm==4) || (ell==4 && mm==3))) continue;

            Mf_lm_val = Ms * f_lm->data[mode_counter];
            retcode = 0;
            retcode = IMRPhenomHMTimeOfFrequencyMode(
                &tfM_lm_val,
                Mf_lm_val,
                pHM,
                &pDPreCompHM,
                ell, mm,
                time_shift,
                DeltatM,
                modgrparams);
            CHECK(SUCCESS == retcode,
                  ERROR_EFUNC, "IMRPhenomHMTimeOfFrequencyMode failed");
            tf_lm->data[mode_counter] = Ms * tfM_lm_val;
            mode_counter++;
        }
    }

    /* BEWARE: loop mode order: 21, 22, 32, 33, 43, 44 */
    *tf21 = tf_lm->data[0];
    *tf22 = tf_lm->data[1];
    *tf32 = tf_lm->data[2];
    *tf33 = tf_lm->data[3];
    *tf43 = tf_lm->data[4];
    *tf44 = tf_lm->data[5];

    /* Cleanup */
    real_vector_free(mock_freqs);
    real_vector_free(tf_lm);
    real_vector_free(f_lm);
    free(pHM);

    return SUCCESS;
}

/* NOTE: for now ugly input/output by hand for each mode */
int IMRPhenomHMComputeInverseFrequencyOfTimeModeByMode(
    double* f22,              /**< [out] value of f_22 (Hz) */
    double* f21,              /**< [out] value of f_21 (Hz) */
    double* f33,              /**< [out] value of f_33 (Hz) */
    double* f32,              /**< [out] value of f_32 (Hz) */
    double* f44,              /**< [out] value of f_44 (Hz) */
    double* f43,              /**< [out] value of f_43 (Hz) */
    double tf22,             /**< [in] value of t_22 (s) */
    double tf21,             /**< [in] value of t_21 (s) */
    double tf33,             /**< [in] value of t_33 (s) */
    double tf32,             /**< [in] value of t_32 (s) */
    double tf44,             /**< [in] value of t_44 (s) */
    double tf43,             /**< [in] value of t_43 (s) */
    double f22_estimate,        /**< [in] guess for the value of f22, will be scaled by m/2 */
    double t_acc,                 /**< Target accuracy of t(f) where to stop refining f */
    double m1,                   /**< primary mass [solar masses] */
    double m2,                   /**< secondary mass [solar masses] */
    double chi1z,                   /**< aligned spin of primary */
    double chi2z,                   /**< aligned spin of secondary */
    double distance,                /**< luminosity distance (Mpc) */
    //const double deltaF,            /**< frequency spacing */
    const double phiRef,            /**< orbital phase at f_ref */
    const double fRef_in,                   /**< reference GW frequency */
    const double Deltat,             /**< Time shift (s) applied a posteriori */
    const int max_iter,                 /**< Maximal number of iterations in bisection */
    const ExtraParams* extraparams,           /**< Additional parameters */
    const ModGRParams* modgrparams            /**< Modified GR parameters */
    //LALDict *extraParams           /**< LALDict struct */
)
{
    int retcode;

    // Make a pointer to LALDict to circumvent a memory leak
    // At the end we will check if we created a LALDict in extraParams
    // and destroy it if we did.
    //LALDict *extraParams_in = extraParams;

    /* sanity checks on input parameters: check pointers, etc. */

    /* Check inputs for sanity */
    CHECK(m1 > 0, ERROR_EDOM, "m1 must be positive.\n");
    CHECK(m2 > 0, ERROR_EDOM, "m2 must be positive.\n");
    CHECK(fabs(chi1z) <= 1.0, ERROR_EDOM, "Aligned spin chi1z \
must be <= 1 in magnitude!\n");
    CHECK(fabs(chi2z) <= 1.0, ERROR_EDOM, "Aligned spin chi2z \
must be <= 1 in magnitude!\n");
    CHECK(fRef_in >= 0, ERROR_EDOM, "Reference frequency must be \
positive or 0.\n");

    double m1_SI = m1 * MSUN_SI;
    double m2_SI = m2 * MSUN_SI;

    double M = m1 + m2;
    double Ms = M * MTSUN_SI;
    double MfRef_in = Ms * fRef_in;

    /* Time shif tin geometric units */
    double DeltatM = Deltat / Ms;

    /* Desired time accuracy for the inversion in geometric units */
    double tM_acc = t_acc / Ms;

    /* setup PhenomHM model storage struct / structs */
    /* Compute quantities/parameters related to PhenomD only once and store them */
    PhenomHMStorage *pHM;
    pHM = malloc(sizeof(PhenomHMStorage));
    /* The interface of init_PhenomHM_Storage requires input frequencies */
    /* We create a mock vector with just 2 arbitrary fequencies */
    real_vector* mock_freqs = real_vector_alloc(2);
    mock_freqs->data[0] = 1e-3 / Ms;
    mock_freqs->data[1] = 1e-1 / Ms;
    retcode = 0;
    retcode = init_PhenomHM_Storage(
        pHM,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        mock_freqs,
        //deltaF,
        fRef_in,
        phiRef);
    CHECK(SUCCESS == retcode, ERROR_EFUNC, "init_PhenomHM_Storage \
failed");

    PhenDAmpAndPhasePreComp pDPreComp22;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreComp22,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm[2][2],
        pHM->Taulm[2][2]);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
    }

    /* Copy amp and phase coefficients for later HM update of pPhi */
    /* TODO: for now, just redo it; we should copy instead */
    PhenDAmpAndPhasePreComp pDPreCompHM;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreCompHM,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm[2][2],
        pHM->Taulm[2][2]);
        //extraParams);
    if (retcode != SUCCESS)
    {
        ERROR(ERROR_EDOM, "IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
    }

    /* Reference frequency for the phase (and not time) -- if 0, use fpeak */
    /* Mfpeak: approximate 'peak' frequency for Phenom waveforms */
    double Mfpeak = pDPreComp22.pAmp.fmaxCalc;
    double MfRef = (MfRef_in == 0.0) ? Mfpeak : MfRef_in;
    pHM->Mf_ref = MfRef;

    // /* compute the reference phase shift need to align the waveform so that
    //  the phase is equal to phiRef at the reference frequency f_ref. */
    // /* the phase shift is computed by evaluating the phase of the
    // (l,m)=(2,2) mode.
    // phi0 is the correction we need to add to each mode. */
    // double phi_22_at_Mfref = IMRPhenomDPhase_OneFrequency(pHM->Mf_ref, pDPreComp22,
    //                                                      1.0, 1.0, modgrparams);
    // //double phi0 = 0.5 * phi_22_at_f_ref + phiRef;
    // double phi_shift = - 1./2 * phi_22_at_Mfref - phiRef;

    /* Compute the reference time shift (t/M) to enforce tf=Deltat at fpeak */
    /* Option to reproduce buggy time alignment in LAL PhenomD/HM */
    double tM_22_at_Mfpeak = 0.;
    if ((extraparams==NULL) || (!(extraparams->use_buggy_LAL_tpeak))) {
      tM_22_at_Mfpeak = 1./(2*PI) * IMRPhenomDPhaseDerivative_OneFrequency(
                                    Mfpeak, pDPreComp22, 1.0, 1.0, modgrparams);
    }
    else {
      tM_22_at_Mfpeak = 1./(2*PI) * DPhiMRD(Mfpeak, &(pDPreComp22.pPhi), 1., 1.);
    }

    /* Note: this is a time shift applied as 2pitime_shift*(Mf-Mfref) */
    /* Does not affect phi=-2phiref at fref, and does not include Deltat */
    /* Deltat is to be added as 2piDeltat*f, regardless of alignment at fref */
    double time_shift = -tM_22_at_Mfpeak;

    /* Vectors for tf_lm and f_lm */
    /* BEWARE: loop mode order: 21, 22, 32, 33, 43, 44 */
    real_vector* tf_lm = real_vector_alloc(6);
    real_vector* f_lm = real_vector_alloc(6);
    tf_lm->data[0] = tf21;
    tf_lm->data[1] = tf22;
    tf_lm->data[2] = tf32;
    tf_lm->data[3] = tf33;
    tf_lm->data[4] = tf43;
    tf_lm->data[5] = tf44;

    /* loop over modes */
    int mode_counter = 0;
    double tfM_lm_val = 0., Mf_lm_val = 0., Mf_lm_guess = 0.;
    for (int ell = 2; ell < L_MAX_PLUS_1; ell++)
    {
        for (int mm = 1; mm < (int)ell + 1; mm++)
        { /* BEWARE: the loop goes through the modes in that order: [21, 22, 32, 33, 43, 44] */
          /* BEWARE: the output indices are harcoded for that order, be careful if you modify this ! */

            if (!((ell==2 && mm==2) || (ell==2 && mm==1) || (ell==3 && mm==3) || (ell==3 && mm==2) || (ell==4 && mm==4) || (ell==4 && mm==3))) continue;

            tfM_lm_val = tf_lm->data[mode_counter] / Ms;
            Mf_lm_guess = Ms * mm/2. * f22_estimate;
            retcode = 0;
            retcode = IMRPhenomHMInverseFrequencyOfTimeMode(
                &Mf_lm_val,
                tfM_lm_val,
                Mf_lm_guess,
                tM_acc,
                pHM,
                &pDPreCompHM,
                ell, mm,
                time_shift,
                DeltatM,
                max_iter,
                modgrparams);
            CHECK(SUCCESS == retcode,
                  ERROR_EFUNC, "IMRPhenomHMInverseFrequencyOfTimeMode failed");
            f_lm->data[mode_counter] = Mf_lm_val / Ms;
            mode_counter++;
        }
    }

    /* BEWARE: loop mode order: 21, 22, 32, 33, 43, 44 */
    *f21 = f_lm->data[0];
    *f22 = f_lm->data[1];
    *f32 = f_lm->data[2];
    *f33 = f_lm->data[3];
    *f43 = f_lm->data[4];
    *f44 = f_lm->data[5];

    /* Cleanup */
    real_vector_free(mock_freqs);
    real_vector_free(tf_lm);
    real_vector_free(f_lm);
    free(pHM);

    return SUCCESS;
}
