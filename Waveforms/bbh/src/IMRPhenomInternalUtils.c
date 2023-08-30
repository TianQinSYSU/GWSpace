/*
 *  Copyright (C) 2017 Sebastian Khan
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

/**
 * \author Sebastian Khan
 *
 * \file
 *
 * \brief Internal (not SWIG'd) Auxiliary functions for phenomenological model development
 *
 * Helper functions for phenomenological waveform models
 * Cannot be used through python SWIG wrapping.
 * NOTE: The convention for naming functions in there is to use
 * the prefix 'PhenomInternal_'
 * This is to avoid the use of defining functions as static and
 * therefore not able to be used by other modules by including
 * LALSimIMRPhenomInternalUtils.h.
 */


#include "IMRPhenomInternalUtils.h"

// /**
//  * Example how to write an internal phenom function
//  */
// void PhenomInternal_UtilsTest(){
//     printf("Hello! I am the PhenomInternal_UtilsTest function\n");
// }

// This function determines whether x and y are approximately equal to a relative accuracy epsilon.
// Note that x and y are compared to relative accuracy, so this function is not suitable for testing whether a value is approximately zero.

/**
 * This function determines whether x and y are approximately equal to a
 * relative accuracy epsilon.
 * Note that x and y are compared to relative accuracy, so this function is not
 * suitable for testing whether a value is approximately zero.
 */
bool PhenomInternal_approx_equal(double x, double y, double epsilon)
{
    return !gsl_fcmp(x, y, epsilon);
}

/**
 * If x and X are approximately equal to relative accuracy epsilon
 * then set x = X.
 * If X = 0 then use an absolute comparison.
 */
void PhenomInternal_nudge(double *x, double X, double epsilon)
{
    if (X != 0.0)
    {
        if (PhenomInternal_approx_equal(*x, X, epsilon))
        {
            INFO("Nudging value %.15g to %.15g\n", *x, X);
            *x = X;
        }
    }
    else
    {
        if (fabs(*x - X) < epsilon)
            *x = X;
    }
}

/**
 * Return the closest higher power of 2
 */
size_t PhenomInternal_NextPow2(const size_t n)
{
    // use pow here, not bit-wise shift, as the latter seems to run against an upper cutoff long before SIZE_MAX, at least on some platforms
    return (size_t)pow(2, ceil(log2(n)));
}

/**
 * Given m1 with aligned-spin chi1z and m2 with aligned-spin chi2z.
 * Enforce that m1 >= m2 and swap spins accordingly.
 * Enforce that the primary object (heavier) is indexed by 1.
 * To be used with aligned-spin waveform models.
 * TODO: There is another function for precessing waveform models
 */
// int PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
//     double *m1,    /**< [out] mass of body 1 */
//     double *m2,    /**< [out] mass of body 2 */
//     double *chi1z, /**< [out] aligned-spin component of body 1 */
//     double *chi2z  /**< [out] aligned-spin component of body 2 */
// )
// {
//     double chi1z_tmp, chi2z_tmp, m1_tmp, m2_tmp;
//     if (*m1 > *m2)
//     {
//         chi1z_tmp = *chi1z;
//         chi2z_tmp = *chi2z;
//         m1_tmp = *m1;
//         m2_tmp = *m2;
//     }
//     else
//     { /* swap spins and masses */
//         chi1z_tmp = *chi2z;
//         chi2z_tmp = *chi1z;
//         m1_tmp = *m2;
//         m2_tmp = *m1;
//     }
//     *m1 = m1_tmp;
//     *m2 = m2_tmp;
//     *chi1z = chi1z_tmp;
//     *chi2z = chi2z_tmp;
//
//     if (*m1 < *m2)
//         ERORR(ERROR_EDOM, "ERORR in EnforcePrimaryIsm1. When trying\
//  to enfore that m1 should be the larger mass.\
//  After trying to enforce this m1 = %f and m2 = %f\n", *m1, *m2);
//
//     return SUCCESS;
// }
