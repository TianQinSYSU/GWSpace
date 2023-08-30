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
 * \brief External (SWIG'd) Auxiliary functions for phenomenological model development
 *
 * Helper functions for phenomenological waveform models
 * Can be used through python SWIG wrapping
 * NOTE: The convention for naming functions in there is to use
 * the prefix 'XLALSimPhenom_'
 */

#include "IMRPhenomUtils.h"
#include "IMRPhenomInternalUtils.h"

// /**
//  * Example how to write an external XLAL phenom function
//  */
// void XLALSimPhenomUtilsTest(){
//     printf("Hello! I am the XLALSimPhenomUtilsTest function\n");
// }

/**
 * Convert from geometric frequency to frequency in Hz
 */
double PhenomUtilsMftoHz(
    double Mf,       /**< Geometric frequency */
    double Mtot_Msun /**< Total mass in solar masses */
)
{
    return Mf / (MTSUN_SI * Mtot_Msun);
}

/**
 * Convert from frequency in Hz to geometric frequency
 */
double PhenomUtilsHztoMf(
    double fHz,      /**< Frequency in Hz */
    double Mtot_Msun /**< Total mass in solar masses */
)
{
    return fHz * (MTSUN_SI * Mtot_Msun);
}

/**
 * compute the frequency domain amplitude pre-factor
 */
double PhenomUtilsFDamp0(
    double Mtot_Msun, /**< total mass in solar masses */
    double distance   /**< distance (m) */
)
{
    return Mtot_Msun * MRSUN_SI * Mtot_Msun * MTSUN_SI / distance;
}
