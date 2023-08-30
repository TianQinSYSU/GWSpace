/*
* Copyright (C) 2016 Lionel London
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

/* .................... */
/* HEADER SECTION       */
/* .................... */
// #ifdef __GNUC__
// #define UNUSED __attribute__((unused))
// #else
// #define UNUSED
// #endif


#include "RingdownCW.h"

/*
* Based on the paper by London and Fauchon-Jones: https://arxiv.org/abs/1810.03550
* Basic NOTE(s):
*   - This file contains a function, CW07102016, which outputs complex valued, UNITLESS, QNM frequencies (i.e. Mw) for various QNMs
*   - Usage: cw = CW07102016( kappa, l, m, n ); where cw = Mw + 1i*M/tau; NOTE that kappa is a function of final spin, l and m
*   - See definition of KAPPA below.
*/

/*
* -------------------------------------------------------------------------------- *
* Low level models: QNM Frequencies
* -------------------------------------------------------------------------------- *
*/

/*
* Domain mapping for dimnesionless BH spin
*/
double RingdownCW_KAPPA(double jf, int l, int m)
{
    /* */
    /* if ( jf > 1.0 ) XLAL_ERROR(XLAL_EDOM, "Spin (dimensionless Kerr parameter) must not be greater than 1.0\n"); */
    /**/
    double alpha = log(2.0 - jf) / log(3);
    double beta = 1.0 / (2.0 + l - abs(m));
    return pow(alpha, beta);
}

/*
* Dimensionless QNM Frequencies: Note that name encodes date of writing
*/
/*TODO: Make the function arg comments compatible with doxygen*/
complex double RingdownCW_CW07102016(double kappa, /* Domain mapping for  remnant BH's spin (Dimensionless) */
                                        int l,        /* Polar eigenvalue */
                                        int input_m,  /* Azimuthal eigenvalue*/
                                        int n)
{ /* Overtone Number*/

    /* Predefine powers to increase efficiency*/
    double kappa2 = kappa * kappa;
    double kappa3 = kappa2 * kappa;
    double kappa4 = kappa3 * kappa;

    /* NOTE that |m| will be used to determine the fit to use, and if input_m < 0, then a conjugate will be taken*/
    int m = abs(input_m);

    /**/
    complex double j = _Complex_I;

    /* Initialize the answer*/
    double complex ans;

    /* Use If-Else ladder to determine which mode function to evaluate*/
    if (2 == l && 2 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (2,2,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.0 + kappa * (1.557847 * cexp(2.903124 * j) +
                             1.95097051 * cexp(5.920970 * j) * kappa +
                             2.09971716 * cexp(2.760585 * j) * kappa2 +
                             1.41094660 * cexp(5.914340 * j) * kappa3 +
                             0.41063923 * cexp(2.795235 * j) * kappa4);
    }
    else if (2 == l && 2 == m && 1 == n)
    {

        /* Fit for (l,m,n) == (2,2,1). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.0 + kappa * (1.870939 * cexp(2.511247 * j) +
                             2.71924916 * cexp(5.424999 * j) * kappa +
                             3.05648030 * cexp(2.285698 * j) * kappa2 +
                             2.05309677 * cexp(5.486202 * j) * kappa3 +
                             0.59549897 * cexp(2.422525 * j) * kappa4);
    }
    else if (3 == l && 2 == m && 0 == n)
    {

        /* Define extra powers as needed*/
        double kappa5 = kappa4 * kappa;
        double kappa6 = kappa5 * kappa;

        /* Fit for (l,m,n) == (3,2,0). This is NOT a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.022464 * cexp(0.004870 * j) +
              0.24731213 * cexp(0.665292 * j) * kappa +
              1.70468239 * cexp(3.138283 * j) * kappa2 +
              0.94604882 * cexp(0.163247 * j) * kappa3 +
              1.53189884 * cexp(5.703573 * j) * kappa4 +
              2.28052668 * cexp(2.685231 * j) * kappa5 +
              0.92150314 * cexp(5.841704 * j) * kappa6;
    }
    else if (4 == l && 4 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (4,4,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 2.0 + kappa * (2.658908 * cexp(3.002787 * j) +
                             2.97825567 * cexp(6.050955 * j) * kappa +
                             3.21842350 * cexp(2.877514 * j) * kappa2 +
                             2.12764967 * cexp(5.989669 * j) * kappa3 +
                             0.60338186 * cexp(2.830031 * j) * kappa4);
    }
    else if (2 == l && 1 == m && 0 == n)
    {

        /* Define extra powers as needed*/
        double kappa5 = kappa4 * kappa;
        double kappa6 = kappa5 * kappa;

        /* Fit for (l,m,n) == (2,1,0). This is NOT a zero-damped mode in the extremal Kerr limit.*/
        ans = 0.589113 * cexp(0.043525 * j) +
              0.18896353 * cexp(2.289868 * j) * kappa +
              1.15012965 * cexp(5.810057 * j) * kappa2 +
              6.04585476 * cexp(2.741967 * j) * kappa3 +
              11.12627777 * cexp(5.844130 * j) * kappa4 +
              9.34711461 * cexp(2.669372 * j) * kappa5 +
              3.03838318 * cexp(5.791518 * j) * kappa6;
    }
    else if (3 == l && 3 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (3,3,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (2.095657 * cexp(2.964973 * j) +
                             2.46964352 * cexp(5.996734 * j) * kappa +
                             2.66552551 * cexp(2.817591 * j) * kappa2 +
                             1.75836443 * cexp(5.932693 * j) * kappa3 +
                             0.49905688 * cexp(2.781658 * j) * kappa4);
    }
    else if (3 == l && 3 == m && 1 == n)
    {

        /* Fit for (l,m,n) == (3,3,1). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (2.339070 * cexp(2.649692 * j) +
                             3.13988786 * cexp(5.552467 * j) * kappa +
                             3.59156756 * cexp(2.347192 * j) * kappa2 +
                             2.44895997 * cexp(5.443504 * j) * kappa3 +
                             0.70040804 * cexp(2.283046 * j) * kappa4);
    }
    else if (4 == l && 3 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (4,3,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (0.205046 * cexp(0.595328 * j) +
                             3.10333396 * cexp(3.016200 * j) * kappa +
                             4.23612166 * cexp(6.038842 * j) * kappa2 +
                             3.02890198 * cexp(2.826239 * j) * kappa3 +
                             0.90843949 * cexp(5.915164 * j) * kappa4);
    }
    else if (5 == l && 5 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (5,5,0). This is a zero-damped mode in the extremal Kerr limit. */
        ans = 2.5 + kappa * (3.240455 * cexp(3.027869 * j) +
                             3.49056455 * cexp(6.088814 * j) * kappa +
                             3.74704093 * cexp(2.921153 * j) * kappa2 +
                             2.47252790 * cexp(6.036510 * j) * kappa3 +
                             0.69936568 * cexp(2.876564 * j) * kappa4);
    }
    else
    {

        /**/
        ans = 0.0;

    } /* END of IF-ELSE Train for QNM cases */

    /* If m<0, then take the *Negative* conjugate */
    if (input_m < 0)
    {
        /**/
        ans = -conj(ans);
    }

    return ans;

} /* END of CW07102016 */
