/*  Copyright (C) 2014 E. A. Huerta, Prayush Kumar
 *                2022 Han Wang
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
 *  Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 * This code describes the gravitational radiation emitted by compact binaries with low to moderate
 * values of eccentricity [0 < e < 0.5]. The waveform phase includes post-Newtonian corrections
 * up to 3.5 order and eccentricity corrections up to order e^8 at each post-Newtonian order.
 * The waveform amplitude is modeled within the restricted post-Newtonian approach.
 */


#include "InspiralOptimizedCoefficientsEccentricityFD.c"
#include "InspiralEccentricFD.h"

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif


typedef struct
tagexpnCoeffsEPC {

    /*Input params*/
    //double f0, e0;

    /* Coefficients to reconstruct the phase*/
    double a0, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11, a12,a13,a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40;

    /*Coefficients to reconstruct z_real*/
    /*plus component*/
    double b1p, b2p, b3p, b4p, b5p, b6p, b7p, b8p, b9p, b10p, b11p, b12p, b13p, b14p, b15p, b16p, b17p, b18p, b19p, b20p, b21p, b22p, b23p, b24p, b25p, b26p, b27p, b28p, b29p;
    /*cross component*/
    double b1c, b2c, b3c, b4c, b5c, b6c, b7c, b8c, b9c, b10c, b11c, b12c, b13c, b14c, b15c, b16c, b17c, b18c, b19c, b20c, b21c, b22c, b23c, b24c, b25c, b26c, b27c, b28c, b29c;

    /*Coefficients to reconstruct z_imaginary*/
    /*plus component*/
    double d1p, d2p, d3p, d4p, d5p, d6p, d7p, d8p, d9p, d10p, d11p, d12p, d13p, d14p, d15p, d16p, d17p, d18p, d19p, d20p, d21p, d22p, d23p, d24p, d25p, d26p, d27p, d28p, d29p;
    /*cross component*/
    double d1c, d2c, d3c, d4c, d5c, d6c, d7c, d8c, d9c, d10c, d11c, d12c, d13c, d14c, d15c, d16c, d17c, d18c, d19c, d20c, d21c, d22c, d23c, d24c, d25c, d26c, d27c, d28c, d29c;

    /* symmetric mass ratio, total mass, component masses*/
    //double nu,m,m1,m2,mu,Mtotal;

}expnCoeffsEPC;

static double PhaseAccTaylorF2(int k, double f, expnCoeffsEPC *ak){
    //Note: 1408.3406, Appendix A
    double k4, k5, k19, k19ov3, k19ov9, k19ov18, f1, f2, f3, f4, f8, fact, lo1, lo2, lo3, lo4,lfact;
    k4=f*f*f*f;
    k5=f*f*f*f*f;
    k19=k5*k5*k5*k4;
    k19ov3=cbrt(k19);
    k19ov9=cbrt(k19ov3);
    k19ov18=sqrt(k19ov9);
    f8=1./(k19ov18);
    f1=f8*f8;
    f2=f1*f1*f1;
    f3=f1*f1;
    f4=f3*f3;
    fact=ak->a1*f;
    lo1=fact;
    lo4=cbrt(lo1);
    lo3=lo4*lo4;
    lo2=lo3*lo3*lo4;
    lfact=6.*sqrt(6.);

    switch( k )
    {
        case 1: return(ak->a0*(1./lo2)*( 1.+ ak->a2*f1 + ak->a3*f2 + ak->a4*f3 + ak->a5*f4));

        case 2: return(ak->a0*(1./lo1)*(ak->a6 + ak->a7*f1 + ak->a8*f2 + ak->a9*f3 + ak->a10*f4));

        case 3: return(ak->a0*(1./lo3)*(ak->a11 + ak->a12*f1 + ak->a13*f2 + ak->a14*f3 + ak->a15*f4));

        case 4: return(ak->a0*(1./lo4)*(ak->a16 + ak->a17*f1 + ak->a18*f2 + ak->a19*f3 + ak->a20*f4));

        case 5: return(ak->a0*(ak->a21 + ak->a21*log(lfact*lo1) +ak->a22*f1+ak->a23*f2 + ak->a24*f3+ak->a25*f4));

        case 6: return(ak->a0*(lo4)*(ak->a26 + log(4.*lo4)*(ak->a27 + ak->a28*f1 + ak->a29*f2 + ak->a30*f3 + ak->a31*f4) + ak->a32*f1 + ak->a33*f2 + ak->a34*f3 + ak->a35*f4));

        case 7: return(ak->a0*(lo3)*( ak->a36 + ak->a37*f1+ ak->a38*f2+ ak->a39*f3+ak->a40*f4) );

        default: return 0;

    }
}

//Note: 0906.0313 Appendix B, Eq 3.11, Eq 4.20, Eq 4.31
static double zeta_generic_re_plus(int k, double f, expnCoeffsEPC *ak) {
    double k4, k5, k19, k19ov3, k19ov9, k19ov18, f1,f2,f3,f4,f5,f6,f7,f8;
    k4=f*f*f*f;
    k5=f*f*f*f*f;
    k19=k5*k5*k5*k4;
    k19ov3=cbrt(k19);
    k19ov9=cbrt(k19ov3);
    k19ov18=sqrt(k19ov9);
    f8=1./(k19ov18);
    f1=f8*f8;
    f2=f1*f1*f1;
    f3=f1*f1;
    f4=f3*f3;
    f5=f2*f8;
    f6=f3*f8;
    f7=f1*f8;

    switch( k )
    {
        case 1: return(ak->b1p*f5 + ak->b2p*f6 + ak->b3p*f7 + ak->b4p*f8);

        case 2: return(ak->b5p + ak->b6p*f1 + ak->b7p*f2 + ak->b8p*f3 + ak->b9p*f4);

        case 3: return(ak->b10p*f5 + ak->b11p*f6 + ak->b12p*f7 + ak->b13p*f8);

        case 4: return(ak->b14p*f1 + ak->b15p*f2 + ak->b16p*f3 + ak->b17p*f4);

        case 5: return(ak->b18p*f5 + ak->b19p*f6 + ak->b20p*f7);

        case 6: return(ak->b21p*f2 + ak->b22p*f3 + ak->b23p*f4);

        case 7: return(ak->b24p*f5 + ak->b25p*f6);

        case 8: return(ak->b26p*f2 + ak->b27p*f4);

        case 9: return(ak->b28p*f5);

        case 10: return(ak->b29p*f4);

        default: return 0;

    }
}

static double zeta_generic_re_cross(int k, double f, expnCoeffsEPC *ak) {
    double k4, k5, k19, k19ov3, k19ov9, k19ov18, f1,f2,f3,f4,f5,f6,f7,f8;
    k4=f*f*f*f;
    k5=f*f*f*f*f;
    k19=k5*k5*k5*k4;
    k19ov3=cbrt(k19);
    k19ov9=cbrt(k19ov3);
    k19ov18=sqrt(k19ov9);
    f8=1./(k19ov18);
    f1=f8*f8;
    f2=f1*f1*f1;
    f3=f1*f1;
    f4=f3*f3;
    f5=f2*f8;
    f6=f3*f8;
    f7=f1*f8;

    switch( k )
    {
        case 1: return(ak->b1c*f5 + ak->b2c*f6 + ak->b3c*f7 + ak->b4c*f8);

        case 2: return(ak->b5c + ak->b6c*f1 + ak->b7c*f2 + ak->b8c*f3 + ak->b9c*f4);

        case 3: return(ak->b10c*f5 + ak->b11c*f6 + ak->b12c*f7 + ak->b13c*f8);

        case 4: return(ak->b14c*f1 + ak->b15c*f2 + ak->b16c*f3 + ak->b17c*f4);

        case 5: return(ak->b18c*f5 + ak->b19c*f6 + ak->b20c*f7);

        case 6: return(ak->b21c*f2 + ak->b22c*f3 + ak->b23c*f4);

        case 7: return(ak->b24c*f5 + ak->b25c*f6);

        case 8: return(ak->b26c*f2 + ak->b27c*f4);

        case 9: return(ak->b28c*f5);

        case 10: return(ak->b29c*f4);

        default: return 0;

    }
}


static double zeta_generic_im_plus(int k, double f, expnCoeffsEPC *ak) {
    double k4, k5, k19, k19ov3, k19ov9, k19ov18, f1,f2,f3,f4,f5,f6,f7,f8;
    k4=f*f*f*f;
    k5=f*f*f*f*f;
    k19=k5*k5*k5*k4;
    k19ov3=cbrt(k19);
    k19ov9=cbrt(k19ov3);
    k19ov18=sqrt(k19ov9);
    f8=1./(k19ov18);
    f1=f8*f8;
    f2=f1*f1*f1;
    f3=f1*f1;
    f4=f3*f3;
    f5=f2*f8;
    f6=f3*f8;
    f7=f1*f8;

    switch( k )
    {
        case 1: return(ak->d1p*f5 + ak->d2p*f6 + ak->d3p*f7 + ak->d4p*f8);

        case 2: return(ak->d5p + ak->d6p*f1 + ak->d7p*f2 + ak->d8p*f3 + ak->d9p*f4);

        case 3: return(ak->d10p*f5 + ak->d11p*f6 + ak->d12p*f7 + ak->d13p*f8);

        case 4: return(ak->d14p*f1 + ak->d15p*f2 + ak->d16p*f3 + ak->d17p*f4);

        case 5: return(ak->d18p*f5 + ak->d19p*f6 + ak->d20p*f7);

        case 6: return(ak->d21p*f2 + ak->d22p*f3 + ak->d23p*f4);

        case 7: return(ak->d24p*f5 + ak->d25p*f6);

        case 8: return(ak->d26p*f2 + ak->d27p*f4);

        case 9: return(ak->d28p*f5);

        case 10: return(ak->d29p*f4);

        default: return 0;

    }
}

static double zeta_generic_im_cross(int k, double f, expnCoeffsEPC *ak) {
    double k4, k5, k19, k19ov3, k19ov9, k19ov18, f1,f2,f3,f4,f5,f6,f7,f8;
    k4=f*f*f*f;
    k5=f*f*f*f*f;
    k19=k5*k5*k5*k4;
    k19ov3=cbrt(k19);
    k19ov9=cbrt(k19ov3);
    k19ov18=sqrt(k19ov9);
    f8=1./(k19ov18);
    f1=f8*f8;
    f2=f1*f1*f1;
    f3=f1*f1;
    f4=f3*f3;
    f5=f2*f8;
    f6=f3*f8;
    f7=f1*f8;

    switch( k )
    {
        case 1: return(ak->d1c*f5 + ak->d2c*f6 + ak->d3c*f7 + ak->d4c*f8);

        case 2: return(ak->d5c + ak->d6c*f1 + ak->d7c*f2 + ak->d8c*f3 + ak->d9c*f4);

        case 3: return(ak->d10c*f5 + ak->d11c*f6 + ak->d12c*f7 + ak->d13c*f8);

        case 4: return(ak->d14c*f1 + ak->d15c*f2 + ak->d16c*f3 + ak->d17c*f4);

        case 5: return(ak->d18c*f5 + ak->d19c*f6 + ak->d20c*f7);

        case 6: return(ak->d21c*f2 + ak->d22c*f3 + ak->d23c*f4);

        case 7: return(ak->d24c*f5 + ak->d25c*f6);

        case 8: return(ak->d26c*f2 + ak->d27c*f4);

        case 9: return(ak->d28c*f5);

        case 10: return(ak->d29c*f4);

        default: return 0;

    }
}


static int
EPCSetup(
         expnCoeffsEPC *ak,                    /**< coefficients for EPC evolution [modified] */
         const double m1,                      /**< Mass of companion 1 (kg) */
         const double m2,                      /**< Mass of companion 2 (kg) */
         const double fStart,                  /**< Start GW frequency (Hz) */
         const double i,                       /**< Polar inclination of source (rad) */
         const double inclination_azimuth,     /**< Azimuthal component of inclination angles */
         const double e_min                    /**< Initial eccentricity at f_min: range [0, 0.5] */
         )
{

    const double m = m1 + m2;
    const double Mtotal = m * MTSUN_SI;  /* total mass in seconds */
    const double eta = m1 * m2 / (m * m);
    const double e0=e_min;
    const double inc=i;
    const double bet=inclination_azimuth;
    const double f0=fStart;

    /* Coefficients to reconstruct the phase*/

    ak->a0 = C0(eta);                           ak->b1p = z1(e0, f0, inc, bet, 1., 0.);
    ak->a1 = C1(Mtotal);                        ak->b2p = z2(e0, f0, inc, bet, 1., 0.);
    ak->a2 = C2(e0, f0);                        ak->b3p = z3(e0, f0, inc, bet, 1., 0.);
    ak->a3 = C3(e0, f0);                        ak->b4p = z4(e0, f0, inc, bet, 1., 0.);
    ak->a4 = C4(e0, f0);                        ak->b5p = z5(f0, inc, bet, 1., 0.);
    ak->a5 = C5(e0, f0);                        ak->b6p = z6(e0, f0, inc, bet, 1., 0.);
    ak->a6 = C6(eta);                           ak->b7p = z7(e0, f0, inc, bet, 1., 0.);
    ak->a7 = C7(eta, e0, f0);                   ak->b8p = z8(e0, f0, inc, bet, 1., 0.);
    ak->a8 = C8(eta, e0, f0);                   ak->b9p = z9(e0, f0, inc, bet, 1., 0.);
    ak->a9 = C9(eta, e0, f0);                   ak->b10p = z10(e0, f0, inc, bet, 1., 0.);
    ak->a10 = C10(eta, e0, f0);                 ak->b11p = z11(e0, f0, inc, bet, 1., 0.);
    ak->a11 = C11(eta);                         ak->b12p = z12(e0, f0, inc, bet, 1., 0.);
    ak->a12 = C12(eta, e0, f0);                 ak->b13p = z13(e0, f0, inc, bet, 1., 0.);
    ak->a13 = C13(eta, e0, f0);                 ak->b14p = z14(e0, f0, inc, bet, 1., 0.);
    ak->a14 = C14(eta, e0, f0);                 ak->b15p = z15(e0, f0, inc, bet, 1., 0.);
    ak->a15 = C15(eta, e0, f0);                 ak->b16p = z16(e0, f0, inc, bet, 1., 0.);
    ak->a16 = C16(eta);                         ak->b17p = z17(e0, f0, inc, bet, 1., 0.);
    ak->a17 = C17(eta, e0, f0);                 ak->b18p = z18(e0, f0, inc, bet, 1., 0.);
    ak->a18 = C18(eta, e0, f0);                 ak->b19p = z19(e0, f0, inc, bet, 1., 0.);
    ak->a19 = C19(eta, e0, f0);                 ak->b20p = z20(e0, f0, inc, bet, 1., 0.);
    ak->a20 = C20(eta, e0, f0);                 ak->b21p = z21(e0, f0, inc, bet, 1., 0.);
    ak->a21 = C21(eta);                         ak->b22p = z22(e0, f0, inc, bet, 1., 0.);
    ak->a22 = C22(eta, e0, f0);                 ak->b23p = z23(e0, f0, inc, bet, 1., 0.);
    ak->a23 = C23(eta, e0, f0);                 ak->b24p = z24(e0, f0, inc, bet, 1., 0.);
    ak->a24 = C24(eta, e0, f0);                 ak->b25p = z25(e0, f0, inc, bet, 1., 0.);
    ak->a25 = C25(eta, e0, f0);                 ak->b26p = z26(e0, f0, inc, bet, 1., 0.);
    ak->a26 = C26(eta);                         ak->b27p = z27(e0, f0, inc, bet, 1., 0.);
    ak->a27 = C27(eta);                         ak->b28p = z28(e0, f0, inc, bet, 1., 0.);
    ak->a28 = C28(eta, e0, f0);                 ak->b29p = z29(e0, f0, inc, bet, 1., 0.);
    ak->a29 = C29(eta, e0, f0);                 ak->d1p = q1(e0, f0, inc, bet, 1., 0.);
    ak->a30 = C30(eta, e0, f0);                 ak->d2p = q2(e0, f0, inc, bet,1., 0.);
    ak->a31 = C31(eta, e0, f0);                 ak->d3p = q3(e0, f0, inc, bet, 1., 0.);
    ak->a32 = C32(eta, e0, f0);                 ak->d4p = q4(e0, f0, inc, bet, 1., 0.);
    ak->a33 = C33(eta, e0, f0);                 ak->d5p = q5(f0, inc, bet, 1., 0.);
    ak->a34 = C34(eta, e0, f0);                 ak->d6p = q6(e0, f0, inc, bet, 1., 0.);
    ak->a35 = C35(eta, e0, f0);                 ak->d7p = q7(e0, f0, inc, bet, 1., 0.);
    ak->a36 = C36(eta);                         ak->d8p = q8(e0, f0, inc, bet, 1., 0.);
    ak->a37 = C37(eta, e0, f0);                 ak->d9p = q9(e0, f0, inc, bet, 1., 0.);
    ak->a38 = C38(eta, e0, f0);                 ak->d10p = q10(e0, f0, inc, bet, 1., 0.);
    ak->a39 = C39(eta, e0, f0);                 ak->d11p = q11(e0, f0, inc, bet, 1., 0.);
    ak->a40 = C40(eta, e0, f0);                 ak->d12p = q12(e0, f0, inc, bet, 1., 0.);
    ak->d13p = q13(e0, f0, inc, bet, 1., 0.);   ak->d14p = q14(e0, f0, inc, bet, 1., 0.);
    ak->d15p = q15(e0, f0, inc, bet, 1., 0.);   ak->d16p = q16(e0, f0, inc, bet, 1., 0.);
    ak->d17p = q17(e0, f0, inc, bet, 1., 0.);   ak->d18p = q18(e0, f0, inc, bet, 1., 0.);
    ak->d19p = q19(e0, f0, inc, bet, 1., 0.);   ak->d20p = q20(e0, f0, inc, bet, 1., 0.);
    ak->d21p = q21(e0, f0, inc, bet, 1., 0.);   ak->d22p = q22(e0, f0, inc, bet, 1., 0.);
    ak->d23p = q23(e0, f0, inc, bet, 1., 0.);   ak->d24p = q24(e0, f0, inc, bet, 1., 0.);
    ak->d25p = q25(e0, f0, inc, bet, 1., 0.);   ak->d26p = q26(e0, f0, inc, bet, 1., 0.);
    ak->d27p = q27(e0, f0, inc, bet, 1., 0.);   ak->d28p = q28(e0, f0, inc, bet, 1., 0.);
    ak->d29p = q29(e0, f0, inc, bet, 1., 0.);

    ak->b1c = z1(e0, f0, inc, bet, 0., 1.);     ak->b2c = z2(e0, f0, inc, bet, 0., 1.);
    ak->b3c = z3(e0, f0, inc, bet, 0., 1.);     ak->b4c = z4(e0, f0, inc, bet, 0., 1.);
    ak->b5c = z5(f0, inc, bet, 0., 1.);         ak->b6c = z6(e0, f0, inc, bet, 0., 1.);
    ak->b7c = z7(e0, f0, inc, bet, 0., 1.);     ak->b8c = z8(e0, f0, inc, bet, 0., 1.);
    ak->b9c = z9(e0, f0, inc, bet, 0., 1.);     ak->b10c = z10(e0, f0, inc, bet, 0., 1.);
    ak->b11c = z11(e0, f0, inc, bet, 0., 1.);   ak->b12c = z12(e0, f0, inc, bet, 0., 1.);
    ak->b13c = z13(e0, f0, inc, bet, 0., 1.);   ak->b14c = z14(e0, f0, inc, bet, 0., 1.);
    ak->b15c = z15(e0, f0, inc, bet, 0., 1.);   ak->b16c = z16(e0, f0, inc, bet, 0., 1.);
    ak->b17c = z17(e0, f0, inc, bet, 0., 1.);   ak->b18c = z18(e0, f0, inc, bet, 0., 1.);
    ak->b19c = z19(e0, f0, inc, bet, 0., 1.);   ak->b20c = z20(e0, f0, inc, bet, 0., 1.);
    ak->b21c = z21(e0, f0, inc, bet, 0., 1.);   ak->b22c = z22(e0, f0, inc, bet, 0., 1.);
    ak->b23c = z23(e0, f0, inc, bet, 0., 1.);   ak->b24c = z24(e0, f0, inc, bet, 0., 1.);
    ak->b25c = z25(e0, f0, inc, bet, 0., 1.);   ak->b26c = z26(e0, f0, inc, bet, 0., 1.);
    ak->b27c = z27(e0, f0, inc, bet, 0., 1.);   ak->b28c = z28(e0, f0, inc, bet, 0., 1.);
    ak->b29c = z29(e0, f0, inc, bet, 0., 1.);

    ak->d1c = q1(e0, f0, inc, bet, 0., 1.);     ak->d2c = q2(e0, f0, inc, bet, 0., 1.);
    ak->d3c = q3(e0, f0, inc, bet, 0., 1.);     ak->d4c = q4(e0, f0, inc, bet, 0., 1.);
    ak->d5c = q5(f0, inc, bet, 0., 1.);         ak->d6c = q6(e0, f0, inc, bet, 0., 1.);
    ak->d7c = q7(e0, f0, inc, bet, 0., 1.);     ak->d8c = q8(e0, f0, inc, bet, 0., 1.);
    ak->d9c = q9(e0, f0, inc, bet, 0., 1.);     ak->d10c = q10(e0, f0, inc, bet, 0., 1.);
    ak->d11c =q11(e0, f0, inc, bet, 0., 1.);    ak->d12c = q12(e0, f0, inc, bet, 0., 1.);
    ak->d13c = q13(e0, f0, inc, bet, 0., 1.);   ak->d14c = q14(e0, f0, inc, bet, 0., 1.);
    ak->d15c = q15(e0, f0, inc, bet, 0., 1.);   ak->d16c = q16(e0, f0, inc, bet, 0., 1.);
    ak->d17c = q17(e0, f0, inc, bet, 0., 1.);   ak->d18c = q18(e0, f0, inc, bet, 0., 1.);
    ak->d19c = q19(e0, f0, inc, bet, 0., 1.);   ak->d20c = q20(e0, f0, inc, bet, 0., 1.);
    ak->d21c = q21(e0, f0, inc, bet, 0., 1.);   ak->d22c = q22(e0, f0, inc, bet, 0., 1.);
    ak->d23c = q23(e0, f0, inc, bet, 0., 1.);   ak->d24c = q24(e0, f0, inc, bet, 0., 1.);
    ak->d25c = q25(e0, f0, inc, bet, 0., 1.);   ak->d26c = q26(e0, f0, inc, bet, 0., 1.);
    ak->d27c = q27(e0, f0, inc, bet, 0., 1.);   ak->d28c = q28(e0, f0, inc, bet, 0., 1.);
    ak->d29c = q29(e0, f0, inc, bet, 0., 1.);

    return 0;

}


static double Heaviside( double x){
if( x >= 0 )
    return 1;

return 0;
}


int SimInspiralEccentricFD(
        Complex16FDWaveform **htilde,           /**< FD waveform: plus&cross polarization */
        const double phiRef,                    /**< Orbital coalescence phase (rad) */
        const double deltaF,                    /**< Frequency resolution */
        const double m1_SI,                     /**< Mass of companion 1 (kg) */
        const double m2_SI,                     /**< Mass of companion 2 (kg) */
        const double fStart,                    /**< Start GW frequency (Hz) */
        const double fEnd,                      /**< Highest GW frequency (Hz): end at Schwarzschild ISCO */
        const double i,                         /**< Polar inclination of source (rad) */
        const double r,                         /**< Distance of source (m) */
        const double inclination_azimuth,       /**< Azimuthal component of inclination angles [0, 2 PI]*/
        const double e_min,                     /**< Initial eccentricity at frequency f_min: range [0, 0.4] */
        const bool space_cutoff                 /**< BOOL: if further cutoff the frequency (ONLY for space detector) */
)
{
    const double m1 = m1_SI / MSUN_SI;
    const double m2 = m2_SI / MSUN_SI;
    const double m = m1 + m2;
    const double Mtotal = m * MTSUN_SI;  /* total mass in seconds */
    const double eta = m1 * m2 / (m * m);
    const double piM = PI * Mtotal;
    const double vISCO = 1. / sqrt(6.);
    const double fISCO = vISCO * vISCO * vISCO / piM;
    const double mchirp= pow(eta, 3./5.)*Mtotal;
    double shft, f_max, fupper = 2.*fISCO, f_yr=0., heavy;
    double f;

    gsl_complex cphase, exphase, czeta_FPlus,czeta_FCross, czeta_FPlus_times_exphase,czeta_FCross_times_exphase;
    const double zr=0.;
    double Amplitude, zim, Phaseorder, pc_re_hplus, pc_im_hplus, pc_re_hcross, pc_im_hcross, re_hplus, im_hplus, re_hcross, im_hcross;
    size_t j, n, jStart;

    expnCoeffsEPC ak;
    EPCSetup( &ak, m1, m2, fStart, i, inclination_azimuth, e_min);


    complex double *data_p = NULL;
    complex double *data_c = NULL;
    double tC = 0.;  // coalesce at t=0, FIXME: Why `lal` use -1. / deltaF ?

    Complex16FDWaveform *htilde_;

    /* Perform some initial checks */
    if (!htilde) ERROR(PD_EFAULT, NULL);
    if (*htilde) ERROR(PD_EFAULT, NULL);
    if (m1_SI <= 0) ERROR(PD_EDOM, NULL);
    if (m2_SI <= 0) ERROR(PD_EDOM, NULL);
    if (fStart <= 0) ERROR(PD_EDOM, NULL);
    if (r <= 0) ERROR(PD_EDOM, NULL);
    if (fEnd < 0) ERROR(PD_EDOM, NULL);


    /* allocate htilde_p and htilde_c*/
    if ( fEnd == 0. ) // End at ISCO
        f_max = fISCO;
    else // End at user-specified freq.
        f_max = fEnd;
    n = (size_t) (f_max / deltaF + 1);
    if (space_cutoff){
        fupper = (2.*fISCO > 1.) ? 1. : 2.*fISCO;
        f_yr = fStart;
    }

    htilde_ = CreateComplex16FDWaveform(deltaF, n);
    if (!htilde_) ERROR(PD_EFUNC, NULL);
    *htilde = htilde_;
    data_p = htilde_->data_p;
    data_c = htilde_->data_c;

    /* extrinsic parameters*/
    Amplitude = -sqrt(5./384.)*pow(PI, -2./3.)*(pow(mchirp,5./6.)/r)*C_SI;
    shft = TWOPI * tC;

    jStart = (size_t) ceil(fStart / deltaF);
    f = (double) jStart*deltaF;

    /* In order to decompose the waveform in the form h = F_+ h_+ + F_x h_x,
     * we decompose the amplitude function using a two basis decomposition.
     * Note that the functions (zeta_real and zeta_im) depend on several parameters, including F_+ and F_x.
     * Since zeta_real and zeta_im are linear function in the antenna pattern functions,
     * czeta_FPlus and czeta_FCross are used to extract F_+ and F_x from the waveform amplitude*/

    for(j=jStart;j<n;j++){
        pc_re_hplus=0.;
        pc_im_hplus=0.;
        pc_re_hcross=0.;
        pc_im_hcross=0.;
        Phaseorder=0.;

        for(int k=1; k<8; k++){
            Phaseorder+=-PhaseAccTaylorF2(k, f, &ak);
        }

        for(int lm=1;lm<11;lm++){

            zim=PI/4. + pow(((double)lm)/2.,8./3.)*Phaseorder - shft*f + ((double)lm)*phiRef;

            cphase = gsl_complex_rect (zr,zim);

            czeta_FPlus = gsl_complex_rect (((double)zeta_generic_re_plus(lm, f, &ak)),((double)zeta_generic_im_plus(lm, f, &ak)));
            czeta_FCross=gsl_complex_rect (((double)zeta_generic_re_cross(lm, f, &ak)),((double)zeta_generic_im_cross(lm, f, &ak)));

            exphase=gsl_complex_exp (cphase);

            czeta_FPlus_times_exphase  = gsl_complex_mul(czeta_FPlus, exphase);
            czeta_FCross_times_exphase = gsl_complex_mul(czeta_FCross, exphase);

            heavy = Heaviside(((double)lm)*fupper-2.*f) * Heaviside(2.*f-((double)lm)*f_yr);
            re_hplus = heavy * pow(((double)lm)/2., 2./3.) * GSL_REAL(czeta_FPlus_times_exphase);
            im_hplus = heavy * pow(((double)lm)/2., 2./3.) * GSL_IMAG(czeta_FPlus_times_exphase);
            re_hcross= heavy * pow(((double)lm)/2., 2./3.) * GSL_REAL(czeta_FCross_times_exphase);
            im_hcross= heavy * pow(((double)lm)/2., 2./3.) * GSL_IMAG(czeta_FCross_times_exphase);

            pc_re_hplus+=re_hplus;
            pc_im_hplus+=im_hplus;
            pc_re_hcross+=re_hcross;
            pc_im_hcross+=im_hcross;
        }

        /*Note that h(f)= FPlus*data_p + FCross*data_c, where the polarizations are given by:*/
        data_p[j] = Amplitude*pow(f,-7./6.)*(pc_re_hplus  + pc_im_hplus*1.0j);
        data_c[j] = Amplitude*pow(f,-7./6.)*(pc_re_hcross + pc_im_hcross*1.0j);

        f+=deltaF;
    }

    return PD_SUCCESS;

}


// Han Wang: Add two funcs to output as (amplitude & phase) or (h & phase)
int SimInspiralEccentricFDAmpPhase(
        AmpPhaseFDWaveform ***h_amp_phase,
        const double phiRef,                    /**< Orbital coalescence phase (rad) */
        const double deltaF,                    /**< Frequency resolution */
        const double m1_SI,                     /**< Mass of companion 1 (kg) */
        const double m2_SI,                     /**< Mass of companion 2 (kg) */
        const double fStart,                    /**< Start GW frequency (Hz) */
        const double fEnd,                      /**< Highest GW frequency (Hz): end at Schwarzschild ISCO */
        const double i,                         /**< Polar inclination of source (rad) */
        const double r,                         /**< Distance of source (m) */
        const double inclination_azimuth,       /**< Azimuthal component of inclination angles [0, 2 PI]*/
        const double e_min,                     /**< Initial eccentricity at frequency f_min: range [0, 0.4] */
        const bool space_cutoff                 /**< BOOL: if further cutoff the frequency (ONLY for space detector) */
)
{
    const double m1 = m1_SI / MSUN_SI;
    const double m2 = m2_SI / MSUN_SI;
    const double m = m1 + m2;
    const double Mtotal = m * MTSUN_SI;  /* total mass in seconds */
    const double eta = m1 * m2 / (m * m);
    const double piM = PI * Mtotal;
    const double vISCO = 1. / sqrt(6.);
    const double fISCO = vISCO * vISCO * vISCO / piM;
    const double mchirp= pow(eta, 3./5.)*Mtotal;
    double shft, f_max, fupper = 2.*fISCO, f_yr=0., heavy;
    double f;

    complex double czeta_FPlus, czeta_FCross, hplus_a, hcross_a;
    double Amplitude, phase_tay, Phaseorder, h_phase;
    size_t j, n, jStart;

    expnCoeffsEPC ak;
    EPCSetup( &ak, m1, m2, fStart, i, inclination_azimuth, e_min);


    complex double *data_p_a[10] = {}, *data_c_a[10] = {};
    double *data_pha[10] = {};
    double tC = 0.;  // coalesce at t=0, why `lal` use -1. / deltaF ?

    AmpPhaseFDWaveform *htilde_ap;

    /* Perform some initial checks */
    if (!h_amp_phase) ERROR(PD_EFAULT, NULL);
    if (*h_amp_phase) ERROR(PD_EFAULT, NULL);
    if (m1_SI <= 0) ERROR(PD_EDOM, NULL);
    if (m2_SI <= 0) ERROR(PD_EDOM, NULL);
    if (fStart <= 0) ERROR(PD_EDOM, NULL);
    if (r <= 0) ERROR(PD_EDOM, NULL);
    if (fEnd < 0) ERROR(PD_EDOM, NULL);


    /* allocate htilde_p and htilde_c*/
    if ( fEnd == 0. ) // End at ISCO
        f_max = fISCO;
    else // End at user-specified freq.
        f_max = fEnd;
    n = (size_t) (f_max / deltaF + 1);
    if (space_cutoff){
        fupper = (2.*fISCO > 1.) ? 1. : 2.*fISCO;
        f_yr = fStart;
    }

    *h_amp_phase = (AmpPhaseFDWaveform **) malloc(sizeof(AmpPhaseFDWaveform *) * 10);
    for(int lm=0;lm<10;lm++){

        htilde_ap = CreateAmpPhaseFDWaveform(deltaF, n, lm+1);
        if (!htilde_ap) ERROR(PD_EFUNC, NULL);

        (*h_amp_phase)[lm] = htilde_ap;
        data_p_a[lm] = ((*h_amp_phase)[lm])->amp_p;
        data_c_a[lm] = ((*h_amp_phase)[lm])->amp_c;
        data_pha[lm] = ((*h_amp_phase)[lm])->phase;
    }

    /* extrinsic parameters*/
    Amplitude = -sqrt(5./384.)*pow(PI, -2./3.)*(pow(mchirp,5./6.)/r)*C_SI;
    shft = TWOPI * tC;

    jStart = (size_t) ceil(fStart / deltaF);
    f = (double) jStart*deltaF;

    /* In order to calculate the space antenna pattern, we need to decompose waveform into
     * different eccentric harmonics, and each harmonic has its amplitude & phase.*/

    for(j=jStart;j<n;j++){

        Phaseorder=0.;
        for(int k=1; k<8; k++){
            Phaseorder+=-PhaseAccTaylorF2(k, f, &ak);
        }

        for(int lm=1;lm<11;lm++){
            // Eq.(4.28)
            phase_tay = PI/4. + pow(((double)lm)/2., 8./3.)*Phaseorder - shft*f + ((double)lm)*phiRef;
            // Eq.(4.20)
            czeta_FPlus = zeta_generic_re_plus(lm, f, &ak) + 1.j * zeta_generic_im_plus(lm, f, &ak);
            czeta_FCross= zeta_generic_re_cross(lm, f, &ak)+ 1.j * zeta_generic_im_cross(lm, f, &ak);

            /* Note: You may wonder why we deprecate this part:
             *       `phase_tay` here is the physically real PN Fourier phase, although czeta is complex.
             *       Since the phase is independent of +&x, we only save it once for each harmonic. */
            // Eq.(4.22)
            //czeta_FPlus_arg  = atan2(GSL_IMAG(czeta_FPlus), GSL_REAL(czeta_FPlus));
            //czeta_FPlus_abs  = gsl_complex_abs(czeta_FPlus);
            //czeta_FPlus_arg  = gsl_complex_arg(czeta_FPlus);

            // Set Heaviside funcs like Eq.(5.5) & Eq.(5.6)
            heavy = Heaviside(((double)lm)*fupper-2.*f) * Heaviside(2.*f-((double)lm)*f_yr);
            hplus_a = heavy * pow(((double)lm)/2., 2./3.) * czeta_FPlus;
            hcross_a= heavy * pow(((double)lm)/2., 2./3.) * czeta_FCross;
            h_phase = heavy * phase_tay;

            data_p_a[lm-1][j] = Amplitude*pow(f, -7./6.)*hplus_a;
            data_c_a[lm-1][j] = Amplitude*pow(f, -7./6.)*hcross_a;
            data_pha[lm-1][j] = h_phase;
        }

        f+=deltaF;
    }

    return PD_SUCCESS;

}


int SimInspiralEccentricFDAndPhase(
        AmpPhaseFDWaveform ***h_and_phase,
        const double phiRef,                    /**< Orbital coalescence phase (rad) */
        const double deltaF,                    /**< Frequency resolution */
        const double m1_SI,                     /**< Mass of companion 1 (kg) */
        const double m2_SI,                     /**< Mass of companion 2 (kg) */
        const double fStart,                    /**< Start GW frequency (Hz) */
        const double fEnd,                      /**< Highest GW frequency (Hz): end at Schwarzschild ISCO */
        const double i,                         /**< Polar inclination of source (rad) */
        const double r,                         /**< Distance of source (m) */
        const double inclination_azimuth,       /**< Azimuthal component of inclination angles [0, 2 PI]*/
        const double e_min,                     /**< Initial eccentricity at frequency f_min: range [0, 0.4] */
        const bool space_cutoff                 /**< BOOL: if further cutoff the frequency (ONLY for space detector) */
)
{
    const double m1 = m1_SI / MSUN_SI;
    const double m2 = m2_SI / MSUN_SI;
    const double m = m1 + m2;
    const double Mtotal = m * MTSUN_SI;  /* total mass in seconds */
    const double eta = m1 * m2 / (m * m);
    const double piM = PI * Mtotal;
    const double vISCO = 1. / sqrt(6.);
    const double fISCO = vISCO * vISCO * vISCO / piM;
    const double mchirp= pow(eta, 3./5.)*Mtotal;
    double shft, f_max, fupper = 2.*fISCO, f_yr=0., heavy;
    double f;

    complex double czeta_FPlus, czeta_FCross, hplus_a, hcross_a, exp_phase;
    double Amplitude, phase_tay, Phaseorder, h_phase;
    size_t j, n, jStart;

    expnCoeffsEPC ak;
    EPCSetup( &ak, m1, m2, fStart, i, inclination_azimuth, e_min);


    complex double *data_p_a[10] = {}, *data_c_a[10] = {};
    double *data_pha[10] = {};
    double tC = 0.;  // coalesce at t=0, why `lal` use -1. / deltaF ?

    AmpPhaseFDWaveform *htilde_ap;

    /* Perform some initial checks */
    if (!h_and_phase) ERROR(PD_EFAULT, NULL);
    if (*h_and_phase) ERROR(PD_EFAULT, NULL);
    if (m1_SI <= 0) ERROR(PD_EDOM, NULL);
    if (m2_SI <= 0) ERROR(PD_EDOM, NULL);
    if (fStart <= 0) ERROR(PD_EDOM, NULL);
    if (r <= 0) ERROR(PD_EDOM, NULL);
    if (fEnd < 0) ERROR(PD_EDOM, NULL);


    /* allocate htilde_p and htilde_c*/
    if ( fEnd == 0. ) // End at ISCO
        f_max = fISCO;
    else // End at user-specified freq.
        f_max = fEnd;
    n = (size_t) (f_max / deltaF + 1);
    if (space_cutoff){
        fupper = (2.*fISCO > 1.) ? 1. : 2.*fISCO;
        f_yr = fStart;
    }

    *h_and_phase = (AmpPhaseFDWaveform **) malloc(sizeof(AmpPhaseFDWaveform *) * 10);
    for(int lm=0;lm<10;lm++){

        htilde_ap = CreateAmpPhaseFDWaveform(deltaF, n, lm+1);
        if (!htilde_ap) ERROR(PD_EFUNC, NULL);

        (*h_and_phase)[lm] = htilde_ap;
        data_p_a[lm] = ((*h_and_phase)[lm])->amp_p;
        data_c_a[lm] = ((*h_and_phase)[lm])->amp_c;
        data_pha[lm] = ((*h_and_phase)[lm])->phase;
    }

    /* extrinsic parameters*/
    Amplitude = -sqrt(5./384.)*pow(PI, -2./3.)*(pow(mchirp,5./6.)/r)*C_SI;
    shft = TWOPI * tC;

    jStart = (size_t) ceil(fStart / deltaF);
    f = (double) jStart*deltaF;

    /* In order to calculate the space antenna pattern, we need to decompose waveform into
     * different eccentric harmonics, for here, we directly output h instead of amplitude, and also the phase.*/

    for(j=jStart;j<n;j++){

        Phaseorder=0.;
        for(int k=1; k<8; k++){
            Phaseorder+=-PhaseAccTaylorF2(k, f, &ak);
        }

        for(int lm=1;lm<11;lm++){
            // Eq.(4.28)
            phase_tay = PI/4. + pow(((double)lm)/2., 8./3.)*Phaseorder - shft*f + ((double)lm)*phiRef;
            exp_phase = cexp(1.j * phase_tay);
            // Eq.(4.20)
            czeta_FPlus = zeta_generic_re_plus(lm, f, &ak) + 1.j * zeta_generic_im_plus(lm, f, &ak);
            czeta_FCross= zeta_generic_re_cross(lm, f, &ak)+ 1.j * zeta_generic_im_cross(lm, f, &ak);

            // Set Heaviside funcs like Eq.(5.5) & Eq.(5.6)
            heavy = Heaviside(((double)lm)*fupper-2.*f) * Heaviside(2.*f-((double)lm)*f_yr);
            hplus_a = heavy * pow(((double)lm)/2., 2./3.) * czeta_FPlus  * exp_phase;
            hcross_a= heavy * pow(((double)lm)/2., 2./3.) * czeta_FCross * exp_phase;
            h_phase = phase_tay;  // may do not need to multiply `heavy` for phase?

            data_p_a[lm-1][j] = Amplitude*pow(f, -7./6.)*hplus_a;
            data_c_a[lm-1][j] = Amplitude*pow(f, -7./6.)*hcross_a;
            data_pha[lm-1][j] = h_phase;
        }

        f+=deltaF;
    }

    return PD_SUCCESS;

}
