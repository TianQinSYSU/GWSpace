/* ===============================================
 * File Name: TianQin.c
 * Author: ekli
 * Mail: lekf123@163.com
 * Created Time: 2022-11-25 20:22:32
 * ===============================================
 */

#include <stdio.h>
#include <complex.h>

#include "constants.h"
#include "utils.h"
#include "TianQin.h"


typedef struct tag_evlGslr {
    double complex G12;
    double complex G21;
    double complex G23;
    double complex G32;
    double complex G31;
    double complex G13;
 } evlGslr;

#define cmplx double complex

double d_dot_product_1d(double *arr1, double *arr2)
{
    double out = 0.0;
    for (int i=0; i<3; i++) {
        out += arr1[i] * arr2[i];
    }
    return out;
}

cmplx d_vec_H_vec_product(double *arr1, cmplx *H, double *arr2)
{
    cmplx out;
    out.re = 0.0;
    out.im = 0.0;
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            out.re += arr1[i] * H[i*3+j].re * arr[j];
            out.im += arr1[i] * H[i*3+j].im * arr[j];
        }
    }
    return out;
}

double d_sinc(double x)
{
    if (x == 0.0) return 1.0;
    else return sin(x)/x;
}

evlGslr EvaluateGslr(evlGslr *out_Gslr, double tf, double freq, cmplx *H, double *k)
{
    double alpbet = EarthOrbitOmega_SI *tf + 0.0 - 0.0; // kappa = 0; beta = 0;
    double csab = cos(alpbet);
    double snab = sin(alpbet);
    double ecc = EarthEccentricity;

    double hat_p0[3] = { 
                csab + ecc * (csab**2 - 2) - 1.5 * ecc**2 * csab * snab**2,
                snab * (1 +ecc* csab + 0.5 * ecc**2 * (csab**2 -2 )),
                0.0 };
    
    double cst = cos(J0806_theta);
    double snt = sin(J0806_theta);
    double csp = cos(J0806_phi);
    double snp = sin(J0806_phi);

    alpbet = TianQinOmega * tf + 0.0 - 0.0; // lambda = 0; beta = 0;
    csab = cos(alpbet);
    snab = sin(alpbet);

    double hat_p1[3] = {
                cst * csp * csab - snp * snab,
                cst * snp * csab - csp * snab,
                -snt * csab };
    
    alpbet = TianQinOmega * tf + 2.0/3.0*PI + 0.0 - 0.0; // lambda = 0; beta = 0;
    csab = cos(alpbet);
    snab = sin(alpbet);

    double hat_p2[3] = {
                cst * csp * csab - snp * snab,
                cst * snp * csab - csp * snab,
                -snt * csab };
    
    alpbet = TianQinOmega * tf + 4.0/3.0*PI + 0.0 - 0.0; // lambda = 0; beta = 0;
    csab = cos(alpbet);
    snab = sin(alpbet);

    double hat_p3[3] = {
                cst * csp * csab - snp * snab,
                cst * snp * csab - csp * snab,
                -snt * csab };

    // comput k \cdot nl & nl \cdot H \codt nl
    double n[3];

    // for n1 --> slr = 2 1 3
    for (int i=0; i<3; i++) {
        n[i] = (hat_p3[i] - hat_p2[i]) * INVSQRT3;
    }

    double kn1 = d_dot_product_1d(k, n);         // k \cdot n1
    cmplx n1Hn1 = d_vec_H_vec_product(n, H, n);  // n1 \cdot H \cdot n1
    
    // for n2 --> slr = 3 2 1
    for (int i=0; i<3; i++) {
        n[i] = (hat_p1[i] - hat_p3[i]) * INVSQRT3;
    }
    double kn2 = d_dot_product_1d(k, n);         // k \cdot n2
    cmplx n2Hn2 = d_vec_H_vec_product(n, H, n);  // n2 \cdot H \cdot n2
    
    // for n3 --> slr = 1 3 2
    for (int i=0; i<3; i++) {
        n[i] = (hat_p2[i] - hat_p1[i]) * INVSQRT3;
    }
    double kn3 = d_dot_product_1d(k, n);         // k \cdot n2
    cmplx n3Hn3 = d_vec_H_vec_product(n, H, n);  // n2 \cdot H \cdot n2

    // comput k\cdot(pr + ps) = k \cdot p_0 + k \cdot (pr_L + ps_L) 
    // = R k\cdot hat_p0 + L/sqrt{3} k\cdot (hat_pr + hat_ps)
    double hat_psr[3] = {
                (hat_p1[0] + hat_p2[0]) * INVSQRT3,
                (hat_p1[1] + hat_p2[1]) * INVSQRT3,
                (hat_p1[2] + hat_p2[2]) * INVSQRT3 };
    double hat_kp1p2 = d_dot_product_1d(k, hat_psr);

    hat_psr[3] = {
                (hat_p2[0] + hat_p3[0]) * INVSQRT3,
                (hat_p2[1] + hat_p3[1]) * INVSQRT3,
                (hat_p2[2] + hat_p3[2]) * INVSQRT3 };
    double hat_kp2p3 = d_dot_product_1d(k, hat_psr);

    hat_psr[3] = {
                (hat_p1[0] + hat_p3[0]) * INVSQRT3,
                (hat_p1[1] + hat_p3[1]) * INVSQRT3,
                (hat_p1[2] + hat_p3[2]) * INVSQRT3 };
    double hat_kp3p1 = d_dot_product_1d(k, hat_psr);

    double hat_kp0 = d_dot_product_1d(k, hat_p0);

    // compute single link transfer 
    double R2L = AU_SI/TianQinArmLength; // R/L
    double temp;
    double u = PI * freq * TianQinArmLength / C_SI;
    
    cmplx factor_exp12;
    temp = u * (1 + 2 * hat_kp0 * R2L + hat_kp1p2 );
    factor_exp12.re = cos(temp);
    factor_exp12.im = sin(temp);
    
    cmplx factor_exp23;
    temp = u * (1 + 2 * hat_kp0 * R2L + hat_kp2p3 );
    factor_exp23.re = cos(temp);
    factor_exp23.im = sin(temp);
    
    cmplx factor_exp31;
    temp = u * (1 + 2 * hat_kp0 * R2L + hat_kp3p1 );
    factor_exp31.re = cos(temp);
    factor_exp31.im = sin(temp);

    double factor_sinc12 = 0.5 * u * d_sinc(u * (1. - kn3)); // slr-> 132
    double factor_sinc21 = 0.5 * u * d_sinc(u * (1. + kn3)); // slr-> 231
    double factor_sinc23 = 0.5 * u * d_sinc(u * (1. - kn1)); // slr-> 213
    double factor_sinc32 = 0.5 * u * d_sinc(u * (1. + kn3)); // slr-> 312
    double factor_sinc31 = 0.5 * u * d_sinc(u * (1. - kn2)); // slr-> 321
    double factor_sinc13 = 0.5 * u * d_sinc(u * (1. + kn2)); // slr-> 123

    evlGslr out_Gslr;

    tmp = factor_sinc12 * n3Hn3;
    out_Gslr.G12.re = tmp * factor_exp12.re;
    out_Gslr.G12.im = tmp * factor_exp12.im;

    tmp = factor_sinc21 * n3Hn3;
    out_Gslr.G21.re = tmp * factor_exp12.re;
    out_Gslr.G21.im = tmp * factor_exp12.im;
    
    tmp = factor_sinc23 * n1Hn1;
    out_Gslr.G23.re = tmp * factor_exp23.re;
    out_Gslr.G23.im = tmp * factor_exp23.im;
    
    tmp = factor_sinc32 * n1Hn1;
    out_Gslr.G32.re = tmp * factor_exp23.re;
    out_Gslr.G32.im = tmp * factor_exp23.im;
    
    tmp = factor_sinc31 * n2Hn2;
    out_Gslr.G31.re = tmp * factor_exp31.re;
    out_Gslr.G31.im = tmp * factor_exp31.im;
    
    tmp = factor_sinc13 * n2Hn2;
    out_Gslr.G13.re = tmp * factor_exp31.re;
    out_Gslr.G13.im = tmp * factor_exp31.im;

    return out_Gslr;
}

int TDICombinationFD(evlGslr *in_Gslr, double freq)
{

    return SUCCESS;
}

void main()
{
    
}
