#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "LISA.h"
#include "Constants.h"

void instrument_noise(double f, double *SAE, double *SXYZ)
{
    //Power spectral density of the detector noise and transfer frequency
    double red, Sloc;
    double trans;
        
    red  = 16.0*(pow((2.0e-5/f), 10.0)+ (1.0e-4/f)*(1.0e-4/f));
    Sloc = 2.89e-24;
    
    // Calculate the power spectral density of the detector noise at the given frequency
    trans = pow(sin(f/fstar), 2.0);
    
    *SAE = 16.0/3.0*trans*( (2.0+cos(f/fstar))*(Sps + Sloc) 
    					    +2.0*( 3.0 + 2.0*cos(f/fstar) + cos(2.0*f/fstar) )
    					        *( Sloc/2.0 + Sacc/pow(2.0*PI*f,4.0)*(1.0+red) ) )
    					  / pow(2.0*Larm,2.0);
    
    *SXYZ = 4.0*trans*( 4.0*(Sps+Sloc) 
                      + 8.0*( 1.0+pow(cos(f/fstar),2.0) )*( Sloc/2.0 + Sacc/pow(2.0*PI*f,4.0)*(1.0+red) ) )
                       / pow(2.0*Larm,2.0);
    
    return;
}

void spacecraft_LISA(double t, double *x, double *y, double *z)
{
	double alpha;
	double beta1, beta2, beta3;
	double sa, sb, ca, cb;

	alpha = 2.*PI*fm*t + kappa;

	beta1 = 0. + lambda;
	beta2 = 2.*PI/3. + lambda;
	beta3 = 4.*PI/3. + lambda;

	sa = sin(alpha);
	ca = cos(alpha);

	sb = sin(beta1);
	cb = cos(beta1);
	x[0] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[0] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[0] = -SQ3*AU*ec*(ca*cb + sa*sb);

	sb = sin(beta2);
	cb = cos(beta2);
	x[1] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[1] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[1] = -SQ3*AU*ec*(ca*cb + sa*sb);

	sb = sin(beta3);
	cb = cos(beta3);
	x[2] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
	y[2] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
	z[2] = -SQ3*AU*ec*(ca*cb + sa*sb);

	return;
}

void spacecraft_TianQin(double t, double *x, double *y, double *z)
{

    double alpha = PI2*t*3.168753575e-8;
    
    //calculate alpha_n
    double kappa1 = 0.;
    double kappa2 = 2.0943951023932; //2.*pi/3.;
    double kappa3 = 4.18879020478639;//4.*pi/3.;
    double lambda = PIon2;
    double fsc = 3.170979198e-06; //1/(3.65day);

    double an_1 = PI2*t*fsc + kappa1 + lambda;
    double an_2 = PI2*t*fsc + kappa2 + lambda;
    double an_3 = PI2*t*fsc + kappa3 + lambda;
    
    double R = AU;
    double R1 = 1.0e8;

    double beta = -2.102137 - M_PI + M_PI/4.;
    double beta1 = 0;

    double phi_s = 120.5*RAD2DEG;
    double theta_s = -4.7*RAD2DEG;
    
    double sp = sin(phi_s);
    double st = sin(theta_s);
    double cp = cos(phi_s);
    double ct = cos(theta_s);

    double sab = sin(alpha-beta);
    double cab = cos(alpha-beta);

    double ecc = 0.0167;

    //TianQin orbit function
    double x_earth = R*cab + R*ecc*(cos(2*(alpha-beta))-3)/2.0;
    double y_earth = R*sab + R*ecc*sin(2*(alpha-beta))/2.0;
    double z_earth = 0.0;


    x[0] = R1*(cp*st*sin(an_1-beta1) + cos(an_1-beta1)*sp) + x_earth;
    y[0] = R1*(sp*st*sin(an_1-beta1) - cos(an_1-beta1)*cp) + y_earth;
    z[0] = -R1*sin(an_1-beta1)*ct + z_earth;
    
    x[1] = R1*(cp*st*sin(an_2-beta1) + cos(an_2-beta1)*sp) + x_earth;
    y[1] = R1*(sp*st*sin(an_2-beta1) - cos(an_2-beta1)*cp) + y_earth;
    z[1] = -R1*sin(an_2-beta1)*ct + z_earth;
    
    x[2] = R1*(cp*st*sin(an_3-beta1) + cos(an_3-beta1)*sp) + x_earth;
    y[2] = R1*(sp*st*sin(an_3-beta1) - cos(an_3-beta1)*cp) + y_earth;
    z[2] = -R1*sin(an_3-beta1)*ct + z_earth;

    return;
}
