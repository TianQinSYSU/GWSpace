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

void spacecraft(double t, double *x, double *y, double *z)
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
