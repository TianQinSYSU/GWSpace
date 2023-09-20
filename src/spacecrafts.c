#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "spacecrafts.h"
#include "Constants.h"

void spacecraft(char detector[], int N, double *t, double *x, double *y, double *z, double *Larm)
{
    for (int i=0; i<N; i++) {
        if (strcmp(detector, "TianQin") == 0) {
            spacecraft_TianQin(t[i], &x[3*i], &y[3*i], &z[3*i]);
            Larm[0] = armLength_tq;
        } else if (strcmp(detector, "LISA") == 0) {
            spacecraft_LISA(t[i], &x[3*i], &y[3*i], &z[3*i]);
            Larm[0] = armLength_lisa;
        } else if (strcmp(detector, "TaiJi") == 0) {
            spacecraft_TaiJi(t[i], &x[3*i], &y[3*i], &z[3*i]);
            Larm[0] = armLength_tj;
        }       
    }
    return;
}

void spacecraft_LISA(double t, double *x, double *y, double *z)
{
    // expend to second order of ecc, see arxiv:gr-qc/0311069
    double a, b;
    a = Omega_lisa*t + kappa_lisa - Perihelion_Ang;
    for (int i=0; i<3; i++) {
        b = i*2*PI/3. + lambda_lisa;
        x[i] = cos(a) + 0.5*ecc_lisa*( (cos(2*a-b) - 3*cos(b)) 
            +0.25*ecc_lisa*(3*cos(3*a-2*b) - 10*cos(a) - 5*cos(a-2*b)) );
        y[i] = sin(a) + 0.5*ecc_lisa*( (sin(2*a-b) - 3*sin(b))
            +0.25*ecc_lisa*(3*sin(3*a-2*b) - 10*cos(a) - 5*cos(a-2*b)) );
        z[i] = -SQRT3*ecc_lisa*(cos(a-b) + 
                ecc_lisa*(1 + sin(a-b)*sin(a-b)) );
        x[i] = AU_SI*x[i];
        y[i] = AU_SI*y[i];
        z[i] = AU_SI*z[i];
    }
	return;
}

void spacecraft_TaiJi(double t, double *x, double *y, double *z)
{
    // expend to second order of ecc, see arxiv:gr-qc/0311069
    double a, b;
    a = Omega_tj*t + kappa_tj - Perihelion_Ang;
    for (int i=0; i<3; i++) {
        b = i*2*PI/3. + lambda_tj;
        x[i] = cos(a) + 0.5*ecc_tj*( (cos(2*a-b) - 3*cos(b)) 
            +0.25*ecc_tj*(3*cos(3*a-2*b) - 10*cos(a) - 5*cos(a-2*b)) );
        y[i] = sin(a) + 0.5*ecc_tj*( (sin(2*a-b) - 3*sin(b))
            +0.25*ecc_tj*(3*sin(3*a-2*b) - 10*cos(a) - 5*cos(a-2*b)) );
        z[i] = -SQRT3*ecc_tj*(cos(a-b) + 
                ecc_tj*(1 + sin(a-b)*sin(a-b)) );
        x[i] = AU_SI*x[i];
        y[i] = AU_SI*y[i];
        z[i] = AU_SI*z[i];
    }
	return;
}

void spacecraft_TianQin(double t, double *x, double *y, double *z)
{
    // earth position
    double alpha = EarthOrbitOmega_SI * t + kappa_tq; // + 0.3490658503988659; // this is ahead of LISA
    double beta = Perihelion_Ang;
    double sna = sin(alpha - beta);
    double csa = cos(alpha - beta);
    double ecc = EarthEccentricity;
    double ecc2 = ecc*ecc;

    double x_earth = AU_SI *( csa - ecc * (1+sna*sna) - 1.5*ecc2 * csa*sna*sna);
    double y_earth = AU_SI * (sna + ecc *sna*csa + 0.5*ecc2 * sna*(1-3*sna*sna));
    double z_earth = 0.0;
    
    //TianQin orbit function
    //calculate alpha_n
    double alpha_tq = Omega_tq * t + lambda_tq;
    
    double sp = sin(J0806_phi);
    double cp = cos(J0806_phi);
    double st = sin(J0806_theta);
    double ct = cos(J0806_theta);

    for (int i=0; i<3; i++){
        alpha = alpha_tq + i * 2.*PI/3.;
        csa = cos(alpha); sna = sin(alpha);

        x[i] = ct * cp * sna + sp * csa;
        y[i] = ct * sp * sna - cp * csa;
        z[i] = - st * sna;

        x[i] *= Radius_tq; y[i] *= Radius_tq; z[i] *= Radius_tq;
        x[i] += x_earth; y[i] += y_earth; z[i] += z_earth;
    }

    return;
}
