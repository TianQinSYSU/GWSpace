/* ===============================================
 * File Name: TianQin.c
 * Author: ekli
 * Mail: lekf123@163.com
 * Created Time: 2022-11-25 20:22:32
 * ===============================================
 */

#include <stdio.h>

#include constants

typedef tagOrbit
{
    double *x;
    double *y;
    double *z;
    size_t *size;
} Orbit;

typedef tagSpacecraftOrbit
{
    char   *detector;
    double armlength;
    double radius;
    double point_theta = 0.0;
    double point_phi = 0.0;
    double spa_lambda = 0.0;
    size_t size;
    double *time;
    Orbit  *S1_local;
    Orbit  *S2_local;
    Orbit  *S3_local;
    Orbit  *S1_SSB;
    Orbit  *S2_SSB;
    Orbit  *S3_SSB;
    Orbit  *So_SSB;

} SpacecraftOrbit;

int TianQin_Spacecraft(SpacecraftOrbit *self, double *time, size_t size)
{
    self->detector = "TianQin";
    self->armlength = SQRT3 *1.0e5; // sqrt{3} radius
    self->radius = 1.0e5;
    self->size = size;
    self->time = time;
    self->point_theta = -0.08203047484373349; // -4.7 degree
    self->point_phi   = 2.103121748653167;    // 120.5 degree


    return 0;
}

int _EarthOrbit(SpacecraftOrbit *self)
{
    double ecc_earth = 0.0167;
    double fm = 1.0/YRSID_SI;
    double alpbet;
    int i;

    for (i=0; i< self.size; i++)
    {
        alpbet = 2*PI * fm *self.time[i] + self.kappa_0 - beta;
        self.So_SSB->x[i] = AU_SI *( cos(alpbet) + 0.5 *ecc_earth *(cos(2*alpbet) - 3) );
        self.So_SSB->y[i] = AU_SI &( sin(alpbet) + 0.5 *ecc_earth * sin(2*alpbet) );
        self.So_SSB->z[i] = 0.0;
    }

    return SUCCESS;
}

double _kappa_n(SpacecraftOrbit *self, int n)
{
    return (n-1) * 2.0* PI /3.0 + self.spa_lambda;
}

int _TianQinOrbits(SpacecraftOrbit *self)
{
    double cstheta = cos(self.point_theta);
    double sntheta = sin(self.point_theta);
    double csphi = cos(self.point_phi);
    double snphi = sin(self.point_phi);
    
    double alpbet;
    double csalpbet;
    double snalpbet;
    int i;
    
    for (i=0; i< self.size; i++)
    {
        alpbet = 2*PI*fsc * self.time[i] + _kappa_n(&self, 1);
        csalpbet = cos(alpbet);
        snalpbet = sin(alpbet);

        self.S1_local->x[i] = self.radius * (csphi * sntheta * snalpbet + csalpbet * snphi);
        self.S1_local->y[i] = self.radius * (snphi * sntheta * snalpbet - csalpbet * csphi);
        self.S1_local->x[i] = - self.radius * snalpbet * cstheta;
        
        alpbet = 2*PI*fsc * self.time[i] + _kappa_n(&self, 2);
        csalpbet = cos(alpbet);
        snalpbet = sin(alpbet);

        self.S2_local->x[i] = self.radius * (csphi * sntheta * snalpbet + csalpbet * snphi);
        self.S2_local->y[i] = self.radius * (snphi * sntheta * snalpbet - csalpbet * csphi);
        self.S2_local->x[i] = - self.radius * snalpbet * cstheta;
        
        alpbet = 2*PI*fsc * self.time[i] + _kappa_n(&self, 3);
        csalpbet = cos(alpbet);
        snalpbet = sin(alpbet);

        self.S3_local->x[i] = self.radius * (csphi * sntheta * snalpbet + csalpbet * snphi);
        self.S3_local->y[i] = self.radius * (snphi * sntheta * snalpbet - csalpbet * csphi);
        self.S3_local->x[i] = - self.radius * snalpbet * cstheta;
    }

    return SUCCESS;
}
