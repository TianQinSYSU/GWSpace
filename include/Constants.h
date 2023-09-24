/**
 * \author Sylvain Marsat, University of Maryland - NASA GSFC
 *
 * \brief C header defining useful physical constants (values taken from LAL).
 * Also defines boolean conventions.
 *
 */

#ifndef _CONSTANTS_H
#define _CONSTANTS_H

#define _XOPEN_SOURCE 500

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

#if defined(__cplusplus)
extern "C" {
#elif 0
} /* so that editors will match preceding brace */
#endif

/***************************************************/
/****** Boolean conventions for loading files ******/

#define SUCCESS 0
#define FAILURE 1
#define NONE -1

/*********************************************************/

/* Mathematica 11.3.0.0 */
#define PI           3.141592653589793238462643383279502884
#define PI2          6.283185307179586231995926937088370323
#define PI_2         1.570796326794896619231321691639751442
#define PI_3         1.047197551196597746154214461093167628
#define PI_4         0.785398163397448309615660845819875721
#define SQRTPI       1.772453850905516027298167483341145183
#define SQRTTWOPI    2.506628274631000502415765284811045253
#define INVSQRTPI    0.564189583547756286948079451560772585
#define INVSQRTTWOPI 0.398942280401432677939946059934381868
#define GAMMA        0.577215664901532860606512090082402431
#define SQRT2        1.414213562373095048801688724209698079
#define SQRT3        1.732050807568877293527446341505872367
#define SQRT6        2.449489742783178098197284074705891392
#define INVSQRT2     0.707106781186547524400844362104849039
#define INVSQRT3     0.577350269189625764509148780501957455
#define INVSQRT6     0.408248290463863016366214012450981898

#define GAMMA 0.577215664901532860606512090082402431


/**********************************************************************/
/**************** Physical constants in SI units **********************/

#define C_SI 299792458.
#define G_SI 6.67259e-11
#define MSUN_SI 1.98892e30
#define MTSUN_SI 4.9254923218988636432342917247829673e-6
#define PC_SI 3.0856775807e16
#define KPC_SI 3085.6775807e16
#define MPC_SI 3085677.5807e16
#define AU_SI 1.4959787066e11
#define YRSID_SI 3.15581497635e7 /* Sideral year as found on http://hpiers.obspm.fr/eop-pc/models/constants.html */

#define DAYSID_SI 86164.09053 // Mean sidereal day

#define EarthEccentricity 0.01671
/* Orbital eccentricity for earth:
* https://handwiki.org/wiki/Astronomy:Orbital_eccentricity
* */

#define EarthOrbitOmega_SI 1.99098659277e-7 /* Orbital pulsation: 2pi/year - use sidereal year as found on http://hpiers.obspm.fr/eop-pc/models/constants.html */

// vernal equinox is Septemper equinox (09-22/23) 
// perihelion is at about 01-03/04
// ##(30 + 31 + 30 + 12)/365.2425 * 2 * PI
// angle measured from the vernal equinox to the perihelion i.e. **Argument of perihelion**
// 102.94719/180 * PI
#define Perihelion_Ang 1.796767421176181 /* angle measured from the vernal equinox to the perihelion */

/***********************************************************/
/* ecliptic longitude and latitude of J0806.3+1527 */
/***********************************************************/
#define J0806_phi 2.103121748653167 // 120.5
#define J0806_theta 1.65282680163863 // -4.7 = 90 + 4.7 

/**********************************************************/
/********** Constants used to relate time scales **********/

#define EPOCH_J2000_0_TAI_UTC 32           /* Leap seconds (TAI-UTC) on the J2000.0 epoch (2000 JAN 1 12h UTC) */
#define EPOCH_J2000_0_GPS 630763213        /* GPS seconds of the J2000.0 epoch (2000 JAN 1 12h UTC) */


/*******************************************/
/**************** NaN **********************/

#ifndef NAN
# define NAN (INFINITY-INFINITY)
#endif

#if 0
{ /* so that editors will match succeeding brace */
#elif defined(__cplusplus)
}
#endif

#endif /* _CONSTANTS_H */
