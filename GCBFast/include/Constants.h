#ifndef Constants_h
#define Constants_h



/* --------------  MATHEMATICAL CONSTANTS  -------------- */
/* Square root of 3 */
#define SQ3   1.73205080757

/* Pi's and frinds */
//use math.h (M_PI) for PI
#define PI 3.141592653589793238462643383279502884
#define PI2   6.283185307179586
#define PIon2 1.57079632679
#define PIon4 0.78539816339

/* Natural log of 2 */
#define LN2 0.693147180559945




/* ----------------  NATURAL CONSTANTS  ----------------- */

/* Speed of light (m/s) */
#define C 299792458.

/* Mass of the Sun (s) */
#define TSUN  4.9169e-6

/* Number of meters in a parsec */
#define PC 3.0856775807e16

/* Number of seconds in a year */
#define YEAR 31457280.0

/* Astronomical unit (meters) */
#define AU 1.49597870660e11

#define MSUN 1.989e30

#define G 6.67e-11


#define EarthEccentricity 0.01671
// Orbital eccentricity for earth:
// https://handwiki.org/wiki/Astronomy:Orbital_eccentricity


#define EarthOrbitOmega_SI 1.99098659277e-7  
/* Orbital pulsation: 2pi/year - use sidereal year as found on http://hpiers.obspm.fr/eop-pc/models/constants.html */

// vernal equinox is Septemper equinox (09-22/23)
// perihelion is at about 01-03/04
// (31 + 30 + 31 + 11)/365.2425 * 2 * PI
#define Perihelion_Ang 1.772  /* angle measured from the vernal equinox to the perihelion */

/***********************************************************/
/* ecliptic longitude and latitude of J0806.3+1527 */
/***********************************************************/
#define J0806_phi 2.103121748653167  // 120.5
#define J0806_theta 1.65282680163863 // -4.7 = 90 + 4.7


#endif /* Constants_h */
