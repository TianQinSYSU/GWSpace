#ifndef _IMR_PHENOMUTILS_H
#define _IMR_PHENOMUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

// #ifdef __GNUC__
// #define UNUSED __attribute__((unused))
// #else
// #define UNUSED
// #endif

#include "struct.h"
#include "constants.h"

// void XLALSimPhenomUtilsTest(void);

double PhenomUtilsMftoHz(double Mf, double Mtot_Msun);
double PhenomUtilsHztoMf(double fHz, double Mtot_Msun);

double PhenomUtilsFDamp0(double Mtot_Msun, double distance);

#ifdef __cplusplus
}
#endif

#endif /* _IMR_PHENOMUTILS_H */
