#ifndef _IMR_PHENOMINTERNALUTILS_H
#define _IMR_PHENOMINTERNALUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

// #ifdef __GNUC__
// #define UNUSED __attribute__((unused))
// #else
// #define UNUSED
// #endif

#include "struct.h"

// UNUSED void PhenomInternal_UtilsTest(void);

UNUSED bool PhenomInternal_approx_equal(double x, double y, double epsilon);

UNUSED void PhenomInternal_nudge(double *x, double X, double epsilon);

UNUSED size_t PhenomInternal_NextPow2(const size_t n);

// UNUSED int PhenomInternal_AlignedSpinEnforcePrimaryIsm1(double *m1, double *m2, double *chi1z, double *chi2z);

#ifdef __cplusplus
}
#endif

#endif /* _IMR_PHENOMINTERNALUTILS_H */
