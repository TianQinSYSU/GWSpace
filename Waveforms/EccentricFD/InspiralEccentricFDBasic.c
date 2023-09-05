// Created by dice on 2022/3/26.
// Copy some structures from `pyIMRPhenomD`(to avoid using `lal`)

#include "InspiralEccentricFD.h"

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif


Complex16FDWaveform* CreateComplex16FDWaveform(
        double deltaF,
        size_t length
){
    Complex16FDWaveform* fs = (Complex16FDWaveform*)malloc(sizeof(Complex16FDWaveform));
    fs->deltaF = deltaF;

    fs->length = length;
    fs->data_p = (double complex *)malloc(sizeof(double complex) * length);
    fs->data_c = (double complex *)malloc(sizeof(double complex) * length);
    if ((fs->data_p == NULL) || (fs->data_c == NULL))
        ERROR(PD_ENOMEM, "Failed to allocated data array.");

    memset(fs->data_p, 0, sizeof(double complex) * length);
    memset(fs->data_c, 0, sizeof(double complex) * length);
    return fs;
}

// https://stackoverflow.com/questions/1879550 Discussions about set pointers to `NULL` after freeing them
void DestroyComplex16FDWaveform(Complex16FDWaveform* wf) {
    free(wf->data_p);
    free(wf->data_c);
    free(wf);
}

AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(
        double deltaF,
        size_t length,
        unsigned int harmonic
){
    AmpPhaseFDWaveform* wf = (AmpPhaseFDWaveform*)malloc(sizeof(AmpPhaseFDWaveform));
    wf->deltaF = deltaF;

    wf->length = length;
    wf->amp_p = (double complex *) malloc(sizeof(double complex) * length);
    wf->amp_c = (double complex *) malloc(sizeof(double complex) * length);
    wf->phase = (double*) malloc(sizeof(double) * length);
    wf->harmonic = harmonic;
    if ((wf->amp_p == NULL) || (wf->amp_c == NULL) || (wf->phase == NULL))
        ERROR(PD_ENOMEM, "Failed to allocated one of the data arrays.");

    memset(wf->amp_p, 0, sizeof(double complex) * length);
    memset(wf->amp_c, 0, sizeof(double complex) * length);
    memset(wf->phase, 0, sizeof(double) * length);
    return wf;
}

void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform* wf) {
    free(wf->amp_p);
    free(wf->amp_c);
    free(wf->phase);
    free(wf);
}

// Simplified code from lal/std/XLALError.c
/* Return the error message associated with an error number or return value. */
const char *ErrorString(int code) {

    if (code <= 0) {    /* this is a return code, not an error number */
        if (code == 0)
            return "PD_SUCCESS";
        else if (code == -1)
            return "Failure";
        else
            return "Unknown return code";
    }

    /* check to see if an internal function call has failed, but the error
     * number was not "or"ed against the mask PD_EFUNC */
    if (code == PD_EFUNC)
        return "Internal function call failed";

    /* use this to report error strings... deals with possible mask for
     * errors arising from internal function calls */
    # define ERROR_STRING(s) \
    ( ( code & PD_EFUNC ) ? "Internal function call failed: " s : (const char *) s )
    switch (code & ~PD_EFUNC) {
        /* these are standard error numbers */
        case PD_EFAULT:
            return ERROR_STRING("Invalid pointer");
        case PD_EDOM:
            return ERROR_STRING("Input domain error");
        case PD_ENOMEM:
            return ERROR_STRING("Memory allocation error");
            /* unrecognized error number */
        default:
            return "Unknown error";
    }
    # undef ERROR_STRING
    //return NULL;        /* impossible to get here */
}

void ERROR(ERROR_type e, char *errstr) {
    fprintf(stderr, "%s in %s:%d:\n %s\n", ErrorString(e), __FILE__, __LINE__, errstr);
    exit(-1);
}
