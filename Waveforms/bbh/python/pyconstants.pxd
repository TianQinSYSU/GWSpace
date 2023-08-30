#
# Copyright (C) 2019 Sylvain Marsat.
#
#


"""
    Cython definitions for numerical constants.
"""


cdef extern from "constants.h":

    cdef int _SUCCESS "SUCCESS"
    cdef int _FAILURE "FAILURE"

    double _PI "PI"
    double _PI_2 "PI_2"
    double _PI_3 "PI_3"
    double _PI_4 "PI_4"
    double _SQRTPI "SQRTPI"
    double _SQRTTWOPI "SQRTTWOPI"
    double _INVSQRTPI "INVSQRTPI"
    double _INVSQRTTWOPI "INVSQRTTWOPI"
    double _GAMMA "GAMMA"
    double _SQRT2 "SQRT2"
    double _SQRT3 "SQRT3"
    double _SQRT6 "SQRT6"
    double _INVSQRT2 "INVSQRT2"
    double _INVSQRT3 "INVSQRT3"
    double _INVSQRT6 "INVSQRT6"

    double _C_SI "C_SI"
    double _G_SI "G_SI"
    double _MSUN_SI "MSUN_SI"
    double _MTSUN_SI "MTSUN_SI"
    double _MRSUN_SI "MRSUN_SI"
    double _PC_SI "PC_SI"
    double _AU_SI "AU_SI"
    double _YRSID_SI "YRSID_SI"

    double _EarthOrbitOmega_SI "EarthOrbitOmega_SI"

    double _EPOCH_J2000_0_TAI_UTC "EPOCH_J2000_0_TAI_UTC"
    double _EPOCH_J2000_0_GPS "EPOCH_J2000_0_GPS"
    double _EPOCH_LISA_0_GPS "EPOCH_LISA_0_GPS"
