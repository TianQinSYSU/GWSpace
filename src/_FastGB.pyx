#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: _FastGB.pyx
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 19:39:21
#==================================

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
cimport cython

cdef extern from "GB.h":
    void Fast_GB(double* pars, long N, double Tobs, double dt,
                 double* XLS, double* YLS, double* ZLS,
                 double* XSL, double* YSL, double* ZSL, 
                 int NP, char detector[])

cdef extern from "spacecrafts.h":
    void spacecraft_LISA(double t, double *x, double *y, double *z);
    void spacecraft_TaiJi(double t, double *x, double *y, double *z);
    void spacecraft_TianQin(double t, double *x, double *y, double *z);


# Define the Cython functions
#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef ComputeXYZ_FD(double[:] params, long N, double Tobs, double dt,
        double[:] XLS, double[:] YLS, double[:] ZLS, 
        double[:] XSL, double[:] YSL, double[:] ZSL, 
        int NP, str detector):

    if (NP != 8):
        print("Only 8 parameters are accepted: %d"%NP)
        return

    cdef double f0, df0, lat, lng, Amp, incl, psi, phi0
    cdef double* pars = <double*>malloc(NP * sizeof(double))
    
    f0 = params[0]
    df0 = params[1]
    lat = params[2]
    lng = params[3]
    Amp = params[4]
    incl = params[5]
    psi = params[6]
    phi0 = params[7]
    
    pars[0] = f0 * Tobs
    pars[1] = np.cos(np.pi/2. - lat)
    pars[2] = lng
    pars[3] = np.log(Amp)
    pars[4] = np.cos(incl)
    pars[5] = psi
    pars[6] = phi0
    pars[7] = df0 * Tobs * Tobs

    #cdef double* c_pars = &paras[0]
    cdef double* c_XLS = &XLS[0]
    cdef double* c_YLS = &YLS[0]
    cdef double* c_ZLS = &ZLS[0]
    cdef double* c_XSL = &XSL[0]
    cdef double* c_YSL = &YSL[0]
    cdef double* c_ZSL = &ZSL[0]

    #cdef char* c_detector = &detector[0]

    Fast_GB(pars, N, Tobs, dt, c_XLS, c_YLS, c_ZLS, c_XSL, c_YSL, c_ZSL, NP, detector.encode('utf-8'))

    free(pars)
    return

cpdef Compute_position(double t, double[:] x, double[:] y, double[:] z, str detector):
    cdef double* c_x = &x[0]
    cdef double* c_y = &y[0]
    cdef double* c_z = &z[0]

    if detector == "LISA" or "lisa":
        spacecraft_LISA(t, c_x, c_y, c_z)
    elif detector == "TianQin" or "TQ":
        spacecraft_TianQin(t, c_x, c_y, c_z)
    elif detector == "TaiJi" or "taiji":
        spacecraft_TaiJi(t, c_x, c_y, c_z)

    return
