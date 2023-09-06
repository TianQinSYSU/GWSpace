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

cdef extern from "GB.h":
    void Fast_GB(double *params, long N, double Tobs, double dt, 
            double *XLS, double *YLS, double *ZLS,
            double* XSL, double* YSL, double* ZSL, 
            int NP);

def call_Fast_GB(params, N, Tobs, dt):
    cdef double *c_params = <double *> malloc(len(params) * sizeof(double))
    cdef double *XLS = <double *> malloc(N * sizeof(double))
    cdef double *YLS = <double *> malloc(N * sizeof(double))
    cdef double *ZLS = <double *> malloc(N * sizeof(double))
    cdef double *XSL = <double *> malloc(N * sizeof(double))
    cdef double *YSL = <double *> malloc(N * sizeof(double))
    cdef double *ZSL = <double *> malloc(N * sizeof(double))
    #cdef double *XLS = NULL
    #cdef double *YLS = NULL
    #cdef double *ZLS = NULL
    #cdef double *XSL = NULL
    #cdef double *YSL = NULL
    #cdef double *ZSL = NULL

    for i in range(len(params)):
        c_params[i] = params[i]

    Fast_GB(c_params, N, Tobs, dt, XLS, YLS, ZLS, XSL, YSL, ZSL, len(params))

    result_XLS = [XLS[i] for i in range(N)]
    result_YLS = [YLS[i] for i in range(N)]
    result_ZLS = [ZLS[i] for i in range(N)]
    result_XSL = [XSL[i] for i in range(N)]
    result_YSL = [YSL[i] for i in range(N)]
    result_ZSL = [ZSL[i] for i in range(N)]

    free(c_params)
    free(XLS)
    free(YLS)
    free(ZLS)
    free(XSL)
    free(YSL)
    free(ZSL)

    return result_XLS, result_YLS, result_ZLS, result_XSL, result_YSL, result_ZSL

