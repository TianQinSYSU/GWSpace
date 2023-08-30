#
# Copyright (C) 2019 Sylvain Marsat.
#
#


"""
    Definitions for structures
"""

from __future__ import print_function

import numpy as np
cimport numpy as np

cdef extern from "struct.h":

    ctypedef struct real_vector:
        double* data;
        size_t size;

    ctypedef struct complex_vector:
        double complex* data;
        size_t size;

    ctypedef struct real_matrix:
        double* data;
        size_t size1;
        size_t size2;

    ctypedef struct complex_array_3d:
        double complex* data;
        size_t size1;
        size_t size2;
        size_t size3;

    real_vector* real_vector_alloc(
        size_t size
    );
    real_vector* real_vector_view(
        double* data,
        size_t size
    );

    void real_vector_free(real_vector* v);
    void real_vector_view_free(real_vector* v)

    complex_vector* complex_vector_alloc(
        size_t size
    );
    complex_vector* complex_vector_view(
        double complex* data,
        size_t size
    )nogil

    void complex_vector_free(complex_vector* v);
    void complex_vector_view_free(complex_vector* v)nogil

    real_matrix* real_matrix_alloc(size_t size1, size_t size2);
    real_matrix* real_matrix_view(
        double* data,
        size_t size1,
        size_t size2
    );
    void real_matrix_free(real_matrix* m);
    void real_matrix_view_free(real_matrix* v);

    complex_array_3d* complex_array_3d_alloc(size_t size1, size_t size2, size_t size3);
    complex_array_3d* complex_array_3d_view(
        double complex* data,
        size_t size1,
        size_t size2,
        size_t size3
    );
    void complex_array_3d_free(complex_array_3d* a);
    void complex_array_3d_view_free(complex_array_3d* v);

    ctypedef struct CAmpPhaseFDData:
        real_vector* freq;
        real_vector* amp_real;
        real_vector* amp_imag;
        real_vector* phase;

    ctypedef struct CAmpPhaseFDSpline:
        real_matrix* spline_amp_real;
        real_matrix* spline_amp_imag;
        real_matrix* spline_phase;

    ctypedef struct AmpPhaseFDMode:
        real_vector* freq_amp;
        real_vector* amp_real;
        real_vector* amp_imag;
        real_vector* freq_phase;
        real_vector* phase;
        real_vector* tf;

    ctypedef struct AmpPhaseFDWaveform:
        double* freq;
        double* amp;
        double* phase;
        size_t length;

    AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(
        size_t length
    );

    void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform* wf);

    int CAmpPhaseFDData_Init(
      CAmpPhaseFDData** h,
      size_t length
    );
    void CAmpPhaseFDData_Destroy(CAmpPhaseFDData* h);
    # CAmpPhaseFDData* CAmpPhaseFDData_view(
    #   real_vector* freq,
    #   real_vector* amp_real,
    #   real_vector* amp_imag,
    #   real_vector* phase
    # );
    # void CAmpPhaseFDData_view_free(CAmpPhaseFDData* h);

    int CAmpPhaseFDSpline_Init(
      CAmpPhaseFDSpline** h,
      size_t length
    );
    void CAmpPhaseFDSpline_Destroy(CAmpPhaseFDSpline* h);

    # Cast C double arrays to numpy via a MemoryView
    # cdef inline real_vector_to_np_array(real_vector* vec):
    #     cdef int n = vec.size
    #     cdef double[::1] view_vec = <(double)[:n]> vec.data
    #     return np.asarray(view_vec)
    # cdef inline real_matrix_to_np_array(real_matrix* mat):
    #     cdef int m = mat.size1
    #     cdef int n = mat.size2
    #     cdef double[::1] view_mat = <(double)[:(m * n)]> mat.data
    #     return np.reshape(np.asarray(view_mat), (m, n))

    # Build a real_vector view of a numpy array data
    # cdef inline real_vector* np_array_to_real_vector(arr):
    #     cdef real_vector* vec = NULL
    #     cdef double[::1] arr_data = arr.data
    #     vec = real_vector_view(&arr_data[0], arr.shape[0])
    #     return vec
    # NOTE: seems problematic, see NOTES
    # cdef inline real_matrix* np_array_to_real_matrix(arr):
    #     cdef real_matrix* mat = NULL
    #     cdef double[::1] arr_data = arr.data
    #     mat = real_matrix_view(&arr_data[0], arr.shape[0], arr.shape[1])
    #     return mat

    # Structure and function for generic function references

    ctypedef struct object_function:
        const void* object;
        double (*function)(const void* object, double);

    double call_object_function(object_function fn, double x)

cdef class py_object_function:
    cdef object_function obfn
    cpdef double call(self, double f)
    cpdef np.ndarray[np.float_t, ndim=1] apply(self, np.ndarray[np.float_t, ndim=1] freq)
