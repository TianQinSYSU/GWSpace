import numpy as np
cimport numpy as np

cdef class py_object_function:

    cpdef double call(self, double f):
        return call_object_function(self.obfn, f)

    cpdef np.ndarray[np.float_t, ndim=1] apply(self, np.ndarray[np.float_t, ndim=1] xs):
        cdef np.ndarray[np.float_t, ndim=1] ys = np.zeros_like(xs)
        for i in range(len(xs)):
            ys[i] = call_object_function(self.obfn, xs[i])
        return ys
