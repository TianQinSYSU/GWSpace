# coding: utf-8
# 2022 Han Wang
# wangh657@mail2.sysu.edu.cn

"""Functions used to connect the EccFD library,
 which is automatically compiled during the `gwspace` installation. """
import os
from numpy import float64, complex128, frombuffer, pi, array, zeros
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from ctypes import cdll, Structure, POINTER, byref, c_double, c_size_t, c_uint, cast
import glob

_dirname = os.path.dirname(__file__)
if _dirname == '':
    _dirname = '.'

try:
    _rlib = cdll.LoadLibrary(_dirname+"/libEccFD.so")
except OSError:
    so_file = glob.glob(_dirname + "/libEccFD*.so")
    _rlib = cdll.LoadLibrary(so_file[0])


class _EccFDWaveform(Structure):
    """Note: The 'data' actually should be POINTER(c_complex), but ctypes do not have that,
    so we finally use buffer to restore the data, then any types of number in POINTER() is OK.
    Additional Note: Now we are using numpy `ndarray.view` here, so POINTER(c_double) is required."""
    _fields_ = [("data_p", POINTER(c_double)),  # complex double
                ("data_c", POINTER(c_double)),  # complex double
                ("deltaF", c_double),
                ("length", c_size_t),
                ]


def gen_ecc_fd_waveform(mass1, mass2, eccentricity, distance,
                        coa_phase=0., inclination=0., long_asc_nodes=0.,
                        delta_f=None, f_lower=None, f_final=0., obs_time=0.):
    """Note: Thanks to https://stackoverflow.com/questions/5658047"""
    f = _rlib.SimInspiralEccentricFD
    htilde = POINTER(_EccFDWaveform)()
    # **htilde, phiRef, deltaF, m1_SI, m2_SI, fStart, fEnd, i, r, inclination_azimuth, e_min
    f.argtypes = [POINTER(POINTER(_EccFDWaveform)),
                  c_double, c_double, c_double, c_double, c_double,
                  c_double, c_double, c_double, c_double, c_double, c_double]
    _ = f(byref(htilde), coa_phase, delta_f, mass1, mass2,
          f_lower, f_final, inclination, distance, long_asc_nodes, eccentricity, obs_time)
    length = htilde.contents.length*2
    hp_, hc_ = (_arr_from_buffer(htilde.contents.data_p, length),
                _arr_from_buffer(htilde.contents.data_c, length))
    _rlib.DestroyComplex16FDWaveform(htilde)
    return hp_.view(complex128), hc_.view(complex128)


class _EccFDAmpPhase(Structure):
    _fields_ = [("amp_p", POINTER(c_double)),  # complex double
                ("amp_c", POINTER(c_double)),  # complex double
                ("phase", POINTER(c_double)),
                ("deltaF", c_double),
                ("length", c_size_t),
                ("harmonic", c_uint),
                ]


def gen_ecc_fd_amp_phase(mass1, mass2, eccentricity, distance,
                         coa_phase=0., inclination=0., long_asc_nodes=0.,
                         delta_f=None, f_lower=None, f_final=0., obs_time=0.):
    f = _rlib.SimInspiralEccentricFDAmpPhase
    h_amp_phase = POINTER(POINTER(_EccFDAmpPhase))()
    # ***h_amp_phase, phiRef, deltaF, m1_SI, m2_SI, fStart, fEnd, i, r, inclination_azimuth, e_min
    f.argtypes = [POINTER(POINTER(POINTER(_EccFDAmpPhase))),
                  c_double, c_double, c_double, c_double, c_double,
                  c_double, c_double, c_double, c_double, c_double, c_double]
    _ = f(byref(h_amp_phase), coa_phase, delta_f, mass1, mass2,
          f_lower, f_final, inclination, distance, long_asc_nodes, eccentricity, obs_time)
    list_of_h = h_amp_phase[:10]
    length = list_of_h[0].contents.length
    amp_p_c_phase = tuple((_arr_from_buffer(list_of_h[j].contents.amp_p, length*2).view(complex128),
                           _arr_from_buffer(list_of_h[j].contents.amp_c, length*2).view(complex128),
                           _arr_from_buffer(list_of_h[j].contents.phase, length)) for j in range(10))
    [_rlib.DestroyAmpPhaseFDWaveform(h) for h in list_of_h]
    return amp_p_c_phase


def gen_ecc_fd_and_phase(mass1, mass2, eccentricity, distance,
                         coa_phase=0., inclination=0., long_asc_nodes=0.,
                         delta_f=None, f_lower=None, f_final=0., obs_time=0.):
    h_and_phase = POINTER(POINTER(_EccFDAmpPhase))()
    f = _rlib.SimInspiralEccentricFDAndPhase
    # ***h_and_phase, phiRef, deltaF, m1_SI, m2_SI, fStart, fEnd, i, r, inclination_azimuth, e_min
    f.argtypes = [POINTER(POINTER(POINTER(_EccFDAmpPhase))),
                  c_double, c_double, c_double, c_double, c_double,
                  c_double, c_double, c_double, c_double, c_double, c_double]
    _ = f(byref(h_and_phase), coa_phase, delta_f, mass1, mass2,
          f_lower, f_final, inclination, distance, long_asc_nodes, eccentricity, obs_time)
    list_h = h_and_phase[:10]
    length = list_h[0].contents.length
    h_p_c = tuple((_arr_from_buffer(list_h[j].contents.amp_p, length*2).view(complex128),
                   _arr_from_buffer(list_h[j].contents.amp_c, length*2).view(complex128)) for j in range(10))
    phase2 = _arr_from_buffer(list_h[1].contents.phase, length)  # only need phase for j=2
    [_rlib.DestroyAmpPhaseFDWaveform(h) for h in list_h]
    return h_p_c, phase2


def _arr_from_buffer(p, length):
    """https://stackoverflow.com/questions/7543675
      frombuffer is faster than fromiter because it creates array without copying
     https://stackoverflow.com/questions/4355524
      The copy() is used for the np.ndarray to acquire ownership,
      then you can safely free pointers to avoid memory leaks."""
    return frombuffer(cast(p, POINTER(c_double*length)).contents, float64).copy()


def gen_ecc_fd_and_tf(tc=0., **kwargs):
    wf, phase = gen_ecc_fd_and_phase(**kwargs)
    length = len(phase)
    freq = kwargs['delta_f'] * array(range(length))
    # use the harmonic j=2 to calculate t_f, and rescale it for different harmonics
    index = (phase != 0).argmax()
    phase = phase[index:]
    tf_spline = Spline(freq[index:], -1 / (2 * pi) * (phase - phase[0])).derivative()
    wf_tf = []

    for i in range(10):
        tf_vec = zeros(shape=(length, ), dtype=float64)
        h_p, h_c = wf[i]
        index = (h_p != 0).argmax()
        tf_vec[index:] = tf_spline(freq[index:]) * ((i + 1) / 2) ** (8 / 3) + tc
        wf_tf.append((h_p, h_c, tf_vec))
    return tuple(wf_tf), freq
