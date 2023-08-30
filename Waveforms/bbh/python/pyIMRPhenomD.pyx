#
# Copyright (C) 2019 Sylvain Marsat.
#
#


"""
    Standalone IMRPhenomD inspiral-merger-ringdown GW waveform model
    for binary black hole coalescences.
"""

import numpy as np
cimport numpy as np
from math import cos

from pyconstants cimport *
from pystruct cimport *

# cdef extern from "IMRPhenomD_internals.h":
#
#     ctypedef struct AmpPhaseFDWaveform:
#         double* freq;
#         double* amp;
#         double* phase;
#         size_t length;
#
#     AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(
#         size_t length
#     );
#
#     void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform* wf);
#
#     ctypedef struct RealVector:
#         double* data;
#         size_t length;
#
#     RealVector* CreateRealVector(
#         size_t length
#     );
#
#     void DestroyRealVector(RealVector* v);

cdef extern from "IMRPhenomD_internals.h":
    # cdef double _MSUN_SI "MSUN_SI"
    # cdef double _MTSUN_SI "MTSUN_SI"
    # cdef double _MRSUN_SI "MRSUN_SI"
    # cdef double _PC_SI "PC_SI"

    ctypedef struct COMPLEX16FrequencySeries:
        double complex *data;
        char *name;
        long epoch;
        double f0;
        double deltaF;
        size_t length;

    COMPLEX16FrequencySeries *CreateCOMPLEX16FrequencySeries(
        const char *name,
        long epoch,
        double f0,
        double deltaF,
        size_t length
    );

    COMPLEX16FrequencySeries *ResizeCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs, size_t length);
    void DestroyCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs);

cdef extern from "IMRPhenomD_internals.h":
    ctypedef struct ModGRParams:
        double alpha;
    ctypedef struct ExtraParams:
        int use_buggy_LAL_tpeak;
        # int scale_mode_freq;

# # Expose some useful constants hidden in C-space
# MSUN_SI = _MSUN_SI
# MTSUN_SI = _MTSUN_SI
# MRSUN_SI = _MRSUN_SI
# PC_SI = _PC_SI

cdef extern from "IMRPhenomD.h":
    int IMRPhenomDGenerateFD(
        COMPLEX16FrequencySeries **htilde, # [out] FD waveform
        const double phi0,                 # Orbital phase at fRef (rad)
        const double fRef_in,              # reference frequency (Hz)
        const double deltaF,               # Sampling frequency (Hz)
        const double m1_SI,                # Mass of companion 1 (kg)
        const double m2_SI,                # Mass of companion 2 (kg)
        const double chi1,                 # Aligned-spin parameter of companion 1
        const double chi2,                 # Aligned-spin parameter of companion 2
        const double f_min,                # Starting GW frequency (Hz)
        const double f_max,                # End frequency; 0 defaults to Mf = \ref f_CUT
        const double distance              # Distance of source (m)
    );

cdef extern from "IMRPhenomD.h":
    int IMRPhenomDGenerateh22FDAmpPhase(
        AmpPhaseFDWaveform** h22,           # [out] FD waveform
        real_vector** tf,                   # [out] tf from analytic derivative of the phase
        double* fpeak,                      # [out] Approximate peak frequency (Hz)
        double* tpeak,                      # [out] tf at peak frequency (s)
        double* phipeak,                    # [out] phase 22 at peak frequency
        double* fstart,                     # [out] Starting frequency (Hz)
        double* tstart,                     # [out] tf at starting frequency (s)
        double* phistart,                   # [out] phase 22 at starting frequency
        real_vector* freq,                  # Input: frequencies (Hz) on which to evaluate h22 FD - will be copied in the output AmpPhaseFDWaveform. Frequencies exceeding max freq covered by PhenomD will be given 0 amplitude and phase.
        const double m1,                    # Mass of companion 1 (solar masses)
        const double m2,                    # Mass of companion 2 (solar masses)
        const double chi1,                  # Aligned-spin parameter of companion 1
        const double chi2,                  # Aligned-spin parameter of companion 2
        const double distance,              # Distance of source (Mpc)
        const double tRef,                  # Time at fRef_for_tRef (s)
        const double phiRef,                # Orbital phase at fRef_for_phiRef (rad)
        const double fRef_for_tRef_in,      # Ref. frequency (Hz) for tRef
        const double fRef_for_phiRef_in,    # Ref. frequency (Hz) for phiRef
        const int force_phiRef_fRef,        # Flag to force phiRef at fRef after adjusting tRef
        const double Deltat,                # Time shift (s) applied a posteriori
        const ExtraParams* extraparams,     # Additional parameters
        const ModGRParams* modgrparams      # Modified GR parameters
    );

cdef extern from "IMRPhenomD.h":
    double IMRPhenomDComputeTimeOfFrequency(
        const double f,                     # Input frequency (Hz): we compute t(f)
        const double m1,                    # Mass of companion 1 (solar masses)
        const double m2,                    # Mass of companion 2 (solar masses)
        const double chi1,                  # Aligned-spin parameter of companion 1
        const double chi2,                  # Aligned-spin parameter of companion 2
        const double distance,              # Distance of source (Mpc)
        const double tRef,                  # Time at fRef_for_tRef (s)
        const double phiRef,                # Orbital phase at fRef_for_phiRef (rad)
        const double fRef_for_tRef_in,      # Ref. frequency (Hz) for tRef
        const double fRef_for_phiRef_in,    # Ref. frequency (Hz) for phiRef
        const int force_phiRef_fRef,        # Flag to force phiRef at fRef after adjusting tRef
        const double Deltat,                # Time shift (s) applied a posteriori
        const ExtraParams* extraparams,     # Additional parameters
        const ModGRParams* modgrparams      # Modified GR parameters
    );
    double IMRPhenomDComputeInverseFrequencyOfTime(
        const double t,                     # Input time (s): we compute f(t)
        const double f_estimate,            # Estimate of f(t), use to initialize
        const double t_acc,                 # Target accuracy of t(f) where to stop refining f
        const double m1,                    # Mass of companion 1 (solar masses)
        const double m2,                    # Mass of companion 2 (solar masses)
        const double chi1,                  # Aligned-spin parameter of companion 1
        const double chi2,                  # Aligned-spin parameter of companion 2
        const double distance,              # Distance of source (Mpc)
        const double tRef,                  # Time at fRef_for_tRef (s)
        const double phiRef,                # Orbital phase at fRef_for_phiRef (rad)
        const double fRef_for_tRef_in,      # Ref. frequency (Hz) for tRef
        const double fRef_for_phiRef_in,    # Ref. frequency (Hz) for phiRef
        const int force_phiRef_fRef,        # Flag to force phiRef at fRef after adjusting tRef
        const double Deltat,                # Time shift (s) applied a posteriori
        const int max_ite,                  # Maximal number of iterations in bisection
        const ExtraParams* extraparams,     # Additional parameters
        const ModGRParams* modgrparams      # Modified GR parameters
    );


cdef extern from "struct.h":
    int ERROR_EINVAL
    int ERROR_EDOM
    int ERROR_EFAULT
    int ERROR_EFUNC
    int ERROR_ENOMEM

cpdef double check_error(double value):
    if value in [ERROR_EINVAL, ERROR_EDOM, ERROR_EFAULT, ERROR_EFUNC, ERROR_ENOMEM]:
        raise ValueError
    return value

cdef class IMRPhenomD:
    """ Generate IMRPhenomD inspiral-merger-ringdown frequency-domain waveform.
    """

    cdef COMPLEX16FrequencySeries *htilde # pointer to waveform structure

    cdef phi0                 # Orbital phase at fRef (rad)
    cdef fRef                 # Reference frequency (Hz)
    cdef deltaF               # Sampling frequency (Hz)
    cdef m1_SI                # Mass of companion 1 (kg)
    cdef m2_SI                # Mass of companion 2 (kg)
    cdef chi1                 # Aligned-spin parameter of companion 1
    cdef chi2                 # Aligned-spin parameter of companion 2
    cdef f_min                # Starting GW frequency (Hz)
    cdef f_max                # End frequency; 0 defaults to Mf = \ref f_CUT
    cdef distance             # Distance of source (m)
    cdef name, length, f0
    cdef fHz, hptilde, hctilde


    def __init__(self, phi0, fRef, deltaF, m1_SI, m2_SI, chi1, chi2, f_min, f_max, distance, inclination):
        """Constructor
        Arguments:
          phi0                  # Orbital phase at fRef (rad)
          fRef_in               # Reference frequency (Hz)
          deltaF                # Sampling frequency (Hz)
          m1_SI                 # Mass of companion 1 (kg)
          m2_SI                 # Mass of companion 2 (kg)
          chi1                  # Aligned-spin parameter of companion 1
          chi2                  # Aligned-spin parameter of companion 2
          f_min                 # Starting GW frequency (Hz)
          f_max                 # End frequency; 0 defaults to Mf = \ref f_CUT
          distance              # Distance of source (m)
          inclination           # Inclination of source (rad)
        """
        # arguments are checked in the C waveform generator
        self.phi0 = phi0
        self.fRef = fRef
        self.deltaF = deltaF
        self.m1_SI = m1_SI
        self.m2_SI = m2_SI
        self.chi1 = chi1
        self.chi2 = chi2
        self.f_min = f_min
        self.f_max = f_max
        self.distance = distance

        self.htilde = NULL

        ret = IMRPhenomDGenerateFD(
            &self.htilde,
            self.phi0,
            self.fRef,
            self.deltaF,
            self.m1_SI,
            self.m2_SI,
            self.chi1,
            self.chi2,
            self.f_min,
            self.f_max,
            self.distance
        );
        if ret == _FAILURE:
            raise ValueError("Call to IMRPhenomDGenerateFD() failed.")

        # Store waveform properties
        self.name = self.htilde.name
        self.length = self.htilde.length
        self.f0 = self.htilde.f0
        #self.deltaF = self.htilde.deltaF
        self.fHz = self.f0 + np.arange(self.length) * self.deltaF

        # Direct copy of C double complex array to numpy via a MemoryView
        cdef double complex[::1] view = <(double complex)[:self.htilde.length]> self.htilde.data
        hptilde = np.asarray(view) # we multiply with the necessary factor below to make this \tilde h_+(f)

        # Produce both polarizations
        cfac = cos(inclination)
        pfac = 0.5 * (1.0 + cfac**2)
        hctilde = np.zeros_like(hptilde)
        hctilde = - 1j * cfac * hptilde
        hptilde *= pfac
        self.hptilde = hptilde
        self.hctilde = hctilde

    def __dealloc__(self):
        """Destructor
        """
        if self.htilde != NULL:
            DestroyCOMPLEX16FrequencySeries(self.htilde)

    def get_waveform(self):
        return np.copy(self.fHz), np.copy(self.hptilde), np.copy(self.hctilde)

# cdef RealVector* ConvertNumpyArrayToRealVector(arr):
#     cdef int n
#     cdef RealVector* vec
#     cdef double* vecdata
#     n = len(arr)
#     vec = CreateRealVector(n)
#     vecdata = vec.data
#     for i in range(n):
#         vecdata[i] = arr[i]
#     return vec


cdef class IMRPhenomDh22AmpPhase:
    """ Generate IMRPhenomD inspiral-merger-ringdown frequency-domain waveform, in Amp/Phase form for h22.
    """

    cdef AmpPhaseFDWaveform* h22 # pointer to waveform structure
    cdef real_vector* Ctf         # pointer to waveform structure
    cdef double fpeak            # Approximate peak frequency (Hz)
    cdef double tpeak            # tf at peak frequency (s)
    cdef double phipeak          # phase 22 at peak frequency
    cdef double fstart           # starting frequency (Hz)
    cdef double tstart           # tf at starting frequency (s)
    cdef double phistart         # phase 22 at starting frequency
    cdef real_vector* Cfreq      # pointer to input frequencies in C structure

    cdef m1                   # Mass of companion 1 (solar masses)
    cdef m2                   # Mass of companion 2 (solar masses)
    cdef chi1                 # Aligned-spin parameter of companion 1 in [-1, 1]
    cdef chi2                 # Aligned-spin parameter of companion 2 in [-1, 1]
    cdef dist                 # Distance of source (Mpc)
    cdef tref                 # Time at fref_for_tref (s)
    cdef phiref               # Orbital phase at fref_for_phiref (rad)
    cdef fref_for_tref        # ref. frequency (Hz) for tref
    cdef fref_for_phiref      # ref. frequency (Hz) for phiref
    cdef force_phiref_fref    # Flag forcing phiref at fref after adjusting tref
    cdef Deltat               # Time shift (s) applied a posteriori
    cdef f_min, f_max
    cdef length
    cdef freq, amp, phase, tf

    cdef ModGRParams* Cmodgrparams # pointer to struct with mod GR params
    cdef ModGRParams Cmodgrparams_struct

    cdef ExtraParams* Cextraparams # pointer to struct with extra params
    cdef ExtraParams Cextraparams_struct

    cdef public object modes  # List of modes (l,m)
    cdef hlm                  # Dictionary of modes
    modes_PhD = [(2,2)]

    def __init__(self,
                 np.ndarray[np.float_t, ndim=1] freq,
                 m1, m2, chi1, chi2, dist,
                 tref=0., phiref=0., fref_for_tref=0., fref_for_phiref=0.,
                 force_phiref_fref=True, Deltat=None,
                 modes=modes_PhD,
                 extra_params=None, mod_gr_params=None):
        """Constructor
        Args:
          freq              # Frequencies (Hz) on which to evaluate the waveform - numpy 1D array
          m1                # Mass of companion 1 (solar masses)
          m2                # Mass of companion 2 (solar masses)
          chi1              # Aligned-spin parameter of companion 1 in [-1,1]
          chi2              # Aligned-spin parameter of companion 2 in [-1,1]
          dist              # Distance of source (Mpc)
        Keyword args:
          tref                # Time at fref_for_tref (s)
          phiref              # Orbital phase at fref_for_phiref (rad)
          fref_for_tref       # ref. frequency (Hz) for tref
          fref_for_phiref     # ref. frequency (Hz) for phiref
          force_phiref_fref   # Flag forcing phiref at fref after adjusting tref
          Deltat              # Time shift (s) applied a posteriori
          modes               # List of modes (l,m) generated - here only (2,2)
          extra_params        # Dictionary of additional parameters
          |- use_buggy_LAL_tpeak # Reproduce bug in tpeak alignment in LAL
          mod_gr_params       # Dictionary of modified GR parameters
        """
        # arguments are checked in the C waveform generator
        self.freq = freq
        self.fpeak = 0.
        self.tpeak = 0.
        self.phipeak = 0.
        self.fstart = 0.
        self.tstart = 0.
        self.phistart = 0.
        self.m1 = m1
        self.m2 = m2
        self.chi1 = chi1
        self.chi2 = chi2
        self.dist = dist
        self.tref = tref
        self.phiref = phiref
        self.fref_for_tref = fref_for_tref
        self.fref_for_phiref = fref_for_phiref
        self.force_phiref_fref = force_phiref_fref
        if Deltat is None:
            Deltat = 0.
        self.Deltat = Deltat
        self.f_min = freq[0]
        self.f_max = freq[-1]
        self.length = len(freq)
        self.modes = modes

        self.h22 = NULL
        self.Ctf = NULL

        # Check input modes
        if not self.modes==[(2,2)]:
            raise ValueError('Only mode (2,2) is available.')

        # Build C structure holding additional params from input python dict
        # if input is None, C struct will stay NULL and code will use default
        self.Cextraparams = NULL
        if extra_params is not None:
            self.Cextraparams_struct.use_buggy_LAL_tpeak = \
                                  extra_params.get('use_buggy_LAL_tpeak', False)
            self.Cextraparams = &self.Cextraparams_struct

        # Build C structure holding modified GR params from input python dict
        # if input is None, C struct will stay NULL and be ignored in C code
        self.Cmodgrparams = NULL
        if mod_gr_params is not None:
            self.Cmodgrparams_struct.alpha = mod_gr_params['alpha'] #PLACEHOLDER
            self.Cmodgrparams = &self.Cmodgrparams_struct

        # Build a real_vector representation of the input numpy array data
        if not freq.flags['C_CONTIGUOUS']:
            raise ValueError('Input numpy array freq is not C_CONTIGUOUS')
        cdef double* freq_data = <double *> &freq[0]
        self.Cfreq = real_vector_view(freq_data, freq.shape[0])

        cdef int Cforce_phiref_fref = <int> self.force_phiref_fref

        ret = IMRPhenomDGenerateh22FDAmpPhase(
            &self.h22,
            &self.Ctf,
            &self.fpeak,
            &self.tpeak,
            &self.phipeak,
            &self.fstart,
            &self.tstart,
            &self.phistart,
            self.Cfreq,
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            self.dist,
            self.tref,
            self.phiref,
            self.fref_for_tref,
            self.fref_for_phiref,
            Cforce_phiref_fref,
            self.Deltat,
            self.Cextraparams,
            self.Cmodgrparams
        );
        if ret != 0:
            raise ValueError("Call to IMRPhenomDGenerateFD() failed.")

        # Direct copy of C double array to numpy via a MemoryView
        cdef double[::1] view_amp = <(double)[:self.h22.length]> self.h22.amp
        cdef double[::1] view_phase = \
                                  <(double)[:self.h22.length]> self.h22.phase
        cdef double[::1] view_tf = <(double)[:self.Ctf.size]> self.Ctf.data
        self.amp = np.asarray(view_amp)
        self.phase = np.asarray(view_phase)
        self.tf = np.asarray(view_tf)

    def __dealloc__(self):
        """Destructor
        """
        if self.h22 != NULL:
            DestroyAmpPhaseFDWaveform(self.h22)
        if self.Cfreq != NULL:
            real_vector_view_free(self.Cfreq)
        if self.Ctf != NULL:
            real_vector_free(self.Ctf)

    def get_waveform(self):
        hlm = {}
        lm = (2,2)
        hlm[lm] = {}
        hlm[lm]['freq'] = np.copy(self.freq)
        hlm[lm]['amp'] = np.copy(self.amp)
        hlm[lm]['phase'] = np.copy(self.phase)
        hlm[lm]['tf'] = np.copy(self.tf)
        return hlm

    def get_fpeak(self):
        return self.fpeak

    def get_tpeak(self):
        return self.tpeak

    def get_phipeak(self):
        return self.phipeak

    def get_fstart(self):
        return self.fstart

    def get_tstart(self):
        return self.tstart

    def get_phistart(self):
        return self.phistart

    def compute_toff(self, f):
        cdef int Cforce_phiref_fref = <int> self.force_phiref_fref
        t = IMRPhenomDComputeTimeOfFrequency(
            f,
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            self.dist,
            self.tref,
            self.phiref,
            self.fref_for_tref,
            self.fref_for_phiref,
            Cforce_phiref_fref,
            self.Deltat,
            self.Cextraparams,
            self.Cmodgrparams
        );
        return t

    def compute_foft(self, t, f_estimate, t_acc, max_iter=100):
        cdef int Cforce_phiref_fref = <int> self.force_phiref_fref
        cdef int Cmax_iter = <int> max_iter
        f = IMRPhenomDComputeInverseFrequencyOfTime(
            t,
            f_estimate,
            t_acc,
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            self.dist,
            self.tref,
            self.phiref,
            self.fref_for_tref,
            self.fref_for_phiref,
            Cforce_phiref_fref,
            self.Deltat,
            Cmax_iter,
            self.Cextraparams,
            self.Cmodgrparams
        );
        return check_error(f)
