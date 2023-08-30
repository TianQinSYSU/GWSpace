#
# Copyright (C) 2019 Sylvain Marsat.
#
#


"""
    Standalone IMRPhenomHM inspiral-merger-ringdown GW waveform model
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

cdef extern from "struct.h":

    ctypedef struct ListAmpPhaseFDMode:
        AmpPhaseFDMode*              hlm;
        int                          l;
        int                          m;
        ListAmpPhaseFDMode*          next;

    ListAmpPhaseFDMode* ListAmpPhaseFDMode_GetMode(
    	   ListAmpPhaseFDMode* list,  # List structure to get this mode from
    	   int l,                     # Mode number l
    	   int m);                    # Mode number m

    ListAmpPhaseFDMode* ListAmpPhaseFDMode_Destroy(
    	   ListAmpPhaseFDMode* list); # List structure to destroy

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

cdef extern from "IMRPhenomHM.h":
    int IMRPhenomHMGethlmModes(
        ListAmpPhaseFDMode **hlms,   # [out] list of modes, FD amp/phase
        double* fpeak,               # [out] Approximate 22 peak frequency (Hz)
        double* tpeak,               # [out] tf 22 at peak frequency (s)
        double* phipeak,             # [out] phase 22 at peak frequency
        double* fstart,              # [out] Starting frequency (Hz)
        double* tstart,              # [out] tf 22 at starting frequency (s)
        double* phistart,            # [out] phase 22 at starting frequency
        real_vector* freq_22,        # frequency vector for lm in Hz
        real_vector* freq_21,        # frequency vector for lm in Hz
        real_vector* freq_33,        # frequency vector for lm in Hz
        real_vector* freq_32,        # frequency vector for lm in Hz
        real_vector* freq_44,        # frequency vector for lm in Hz
        real_vector* freq_43,        # frequency vector for lm in Hz
        double m1_SI,                # primary mass [kg]
        double m2_SI,                # secondary mass [kg]
        double chi1z,                # aligned spin of primary
        double chi2z,                # aligned spin of secondary
        double distance,             # luminosity distance (Mpc)
        const double phiRef,         # orbital phase at f_ref
        double f_ref,                # reference GW frequency
        const double Deltat,         # Time shift (s) applied a posteriori
        const int scale_freq_hm,     # Scale mode freq by m/2 */
        const ExtraParams* extraparams, # Additional parameters
        const ModGRParams* modgrparams  # Modified GR parameters
    );
    int IMRPhenomHMComputeTimeOfFrequencyModeByMode(
        double* tf22,             # [out] value of t_22 (s)
        double* tf21,             # [out] value of t_21 (s)
        double* tf33,             # [out] value of t_33 (s)
        double* tf32,             # [out] value of t_32 (s)
        double* tf44,             # [out] value of t_44 (s)
        double* tf43,             # [out] value of t_43 (s)
        double f22,              # [in] value of f_22 (Hz)
        double f21,              # [in] value of f_21 (Hz)
        double f33,              # [in] value of f_33 (Hz)
        double f32,              # [in] value of f_32 (Hz)
        double f44,              # [in] value of f_44 (Hz)
        double f43,              # [in] value of f_43 (Hz)
        double m1,                   # primary mass [solar masses]
        double m2,                   # secondary mass [solar masses]
        double chi1z,                   # aligned spin of primary
        double chi2z,                   # aligned spin of secondary
        double distance,                # luminosity distance (Mpc)
        const double phiRef,            # orbital phase at f_ref
        const double fRef_in,                   # reference GW frequency
        const double Deltat,             # Time shift (s) applied a posteriori
        const ExtraParams* extraparams,           # Additional parameters
        const ModGRParams* modgrparams            # Modified GR parameters
    );
    int IMRPhenomHMComputeInverseFrequencyOfTimeModeByMode(
        double* f22,              # [out] value of f_22 (Hz)
        double* f21,              # [out] value of f_21 (Hz)
        double* f33,              # [out] value of f_33 (Hz)
        double* f32,              # [out] value of f_32 (Hz)
        double* f44,              # [out] value of f_44 (Hz)
        double* f43,              # [out] value of f_43 (Hz)
        double tf22,             # [in] value of t_22 (s)
        double tf21,             # [in] value of t_21 (s)
        double tf33,             # [in] value of t_33 (s)
        double tf32,             # [in] value of t_32 (s)
        double tf44,             # [in] value of t_44 (s)
        double tf43,             # [in] value of t_43 (s)
        double f22_estimate,     # [in] guess for the value of f22, will be scaled by m/2
        double t_acc,                 # Target accuracy of t(f) where to stop refining f
        double m1,                   # primary mass [solar masses]
        double m2,                   # secondary mass [solar masses]
        double chi1z,                   # aligned spin of primary
        double chi2z,                   # aligned spin of secondary
        double distance,                # luminosity distance (Mpc)
        const double phiRef,            # orbital phase at f_ref
        const double fRef_in,                   # reference GW frequency
        const double Deltat,             # Time shift (s) applied a posteriori
        const int max_iter,                 # Maximal number of iterations in bisection
        const ExtraParams* extraparams,           # Additional parameters
        const ModGRParams* modgrparams            # Modified GR parameters
    );

modes_PhenomHM = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

cdef class IMRPhenomHMhlmAmpPhase:
    """ Generate IMRPhenomHM inspiral-merger-ringdown frequency-domain waveform, in Amp/Phase form for hlms.
    """

    cdef ListAmpPhaseFDMode* listhlm # pointer to waveform list of modes
    cdef double fpeak            # Approximate peak frequency (Hz)
    cdef double tpeak            # tf at peak frequency (s)
    cdef double phipeak          # phase 22 at peak frequency
    cdef double fstart           # starting frequency (Hz)
    cdef double tstart           # tf at starting frequency (s)
    cdef double phistart         # phase 22 at starting frequency

    cdef real_vector* Cfreq_22   # pointer to input freqs lm in C structure
    cdef real_vector* Cfreq_21   # pointer to input freqs lm in C structure
    cdef real_vector* Cfreq_33   # pointer to input freqs lm in C structure
    cdef real_vector* Cfreq_32   # pointer to input freqs lm in C structure
    cdef real_vector* Cfreq_44   # pointer to input freqs lm in C structure
    cdef real_vector* Cfreq_43   # pointer to input freqs lm in C structure

    cdef m1                   # Mass of companion 1 (solar masses)
    cdef m2                   # Mass of companion 2 (solar masses)
    cdef chi1                 # Aligned-spin parameter of companion 1 in [-1, 1]
    cdef chi2                 # Aligned-spin parameter of companion 2 in [-1, 1]
    cdef dist                 # Distance of source (Mpc)
    # cdef tref                 # Time at fref_for_tref (s)
    cdef phiref               # Orbital phase at fref_for_phiref (rad)
    cdef fref                 # ref. frequency (Hz) for tref
    # cdef fref_for_tref        # ref. frequency (Hz) for tref
    # cdef fref_for_phiref      # ref. frequency (Hz) for phiref
    # cdef force_phiref_fref    # Flag forcing phiref at fref after adjusting tref
    cdef Deltat               # Time shift (s) applied a posteriori
    cdef f_min, f_max
    cdef length
    # cdef freq, amp, phase

    cdef ModGRParams* Cmodgrparams # pointer to struct with mod GR params
    cdef ModGRParams Cmodgrparams_struct

    cdef ExtraParams* Cextraparams # pointer to struct with extra params
    cdef ExtraParams Cextraparams_struct

    cdef public object modes  # List of modes (l,m)
    cdef hlm                  # Dictionary of modes
    cdef scale_freq_hm        # Scale mode freq by m/2

    # # Used for the f(t), t(f) functions
    # cdef double tf22            # Time of frequency for mode lm (s)
    # cdef double tf21            # Time of frequency for mode lm (s)
    # cdef double tf33            # Time of frequency for mode lm (s)
    # cdef double tf32            # Time of frequency for mode lm (s)
    # cdef double tf44            # Time of frequency for mode lm (s)
    # cdef double tf43            # Time of frequency for mode lm (s)
    # cdef double f22             # Frequency of time for mode lm (Hz)
    # cdef double f21             # Frequency of time for mode lm (Hz)
    # cdef double f33             # Frequency of time for mode lm (Hz)
    # cdef double f32             # Frequency of time for mode lm (Hz)
    # cdef double f44             # Frequency of time for mode lm (Hz)
    # cdef double f43             # Frequency of time for mode lm (Hz)

    # modes_PhHM = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

    def __init__(self,
                 freq,
                 m1, m2, chi1, chi2, dist,
                 phiref=0., fref=0., Deltat=0.,
                 modes=modes_PhenomHM, scale_freq_hm=True,
                 extra_params=None, mod_gr_params=None):
                 # tref=0., fref_for_tref=0., fref_for_phiref=0.,
                 # force_phiref_fref=True, Deltat=None):
        """Constructor
        Args:
          freq              # Frequencies (Hz) on which to evaluate waveform
                              If numpy 1D array, interpreted as f_22
                              higher modes are returned on frequencies m/2*freq
                              if scale_freq_hm is set to True
                              If dict, dict of np array for each f_lm
                              (NOTE: if dict, scale_freq_hm ignored)
          m1                # Mass of companion 1 (solar masses)
          m2                # Mass of companion 2 (solar masses)
          chi1              # Aligned-spin parameter of companion 1 in [-1,1]
          chi2              # Aligned-spin parameter of companion 2 in [-1,1]
          dist              # Luminosity distance of source (Mpc)
        Keyword args:
          fref                # ref. frequency (Hz)
          phiref              # Orbital phase at fref (rad)
          Deltat              # Time shift (s) applied a posteriori
          modes               # List of modes (l,m) to generate
          scale_freq_hm       # Scale mode freq by m/2 (default True)
                                if input freq is dict, ignored
          extra_params        # Dictionary of additional parameters
          |- use_buggy_LAL_tpeak # Reproduce bug in tpeak alignment in LAL
                                   (default False)
          mod_gr_params       # Dictionary of modified GR parameters (NOT ready yet)
        """
          # tref                # Time at fref_for_tref (s)
          # phiref              # Orbital phase at fref_for_phiref (rad)
          # fref_for_tref       # ref. frequency (Hz) for tref
          # fref_for_phiref     # ref. frequency (Hz) for phiref
          # force_phiref_fref   # Flag forcing phiref at fref after adjusting tref
          # Deltat              # Time shift (s) applied a posteriori

        # Input frequencies can be either a 1d np array for f_22
        # or a dict with a 1d np array for each f_lm
        # TODO: very clunky, need to find how to pass list structures to cython
        cdef np.ndarray[np.float_t, ndim=1] freq_22 = None # Frequencies for the mode lm
        cdef np.ndarray[np.float_t, ndim=1] freq_21 = None # Frequencies for the mode lm
        cdef np.ndarray[np.float_t, ndim=1] freq_33 = None # Frequencies for the mode lm
        cdef np.ndarray[np.float_t, ndim=1] freq_32 = None # Frequencies for the mode lm
        cdef np.ndarray[np.float_t, ndim=1] freq_44 = None # Frequencies for the mode lm
        cdef np.ndarray[np.float_t, ndim=1] freq_43 = None # Frequencies for the mode lm
        if isinstance(freq, np.ndarray):
            freq_22 = freq
        elif isinstance(freq, dict):
            scale_freq_hm = False # Giving explicit mode-by-mode freq supersedes this option
            freq_22 = freq.get((2,2), None)
            freq_21 = freq.get((2,1), None)
            freq_33 = freq.get((3,3), None)
            freq_32 = freq.get((3,2), None)
            freq_44 = freq.get((4,4), None)
            freq_43 = freq.get((4,3), None)
        else:
            raise ValueError('freq must be either a numpy array (for f_22)'\
                            +' or a dict of arrays for the f_lm.')

        # arguments are checked in the C waveform generator
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
        #self.tref = tref
        self.phiref = phiref
        self.fref = fref
        # self.fref_for_tref = fref_for_tref
        # self.fref_for_phiref = fref_for_phiref
        # self.force_phiref_fref = force_phiref_fref
        if Deltat is None:
            Deltat = 0.
        self.Deltat = Deltat
        self.f_min = freq_22[0]
        self.f_max = freq_22[-1]
        self.length = len(freq_22)
        self.modes = modes
        self.scale_freq_hm = scale_freq_hm

        self.listhlm = NULL

        # Build a real_vector representation of the input numpy array data
        # TODO: very clunky, need to find how to pass list structures to cython
        self.Cfreq_22 = NULL
        cdef double* freq_22_data = NULL
        if freq_22 is not None:
            if not freq_22.flags['C_CONTIGUOUS']:
                raise ValueError('Input numpy array freq is not C_CONTIGUOUS')
            freq_22_data = <double *> &freq_22[0]
            self.Cfreq_22 = real_vector_view(freq_22_data, freq_22.shape[0])
        self.Cfreq_21 = NULL
        cdef double* freq_21_data = NULL
        if freq_21 is not None:
            if not freq_21.flags['C_CONTIGUOUS']:
                raise ValueError('Input numpy array freq is not C_CONTIGUOUS')
            freq_21_data = <double *> &freq_21[0]
            self.Cfreq_21 = real_vector_view(freq_21_data, freq_21.shape[0])
        self.Cfreq_33 = NULL
        cdef double* freq_33_data = NULL
        if freq_33 is not None:
            if not freq_33.flags['C_CONTIGUOUS']:
                raise ValueError('Input numpy array freq is not C_CONTIGUOUS')
            freq_33_data = <double *> &freq_33[0]
            self.Cfreq_33 = real_vector_view(freq_33_data, freq_33.shape[0])
        self.Cfreq_32 = NULL
        cdef double* freq_32_data = NULL
        if freq_32 is not None:
            if not freq_32.flags['C_CONTIGUOUS']:
                raise ValueError('Input numpy array freq is not C_CONTIGUOUS')
            freq_32_data = <double *> &freq_32[0]
            self.Cfreq_32 = real_vector_view(freq_32_data, freq_32.shape[0])
        self.Cfreq_44 = NULL
        cdef double* freq_44_data = NULL
        if freq_44 is not None:
            if not freq_44.flags['C_CONTIGUOUS']:
                raise ValueError('Input numpy array freq is not C_CONTIGUOUS')
            freq_44_data = <double *> &freq_44[0]
            self.Cfreq_44 = real_vector_view(freq_44_data, freq_44.shape[0])
        self.Cfreq_43 = NULL
        cdef double* freq_43_data = NULL
        if freq_43 is not None:
            if not freq_43.flags['C_CONTIGUOUS']:
                raise ValueError('Input numpy array freq is not C_CONTIGUOUS')
            freq_43_data = <double *> &freq_43[0]
            self.Cfreq_43 = real_vector_view(freq_43_data, freq_43.shape[0])

        # cdef int Cforce_phiref_fref = <int> self.force_phiref_fref
        cdef int Cscale_freq_hm = <int> self.scale_freq_hm

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

        # ModGRParams not fully implemented yet, notably for setup stage
        if not (mod_gr_params is None):
            raise ValueError('ModGRParams for PhenomHM not fully implemented yet.')

        ret = IMRPhenomHMGethlmModes(
            &self.listhlm,
            &self.fpeak,
            &self.tpeak,
            &self.phipeak,
            &self.fstart,
            &self.tstart,
            &self.phistart,
            self.Cfreq_22,
            self.Cfreq_21,
            self.Cfreq_33,
            self.Cfreq_32,
            self.Cfreq_44,
            self.Cfreq_43,
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            self.dist,
            self.phiref,
            self.fref,
            self.Deltat,
            Cscale_freq_hm,
            self.Cextraparams,
            self.Cmodgrparams
        );

        if ret != 0:
            raise ValueError("Call to IMRPhenomHMGethlmModes() failed.")

        # Read the modes in a dictionary
        # Direct copy of C double array to numpy via a MemoryView
        # TODO: the AmpPhaseFDMode structures allow for a complex amplitude
        # and for different frequency vectors for the amplitude and phase
        # for now, we simply keep a simple freq, amp, phase structure
        cdef AmpPhaseFDMode* Chlm = NULL
        self.hlm = {}
        cdef double[::1] view_freq_amp
        cdef double[::1] view_amp_real
        # cdef double[::1] view_amp_imag
        # cdef double[::1] view_freq_phase
        cdef double[::1] view_phase
        cdef double[::1] view_tf
        for lm in self.modes:
            (l,m) = (lm[0], lm[1])
            self.hlm[lm] = {}
            if not lm in self.modes:
                raise ValueError('Mode not allowed: (%d, %d)' % (l, m))
            Chlm = ListAmpPhaseFDMode_GetMode(self.listhlm, l, m).hlm
            view_freq_amp = \
                        <(double)[:Chlm.freq_amp.size]> Chlm.freq_amp.data
            view_amp_real = \
                        <(double)[:Chlm.amp_real.size]> Chlm.amp_real.data
            # view_amp_imag = \
            #             <(double)[:Chlm.amp_imag.size]> Chlm.amp_imag.data
            # view_freq_phase = \
            #             <(double)[:Chlm.freq_phase.size]> Chlm.freq_phase.data
            view_phase = \
                        <(double)[:Chlm.phase.size]> Chlm.phase.data
            view_tf = \
                        <(double)[:Chlm.tf.size]> Chlm.tf.data
            # self.hlm[lm]['freq_amp'] = np.asarray(view_freq_amp)
            # self.hlm[lm]['amp_real'] = np.asarray(view_amp_real)
            # self.hlm[lm]['amp_imag'] = np.asarray(view_amp_imag)
            # self.hlm[lm]['freq_phase'] = np.asarray(view_freq_phase)
            # self.hlm[lm]['phase'] = np.asarray(view_phase)
            self.hlm[lm]['freq'] = np.asarray(view_freq_amp)
            self.hlm[lm]['amp'] = np.asarray(view_amp_real)
            self.hlm[lm]['phase'] = np.asarray(view_phase)
            self.hlm[lm]['tf'] = np.asarray(view_tf)

    def __dealloc__(self):
        """Destructor
        """
        if self.listhlm != NULL:
            ListAmpPhaseFDMode_Destroy(self.listhlm)

    # NOTE: to return a copy of the modes, copy.deepcopy is noticeably slower
    # TODO: the AmpPhaseFDMode structures allow for a complex amplitude
    # and for different frequency vectors for the amplitude and phase
    # for now, we simply keep a simple freq, amp, phase structure
    def get_waveform(self):
        hlm = {}
        for lm in self.modes:
            (l,m) = (lm[0], lm[1])
            hlm[lm] = {}
            # hlm[lm]['freq_amp'] = np.copy(self.hlm[lm]['freq_amp'])
            # hlm[lm]['amp_real'] = np.copy(self.hlm[lm]['amp_real'])
            # hlm[lm]['amp_imag'] = np.copy(self.hlm[lm]['amp_imag'])
            # hlm[lm]['freq_phase'] = np.copy(self.hlm[lm]['freq_phase'])
            # hlm[lm]['phase'] = np.copy(self.hlm[lm]['phase'])
            hlm[lm]['freq'] = np.copy(self.hlm[lm]['freq'])
            hlm[lm]['amp'] = np.copy(self.hlm[lm]['amp'])
            hlm[lm]['phase'] = np.copy(self.hlm[lm]['phase'])
            hlm[lm]['tf'] = np.copy(self.hlm[lm]['tf'])
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

    def compute_toff_lm(self, f_lm):
        cdef double tf22 = 0.
        cdef double tf21 = 0.
        cdef double tf33 = 0.
        cdef double tf32 = 0.
        cdef double tf44 = 0.
        cdef double tf43 = 0.
        t = IMRPhenomHMComputeTimeOfFrequencyModeByMode(
            &tf22,
            &tf21,
            &tf33,
            &tf32,
            &tf44,
            &tf43,
            f_lm[(2,2)],
            f_lm[(2,1)],
            f_lm[(3,3)],
            f_lm[(3,2)],
            f_lm[(4,4)],
            f_lm[(4,3)],
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            self.dist,
            self.phiref,
            self.fref,
            self.Deltat,
            self.Cextraparams,
            self.Cmodgrparams
        );
        tf_lm = {}
        tf_lm[(2,2)] = tf22
        tf_lm[(2,1)] = tf21
        tf_lm[(3,3)] = tf33
        tf_lm[(3,2)] = tf32
        tf_lm[(4,4)] = tf44
        tf_lm[(4,3)] = tf43
        return tf_lm

    def compute_foft_lm(self, tf_lm, f22_estimate, t_acc, max_iter=100):
        cdef double f22 = 0.
        cdef double f21 = 0.
        cdef double f33 = 0.
        cdef double f32 = 0.
        cdef double f44 = 0.
        cdef double f43 = 0.
        cdef int Cmax_iter = <int> max_iter
        f = IMRPhenomHMComputeInverseFrequencyOfTimeModeByMode(
            &f22,
            &f21,
            &f33,
            &f32,
            &f44,
            &f43,
            tf_lm[(2,2)],
            tf_lm[(2,1)],
            tf_lm[(3,3)],
            tf_lm[(3,2)],
            tf_lm[(4,4)],
            tf_lm[(4,3)],
            f22_estimate,
            t_acc,
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            self.dist,
            self.phiref,
            self.fref,
            self.Deltat,
            Cmax_iter,
            self.Cextraparams,
            self.Cmodgrparams
        );
        f_lm = {}
        f_lm[(2,2)] = f22
        f_lm[(2,1)] = f21
        f_lm[(3,3)] = f33
        f_lm[(3,2)] = f32
        f_lm[(4,4)] = f44
        f_lm[(4,3)] = f43

        return f_lm
