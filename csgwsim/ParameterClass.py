# coding: utf-8
# 2021-2023 Han Wang
# wangh657@mail2.sysu.edu.cn

"""Gravitational Wave Parameter Class. Including convenient convention functions for various usages"""
import numpy as np
from numpy import pi, sin, cos, sqrt
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as Cosmo

import consts as cons

_default = {'DL': 49.102,  # Luminosity distance (Mpc)
            'mass1': 21.44,  # Primary mass (solar mass)
            'mass2': 20.09,  # Secondary mass(solar mass)
            'Lambda': 3.44,  # Longitude
            'Beta': -0.074,  # Latitude
            'phi_c': 0,  # Coalescence phase
            'T_obs': cons.sidereal_year*2,  # Observation time (s)
            'tc': cons.sidereal_year*2,  # Coalescence time (s)
            'iota': 0.6459,  # Inclination angle
            'var_phi': 0,  # Observer phase
            'psi': 1.744, }  # Polarization angle


# Note1: one can use __slots__=('mass1', 'mass2', 'etc') to fix the attributes
#        then the class will not have __dict__ anymore, and attributes in __slots__ are read-only.
# Note2: One can use @DynamicAttrs to avoid warnings of 'no attribute'.
class WaveParas:
    """Required parameters: mass1, mass2, T_obs, location[(Lambda, Beta) or (ra, dec)]"""
    __slots__ = ('DL', 'mass1', 'mass2', 'Lambda', 'Beta', 'ra', 'dec',
                 'phi_c', 'T_obs', 'tc', 'iota', 'var_phi', 'psi',
                 'add_para', 'raw_source_masses')
    _true_para_key = ('DL', 'Mchirp', 'eta', 'phi_c', 'iota', 'tc', 'var_phi', 'psi', 'Lambda', 'Beta')
    fisher_key = ('Mchirp', 'eta', 'DL', 'phi_c', 'iota', 'tc', 'Lambda', 'Beta', 'psi')

    def __init__(self, DL=1., mass1=None, mass2=None, Lambda=None, Beta=None, ra=None, dec=None,
                 phi_c=0., T_obs=None, tc=0., iota=0., var_phi=0., psi=0.,
                 default_test=False, det_frame_para=False, **kwargs):
        self.DL = DL
        self.mass1 = mass1
        self.mass2 = mass2
        self.Lambda = Lambda
        self.Beta = Beta
        self.ra = ra
        self.dec = dec
        self.phi_c = phi_c
        self.T_obs = T_obs
        self.tc = tc
        self.iota = iota
        self.var_phi = var_phi
        self.psi = psi
        self.add_para = kwargs
        if default_test:
            # self.__dict__.update(_default)
            [setattr(self, key, _default[key]) for key in _default.keys()]

        if (self.mass1 is None) or (self.mass2 is None) or (self.T_obs is None):
            raise ValueError('mass1, mass2 and T_obs should NOT be None')

        if (self.Lambda is not None) and (self.Beta is not None):
            # print(f'[WaveParas]<Sky location is given by Ecliptic frame:'
            #       f'\n(lon, lat) in rad:({self.Lambda:.3f}, {self.Beta:.3f})>')
            pass
        else:
            try:
                from basic_funcs import icrs_to_ecliptic
                self.Lambda, self.Beta = icrs_to_ecliptic(ra, dec)
                print(f'[WaveParas]<Sky location is given by ICRS frame:'
                      f'\n(ra, dec) in rad:({ra:.3f}, {dec:.3f})>')
            except AttributeError:
                raise ValueError('ParameterClass without *valid* sky location parameters!') from None

        if not det_frame_para:
            self.raw_source_masses = {'mass1': self.mass1, 'mass2': self.mass2,
                                      'M': self.M, 'Mchirp': self.Mchirp}
            self.mass1 *= 1+self.z
            self.mass2 *= 1+self.z

    @property
    def redshift(self):
        return float(dl_to_z(self.DL))

    @property
    def z(self):
        return self.redshift

    @property
    def M(self):
        return self.mass1 + self.mass2  # Total mass (solar mass)

    @property
    def eta(self):
        return self.mass1 * self.mass2 / self.M**2  # Symmetric mass ratio

    @property
    def Mchirp(self):
        return self.eta**(3/5) * self.M  # Chirp mass (solar mass)

    def __eq__(self, other):
        return all([getattr(self, key) == getattr(other, key) for key in self._true_para_key])

    def wave_para_pycbc(self):
        """Convert parameters to a dict, specially for getting waveform from pycbc."""
        args = {'mass1': self.mass1,
                'mass2': self.mass2,
                'distance': self.DL,
                'coa_phase': self.phi_c,
                'inclination': self.iota,
                'long_asc_nodes': self.var_phi}
        return args

    @property
    def f_min(self):
        return 5**(3/8)/(8*np.pi) * (cons.M_unit*self.Mchirp)**(-5/8) * self.T_obs**(-3/8)

    @property
    def vec_k(self):
        return np.array([-cos(self.Beta)*cos(self.Lambda),
                         -cos(self.Beta)*sin(self.Lambda),
                         -sin(self.Beta)])  # Vector of sources

    def _p0(self):
        """See "LDC-manual-002.pdf" (Eq. 12, 13) & Marsat et al. (Eq. 14)"""
        sib, csb = sin(self.Beta), cos(self.Beta)
        sil, csl = sin(self.Lambda), cos(self.Lambda)
        sil2, csl2 = sin(2*self.Lambda), cos(2*self.Lambda)

        p0_plus = np.array([-sib**2 * csl**2 + sil**2, (sib**2+1)*(-sil*csl),  sib*csb*csl,
                            (sib**2+1)*(-sil*csl),     -sib**2*sil**2+csl**2,  sib*csb*sil,
                            sib*csb*csl,                sib*csb*sil,          -csb**2]).reshape(3, 3)
        p0_cross = np.array([-sib*sil2, sib*csl2,  csb*sil,
                             sib*csl2,  sib*sil2, -csb*csl,
                             csb*sil,  -csb*csl,   0]).reshape(3, 3)
        return p0_plus, p0_cross

    def polarization(self):
        """See "LDC-manual-002.pdf" (Eq. 19)"""
        p0_plus, p0_cross = self._p0()
        p_plus = p0_plus*cos(2*self.psi) + p0_cross*sin(2*self.psi)
        p_cross = - p0_plus*sin(2*self.psi) + p0_cross*cos(2*self.psi)
        return p_plus, p_cross

    def gen_tdi_waveform(self, det, channel, resample, delta_f, f_min, f_max):
        raise NotImplementedError

    def gen_tdi_pycbc_series(self, delta_f, det='TQ', channel='A', f_final=1., f_min=None):
        """For `gen_tdi_waveform`, generate `pycbc.types.FrequencySeries` for further calculation."""
        from pycbc.types import FrequencySeries

        wf_dict = self.gen_tdi_waveform(det, channel, False, delta_f, f_min, f_final)
        if wf_dict['freq'][0] == 0.:
            wf = wf_dict['TDI_'+channel]
        else:
            wf = np.zeros(int(round(f_final/delta_f))+1, dtype=np.complex128)  # 1/1e-5=99999.99999999999
            wf[-1-len(wf_dict['freq']):-1] = wf_dict['TDI_'+channel][:]
        return FrequencySeries(wf, delta_f=delta_f, epoch=-int(1./delta_f))


class WaveParasPYCBC(WaveParas):
    """Waveform Parameters, generally for aligned-spin approximants in PyCBC."""
    __slots__ = ('chi1', 'chi2', 'approximant')
    _true_para_key = WaveParas._true_para_key + ('chi1', 'chi2')
    fisher_key = ('Mchirp', 'eta', 'chi1', 'chi2', 'DL', 'phi_c', 'iota', 'tc', 'ra', 'dec', 'psi')

    def __init__(self, DL=1., mass1=None, mass2=None, Lambda=None, Beta=None, ra=None, dec=None,
                 phi_c=0., T_obs=None, tc=0., iota=0., var_phi=0., psi=0., chi1=0., chi2=0.,
                 approximant="IMRPhenomHM", default_test=False, det_frame_para=False, **kwargs):
        WaveParas.__init__(self, DL, mass1, mass2, Lambda, Beta, ra, dec,
                           phi_c, T_obs, tc, iota, var_phi, psi,
                           default_test, det_frame_para, **kwargs)
        self.chi1 = chi1
        self.chi2 = chi2
        self.approximant = approximant

    def wave_para_pycbc(self):
        args = {'mass1': self.mass1,
                'mass2': self.mass2,
                'spin1z': self.chi1,
                'spin2z': self.chi2,
                'distance': self.DL,
                'coa_phase': self.phi_c,
                'inclination': self.iota,
                'long_asc_nodes': self.var_phi}
        return args

    def gen_ori_waveform(self, delta_f=None, f_min=None, f_max=None):
        from pycbc.waveform import get_fd_waveform

        if not f_min:
            f_min = self.f_min
        return get_fd_waveform(**self.wave_para_pycbc(), approximant=self.approximant,
                               f_lower=f_min, delta_f=delta_f, f_final=f_max)

    def gen_ant_from_wf(self, hp, hc, det):
        """Apply antenna pattern of ground-based detector and time shift in f-domain.

        :param hp: FrequencySeries, h_plus
        :param hc: FrequencySeries, h_cross
        :param det: str, alias of the detector
        :return: ndarray
        """
        from pycbc.detector import Detector
        from pycbc.waveform import utils
        det = Detector(det)
        ra, dec, psi, tc = self.ra, self.dec, self.psi, self.tc

        fp, fc = det.antenna_pattern(ra, dec, psi, t_gps=tc)
        d_t = det.time_delay_from_earth_center(ra, dec, t_gps=tc)
        hap = fp*hp+fc*hc
        hap = utils.apply_fseries_time_shift(hap, dt=d_t+tc)
        return hap.data

    def gen_ant_waveform(self, det, delta_f=None, f_min=None, f_max=None):
        hp, hc = self.gen_ori_waveform(delta_f, f_min, f_max)
        return self.gen_ant_from_wf(hp, hc, det)


class WaveParasPhenomD(WaveParas):
    """Waveform Parameters, specially for pyIMRPhenomD."""
    __slots__ = ('chi1', 'chi2')
    _true_para_key = WaveParas._true_para_key + ('chi1', 'chi2')
    fisher_key = ('Mchirp', 'eta', 'chi1', 'chi2', 'DL', 'phi_c', 'iota', 'tc', 'Lambda', 'Beta', 'psi')

    def __init__(self, DL=1., mass1=None, mass2=None, Lambda=None, Beta=None, ra=None, dec=None,
                 phi_c=0., T_obs=None, tc=0., iota=0., var_phi=0., psi=0., chi1=0., chi2=0.,
                 default_test=False, det_frame_para=False, **kwargs):
        WaveParas.__init__(self, DL, mass1, mass2, Lambda, Beta, ra, dec,
                           phi_c, T_obs, tc, iota, var_phi, psi,
                           default_test, det_frame_para, **kwargs)
        self.chi1 = chi1
        self.chi2 = chi2

    def wave_para_pycbc(self):
        args = {'mass1': self.mass1,
                'mass2': self.mass2,
                'spin1z': self.chi1,
                'spin2z': self.chi2,
                'distance': self.DL,
                'coa_phase': self.phi_c}
        return args

    def wave_para_phenomd(self, f_ref=0.):
        """Convert parameters to a list, specially for getting waveform from `pyIMRPhenomD`."""
        phi_ref = self.phi_c
        m1_si = self.mass1 * cons.M_sun
        m2_si = self.mass2 * cons.M_sun
        chi1, chi2 = self.chi1, self.chi2
        dl_si = self.DL * cons.Mpc
        return phi_ref, f_ref, m1_si, m2_si, chi1, chi2, dl_si

    def _y22(self):
        """See "LDC-manual-002.pdf" (Eq. 31)"""
        y22_o = sqrt(5/4/pi) * cos(self.iota/2)**4 * np.exp(2j*self.var_phi)
        y2_2_conj = sqrt(5/4/pi) * sin(self.iota/2)**4 * np.exp(2j*self.var_phi)
        return y22_o, y2_2_conj

    # p_lm(self, l=2, m=2):
    #     y_lm_o = spin_weighted_spherical_harmonic(-2, l, m, self.iota, self.var_phi)
    #     y_l_m_conj = spin_weighted_spherical_harmonic(-2, l, -m, self.iota, self.var_phi).conjugate()
    @property
    def p22(self):
        """See Marsat et al. (Eq. 16) https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.083011"""
        y22_o, y2_2_conj = self._y22()
        p0_plus, p0_cross = self._p0()
        return (1/2 * y22_o * np.exp(-2j*self.psi) * (p0_plus + 1j*p0_cross) +
                1/2 * y2_2_conj * np.exp(2j*self.psi) * (p0_plus - 1j*p0_cross))

    def gen_ori_waveform(self, delta_f=None, f_min=None, f_max=1., resample=False):
        """Generate f-domain TDI waveform(IMRPhenomD, h22 mode), with or without resampling"""
        if not f_min:
            f_min = self.f_min  # The initial freq _sec

        if resample:
            from TDI_fd_resample import gen_resampled_phenomd
            wf = gen_resampled_phenomd(self, f_min=f_min, f_max=f_max)
        else:
            from pyIMRPhenomD import IMRPhenomDh22AmpPhase
            if delta_f is None:
                delta_f = 1/self.T_obs
            freq_phd = np.arange(np.ceil(f_min/delta_f)*delta_f, f_max, delta_f)
            # freq_phd = np.arange(f_min, f_max, delta_f)  # TODO
            wf_phd_class = IMRPhenomDh22AmpPhase(freq_phd, *self.wave_para_phenomd())
            freq, amp, phase = wf_phd_class.GetWaveform()  # freq, amp, phase
            # WTF, these are actually cython pointers, we should also use .copy() to acquire ownership
            wf = (freq.copy(), amp.copy(), phase.copy())

        return wf, None

    def gen_tdi_from_wf(self, wf, freq=None, det='TQ', channel='AET'):
        # TODO: To be honest this is not elegant to split gen_tdi_waveform into 2 parts,
        #  but it is useful for FIM calculation. Any way to rewrite this?
        from SpaceDetectorResponse import fd_tdi_response_single
        return fd_tdi_response_single(wf, self, channel=channel, det=det, freq=freq)

    def gen_tdi_waveform(self, det='TQ', channel='AET', resample=True, delta_f=None, f_min=None, f_max=1.):
        """Generate f-domain TDI waveform(IMRPhenomD, h22 mode), with or without resampling"""
        wf, freq = self.gen_ori_waveform(delta_f, f_min, f_max, resample)
        return self.gen_tdi_from_wf(wf, freq, det, channel)


class WaveParasEcc(WaveParas):
    """Waveform Parameters including eccentricity, using EccentricFD Waveform."""
    __slots__ = 'eccentricity'
    _true_para_key = WaveParas._true_para_key + ('eccentricity', )
    fisher_key = ('Mchirp', 'eta', 'DL', 'phi_c', 'iota', 'tc', 'Lambda', 'Beta', 'psi', 'eccentricity')

    def __init__(self, DL=1., mass1=None, mass2=None, Lambda=None, Beta=None, ra=None, dec=None,
                 phi_c=0., T_obs=None, tc=0., iota=0., var_phi=0., psi=0., eccentricity=0.,
                 default_test=False, det_frame_para=False, **kwargs):
        WaveParas.__init__(self, DL, mass1, mass2, Lambda, Beta, ra, dec,
                           phi_c, T_obs, tc, iota, var_phi, psi,
                           default_test, det_frame_para, **kwargs)
        self.eccentricity = eccentricity

    def wave_para_pycbc(self):
        args = {'mass1': self.mass1,
                'mass2': self.mass2,
                'distance': self.DL,
                'coa_phase': self.phi_c,
                'inclination': self.iota,
                'long_asc_nodes': self.var_phi,
                'eccentricity': self.eccentricity}
        return args

    def gen_ori_waveform(self, delta_f=None, f_min=None, f_max=1.):
        """Generate f-domain TDI waveform(EccentricFD)"""
        from ecc_waveform_lib import gen_ecc_fd_and_phase

        if not f_min:
            f_min = self.f_min
        if delta_f is None:
            delta_f = 1/self.T_obs

        wf = gen_ecc_fd_and_phase(**self.wave_para_pycbc(), delta_f=delta_f,
                                  f_lower=f_min, f_final=f_max, obs_time=self.T_obs)
        freq = delta_f * np.array(range(len(wf[0][0])))
        return wf, freq

    def gen_tdi_from_wf(self, wf, freq, det='TQ', channel='AET'):
        from SpaceDetectorResponse import fd_tdi_response_ecc
        return fd_tdi_response_ecc(wf, freq, self, channel=channel, det=det)

    def gen_tdi_waveform(self, det='TQ', channel='AET', resample=False, delta_f=None, f_min=None, f_max=1.):
        """Generate f-domain TDI waveform(EccentricFD)"""
        if resample:
            raise NotImplementedError

        wf, freq = self.gen_ori_waveform(delta_f, f_min, f_max)
        return self.gen_tdi_from_wf(wf, freq, det, channel)


waveform_approx = {'PhenomD': WaveParasPhenomD,
                   'EccFD': WaveParasEcc}

_z = np.linspace(0, 10, 1000)
_rc = Cosmo.luminosity_distance(_z)
dl_to_z = interp1d(_rc, _z, kind='cubic')



