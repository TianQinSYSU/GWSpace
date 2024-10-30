#!/opt/miniconda3/bin/python3.12
#-*- coding: utf-8 -*-  
#==================================
# File Name: runGWSpace.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2024-10-21 10:49:29
#==================================

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

import gwspace
from gwspace.constants import DAY, YRSID_SI, MONTH
from gwspace.Waveform import waveforms
from gwspace.response import get_XYZ_td, tdi_XYZ2AET

from gwspace.utils import frequency_noise_from_psd, CholeskyDecomp, get_correlated_noise, calculate_cov

# print path
print(f"The code is installed at {gwspace.__path__}\n")

# define argument parser
parser = argparse.ArgumentParser(description="Using GWSpace to calculate the GW data that from space based detectors")

# add args
parser.add_argument("-i", "--ini-file", type=str, help="Initial files with source parameters")#, required=True)
parser.add_argument("-s", "--source", type=str, help="Source type, such as gcb, bhb, emri, sgwb") #, required=True)
parser.add_argument("-d", "--detector", type=str, help="Detector's name, such as TianQin, LISA, Taiji", default="TianQin")
#parser.add_argument("-G", "--TDI-gen", type=int, help="Generation of TDI method", default=1)
parser.add_argument("-C", "--TDI-channel", type=str, help="TDI channels, such as XYZ, AET", default="AET")
parser.add_argument("-T", "--Tobs", type=str, help="Total observational times, for example: 1*DAY, 3*MONTH, 1*YEAR, etc", default="3*MONTH")
parser.add_argument("-R", "--sample-rate", type=float, help="The sample rate in [Hz]", default=1)
parser.add_argument("-o", "--output", type=str, help="Filename of the output file", default=None)

parser.add_argument("--show-pars", action="store_true", help="Whether show the parameters and the run processes")

parser.add_argument("--check-orbit", action="store_true", help="Check or calculate the orbit of different detectors")
parser.add_argument("--check-wave", action="store_true", help="Check or calculate the original source waveforms of different sources")
parser.add_argument("--check-response", action="store_true", help="Check or calculate the response functions of different detectors")
parser.add_argument("--check-noise", action="store_true", help="Check or calculate the noise of different detectors")
parser.add_argument("--check-gendata", action="store_true", help="Check or calculate the generate data pipeline, return the responsed signals and noise")

parser.add_argument("--show-figs", action="store_true", help="Whether show the figures with the calculated data")

parser.add_argument("--generate-catalog", action="store_true", help="Generate the catalogue")
# Analysis args
args = parser.parse_args()


##=========================================
def dict_to_json(pars_dict, filename):
    pars_js = json.dumps(pars_dict, indent=4)
    with open(filename, "w") as fp:
        fp.write(pars_js)
    return

def read_json_to_dict(filename):
    with open(filename, "r") as fp:
        pars = json.load(fp)
    return pars

def white(psd):
    return np.random.normal(0, np.ones_like(psd))

def white_noise_meta(psd):
    return white(psd) + 1j*white(psd)

def get_fd_noise_xy(NX, NXY):
    nx = np.sqrt(0.5 * NX) * white_noise_meta(NX)
    ny = 0.5*NXY/np.sqrt(0.5*NX) * nx + np.sqrt(0.5*NX - np.abs(0.5*NXY)**2/(0.5*NX)) * white_noise_meta(NX)
    return (nx, ny)

def get_fd_noise_aet(NA, NT):
    nae = np.sqrt(0.5*NA)*white_noise_meta(NA)
    nt = np.sqrt(0.5*NT)*white_noise_meta(NT)
    return (nae, nt)

def get_fd_noise_xyz(Sx, Sxy, df):
    COV_XYZ = 0.5/df*np.array([[Sx, Sxy, Sxy],[Sxy, Sx, Sxy], [Sxy, Sxy, Sx]])
    L_XYZ = CholeskyDecomp(COV_XYZ)
    nx,ny,nz = get_correlated_noise(L_XYZ)
    return nx,ny,nz

def get_fd_noise_AET(SAE,ST, df):
    na = frequency_noise_from_psd(SAE, df)
    ne = frequency_noise_from_psd(SAE, df)
    nt = frequency_noise_from_psd(ST, df)
    return na,ne,nt

def generate_catalogue_MBHB():
    MBHBpars = {}
    MBHBpars["mass1"] = 10**np.random.uniform(3,8)
    MBHBpars["mass2"] = MBHBpars["mass1"] * np.random.uniform(1/18.0, 1)
    MBHBpars.update({
            "chi1": np.random.uniform(-1,1),
            "chi2": np.random.uniform(-1,1),
            "phi_c": np.random.uniform(0,2*np.pi),
            "tc": np.random.uniform(0, 5*YRSID_SI),
            "DL": 10**np.random.uniform(2,5),
            "psi": np.random.uniform(0,2*np.pi),
            "iota": np.random.uniform(0,np.pi),
            "Lambda": np.random.uniform(0,2*np.pi),
            "Beta": np.random.uniform(-np.pi/2, np.pi/2)            
            })
    return MBHBpars

def generate_catalogue_SBHB():
    SBHBpars = {}
    SBHBpars["mass1"] = 10**np.random.uniform(1,3)
    SBHBpars["mass2"] = SBHBpars["mass1"] * np.random.uniform(1/18.0, 1)
    SBHBpars.update({
            "chi1": np.random.uniform(-1,1),
            "chi2": np.random.uniform(-1,1),
            "var_phi": np.random.uniform(0,2*np.pi),
            "tc": np.random.uniform(0, 10*YRSID_SI),
            "DL": 10**np.random.uniform(1,3),
            "psi": np.random.uniform(0,2*np.pi),
            "iota": np.random.uniform(0,np.pi),            
            "Lambda": np.random.uniform(0,2*np.pi),
            "Beta": np.random.uniform(-np.pi/2, np.pi/2),            
            "eccentricity": np.random.uniform(0,0.4)
            })
    return SBHBpars

def generate_catalogue_EMRI():
    EMRIpars = {'M': 10**np.random.uniform(6,9),  # Mass of larger black hole in solar masses
                'mu': 10**np.random.uniform(1,2),  # Mass of compact object in solar masses
            'a': np.random.uniform(-1,1),  # Dimensionless spin of massive black hole, will be ignored in Schwarzschild waveform
            'p0': 12.0,
            'e0': np.random.uniform(0,0.4),
            'x0': 1.0,  # will be ignored in Schwarzschild waveform
            'qS': np.random.uniform(0,np.pi),  # polar sky angle
            'phiS': np.random.uniform(0,2*np.pi),  # azimuthal viewing angle
            'qK': np.random.uniform(0,np.pi),  # polar spin angle
            'phiK': np.random.uniform(0,2*np.pi),  # azimuthal viewing angle
            'dist': 10**np.random.uniform(1,3),  # Luminosity distance in Gpc
            'Phi_phi0': np.random.uniform(0,2*np.pi),
            'Phi_theta0': np.random.uniform(0,np.pi),
            'Phi_r0': 3.0*np.random.uniform(1,2),
            'psi': np.random.uniform(0,2*np.pi),
            'iota': np.random.uniform(0,np.pi)
            }
    return EMRIpars

def generate_catalogue_GCB():
    GCBpars = {}
    GCBpars['mass1'] = np.random.uniform(0.2,1.4)
    GCBpars['mass2'] = np.random.uniform(0.2,GCBpars['mass1'])
    GCBpars.update({'DL': np.random.uniform(1e-3,0.1),
            'phi0': np.random.uniform(0, 2*np.pi),
            'f0': 10**np.random.uniform(-4,-2),
            'fdot': 1.0e-16*np.random.uniform(0.1,9.9),
            'psi': np.random.uniform(0, 2*np.pi),
            'iota': np.random.uniform(0, np.pi),
            'Lambda': np.random.uniform(0, 2*np.pi),
            'Beta': np.random.uniform(-np.pi/2, np.pi/2)
            })
    return GCBpars

def generate_catalogue(types=["gcb"], nstars=[1]):
    catalogs = {}
    
    for tp, ns in zip(types, nstars):
        if tp == "gcb": tp = "GCB"
        if tp == "mbhb": tp = "MBHB"
        if tp == "sbhb": tp = "SBHB"
        if tp == "emri": tp = "EMRI"
        print(f"Generate catalogue of {ns} {tp} sources")
        print(f"Using the code of generate_catalogue_%s"%tp)
        for i in range(ns):
            catalogs["%s%s"%(tp,i)] = eval("generate_catalogue_%s"%tp)()
    if len(nstars) == 1 and nstars[0] == 1:
        return catalogs["%s0"%tp]
    return catalogs

# Convert to parameters
class ConvertPars:

    def __init__(self, args):
        # read the chek mode
        self.check_wave = args.check_wave
        self.check_noise = args.check_noise
        self.check_orbit = args.check_orbit
        self.check_response = args.check_response
        self.check_gendata = args.check_gendata

        self.show_pars = args.show_pars
        self.show_figs = args.show_figs

        # read in the args parameters
        # read in the source pars
        if self.check_wave or self.check_response or self.check_gendata:
            self.pars = read_json_to_dict(args.ini_file)
            self.source = args.source
        
        # read in the detector type
        self.detector = args.detector

        # read in the observational time and convert to frequency
        self.Tobs = eval(args.Tobs)

        # trans Tobs to pars
        if self.check_wave or self.check_response:
            if self.source == 'sgwb' or self.source == 'SGWB':
                self.pars['def']['T_obs'] = self.Tobs
            else:
                self.pars["T_obs"] = self.Tobs

        self.df = 1./self.Tobs
        self.fs_sample = args.sample_rate
        self.dt = 1./self.fs_sample
        self.f_max = 0.5 *self.fs_sample #1./(2*self.dt)

        # TDI pars
        self.TDIchannel = args.TDI_channel
        self.output = args.output

        # show the parameters
        if self.show_pars:
            print("="*80)
            if self.check_wave or self.check_gendata:
                print(f"The source type is {args.source}")
                print(f"The parameters of the source are:\n {self.pars}")
            if self.check_gendata:
                print(f"The source cataloge wil be {self.pars.keys()}")
            if not self.check_wave:
                print(f"The detector is {self.detector}")
            print("-"*80)
            print(f"The total observational time is {args.Tobs} = {self.Tobs} s")
            print(f"The sampling rate is {self.fs_sample} Hz")
            print(f"The minimal and maxminal frequencies will be {self.df} Hz and {self.f_max} Hz")
            if not self.check_wave or not self.check_orbit:
                print(f"The calculation results will be based on the TDI channel: {self.TDIchannel}")
            print("="*80)

    def get_orbits(self):
        from gwspace.Orbit import detectors

        if self.show_pars:
            print("Support detector orbits:\n", detectors.keys())
        
        print("Now calculate the orbits of ", self.detector)

        tf = np.arange(0, self.Tobs, self.dt)

        st = time.time()
        det = detectors[self.detector](tf)
        et = time.time()

        if self.show_pars:
            print(f"Time cost for initial position: {et-st} s")

        outorbit = np.array([tf, 
            det.orbits[0][0], det.orbits[0][1], det.orbits[0][2],
            det.orbits[1][0], det.orbits[1][1], det.orbits[1][2],
            det.orbits[2][0], det.orbits[2][1], det.orbits[2][2]])
        if self.output is not None:
            outfile = self.output
        else:
            outfile = "output/orbits_%s"%(self.detector)
        np.save(outfile, outorbit)
        print(r"Save the orbits data at ", outfile)

        if self.show_figs:
            tf_d = tf/DAY
            
            # Now let's check the orbits
            plt.figure()
            plt.plot(tf_d, det.orbits[0][0], label='x1')
            plt.plot(tf_d, det.orbits[1][0], label='x2')
            plt.plot(tf_d, det.orbits[2][0], label='x3')

            plt.title("The x arxis of three satellites")
            plt.xlabel('Time [Day]')
            plt.ylabel('Position (x) [s]')
            plt.legend()
            plt.tight_layout()

            plt.savefig("%s_pos_x.jpg"%outfile)

            # Now let's check the orbits
            plt.figure()
            plt.plot(tf_d, det.orbits[0][0], label='x1')
            plt.plot(tf_d, det.orbits[1][0], label='x2')
            plt.plot(tf_d, det.orbits[2][0], label='x3')

            plt.title("The x arxis of three satellites (zoom in to see the details)")
            plt.xlabel('Time [Day]')
            plt.ylabel('Position (x) [s]')
            # Zoom in to see the details            
            if self.detector == "TianQin":
                plt.xlim(90, 115)
                plt.ylim(480, 495)
            else:
                plt.xlim(90, 160)
                plt.ylim(400, 500)
            plt.legend()
            plt.tight_layout()

            plt.savefig("%s_pos_x_zoomin.jpg"%outfile)
            
            # Check how constellation center moves (For TianQin, that is how Earth moves)
            plt.figure()
            for i in range(3):
                plt.plot(tf_d, det.p_0[i], label=['x','y','z'][i])
                
            plt.title("The constellation center orbits")
            plt.xlabel('Time [Day]')
            plt.ylabel('Position (x) [s]')
            plt.legend()
            plt.tight_layout()

            plt.savefig("%s_guid.jpg"%outfile)

            #plt.show()

        return
    
    def get_noise(self, return_noise=False):
        print("This is a test of noise")
        from gwspace.Noise import detector_noises
        if self.show_pars:
            print('Support detector noises:', detector_noises.keys())
        print("Now calculate the noise of ", self.detector)

        nn = detector_noises[self.detector]()
        freq = np.arange(0, self.f_max, self.df)
        
        if self.TDIchannel == "XYZ":
            N1, N2 = nn.noise_XYZ(freq)
            #n1, n2 = get_fd_noise_xy(N1, N2)
            n1,n2,n3 = get_fd_noise_xyz(N1, N2, df=self.df)
            #_, cov = calculate_cov(nn1,nn2)
            #n1 = cov
        elif self.TDIchannel == "AET":
            N1, N2 = nn.noise_AET(freq)
            #n1, n2 = get_fd_noise_aet(N1, N2)
            n1,n3,n2 = get_fd_noise_AET(N1,N2, df=self.df)
        #print(self.f_max)
        if return_noise:
            return n1, n2, n3

        print("Save the noise data")
        if self.output is not None:
            outfile = self.output
        else:
            outfile = "output/noise_%s_%s"%(self.detector, self.TDIchannel)
        np.save(outfile, np.array([freq, n1, n2]))
        
        if self.show_figs:
            plt.figure()
            plt.loglog(freq[1:], np.abs(n1[1:]), 'r-', label="A/E" if self.TDIchannel == "AET" else "X")
            #plt.loglog(freq, TQ_E, 'g--', label="E")
            plt.loglog(freq[1:], np.abs(n2[1:]), 'b--', label="T" if self.TDIchannel == "AET" else "Y")
            plt.xlim(1e-4,self.f_max)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel('$\\sqrt{S_n}$ [Hz$^{-1/2}$]')
            plt.legend()
            plt.tight_layout()
            plt.title(r"TianQin's noise in %s channel"%self.TDIchannel)

            plt.savefig("%s_noise.jpg"%outfile)

            plt.show()
        return

    def get_waveform(self):
        print(waveforms.keys())        
        print(f"Generating {self.source} waveforms")

        if self.output is not None:
            outfile = self.output
        else:
            outfile = "output/%s_hphc_wave"%self.source

        tf = np.arange(0, self.Tobs, self.dt)        
        freq = np.arange(0, self.f_max, self.df)
        f_min = self.df

        if self.source == "gcb" or self.source == "GCB":
            wf = waveforms["gcb"](**self.pars)
            hp, hc = wf.get_hphc(tf)
            freq = np.fft.rfftfreq(hp.shape[0], d=self.dt)
            hpf = np.fft.rfft(hp)
            hcf = np.fft.rfft(hc)        
        elif self.source == "mbhb" or self.source == "MBHB":
            wf = waveforms['bhb_PhenomD'](**self.pars)
            amp,phase,tf,ft = self.get_IMRPhenomD(freq, wf)
            hlm = np.sqrt(5/4/np.pi)*amp*np.exp(1j*(phase))
            hpf = 0.5*(1+np.cos(wf.iota)**2)*hlm
            hcf = np.cos(wf.iota)*hlm
            np.save(outfile+'_tf', np.array([tf, ft]))
        elif self.source == "sbhb" or self.source == "SBHB":
            wf = waveforms['bhb_EccFD'](**self.pars)
            hpf, hcf = self.get_EccentricFD(wf)
            f_min = wf.f_min
        elif self.source == "emri" or self.source == "EMRI":
            wf = waveforms['emri'](**self.pars)
            hp, hc = wf.get_hphc(tf)
            freq = np.fft.rfftfreq(hp.shape[0], d=self.dt)
            hpf = np.fft.rfft(hp)
            hcf = np.fft.rfft(hc)
        elif self.source == "SGWB" or self.source == "sgwb":
            import healpy as hp
            from gwspace.SGWB import SGWB
            wf = SGWB(**self.pars["def"])
            res_signal, frange = wf.get_response_signal(**self.pars["signal"])
            # We can first look at what the original signal looked like
            signal_in_gu = wf.get_ori_signal(frange)
        else:
            pass

        print(f"Save the {self.source} waveform")
        if self.source == "SGWB" or self.source == "sgwb":
            np.save(outfile, signal_in_gu)
        else:
            np.save(outfile, np.array([freq, hpf, hcf]))
        
        if self.show_figs:
            if self.source == "sgwb" or self.source == "SGWB":
                hp.mollview(signal_in_gu[:, 0])
                return
            
            plt.figure()
            plt.loglog(freq[1:], np.abs(hpf[1:]), '--r', label="h_+(f)")
            plt.loglog(freq[1:], np.abs(hcf[1:]), ':b', label="h_x(f)")

            plt.xlim(np.max([1e-4, f_min]), np.min([self.f_max, 1]))
            plt.title("Strain of %s"%self.source)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Stain h_+/x")
            plt.savefig("%s_wave.jpg"%outfile)
            plt.legend(loc="best")

            if self.source == "mbhb" or self.source == "MBHB":
                plt.figure()
                plt.semilogy(tf, ft)
                
                plt.xlabel("Time")
                plt.ylabel("Frequency")
                plt.savefig("%s_tf.jpg"%outfile)

            plt.show()
    
        return

    def get_IMRPhenomD(self, freq, wf=None):
        if wf == None:
            wf = waveforms['bhb_PhenomD'](**self.pars)
        
        fmin = np.max([1e-5, wf.f_min, self.df])
        fmin_index = int(np.ceil(fmin/self.df))
        
        #tf = np.zeros_like(freq)
        ft = freq[fmin_index:]
        if self.check_gendata:
            d1 = np.zeros_like(freq, dtype=np.complex128)
            d2 = np.zeros_like(freq, dtype=np.complex128)
            d3 = np.zeros_like(freq, dtype=np.complex128)
            d1[fmin_index:], d2[fmin_index:], d3[fmin_index:] = wf.get_tdi_response(f_series=ft, 
                                                                                    channel=self.TDIchannel, 
                                                                                    det=self.detector)
            return (freq, d1, d2, d3)
        
        amp = np.zeros_like(freq)
        phase = np.zeros_like(freq)
        amps, phases, tfs = wf.get_amp_phase(f_series=ft)
        amp[fmin_index:] = amps[(2, 2)]
        phase[fmin_index:] = phases[(2, 2)]
        #tf = tfs[(2, 2)]
        fmax_index = np.where(np.diff(tfs[(2, 2)]) <= 0)[0][0]
        tf = tfs[(2,2)][:fmax_index]
        ft = ft[:fmax_index]
        return (amp, phase, tf, ft)
    
    def get_EccentricFD(self, wf=None):
        if wf == None:
            wf = waveforms['bhb_EccFD'](**self.pars)
        #fmin_index = int(np.ceil(wf.f_min/self.df))
        if self.check_gendata:
            (d1, d2, d3), f = wf.get_tdi_response(delta_f=self.df,
                                                  channel=self.TDIchannel, 
                                                  det=self.detector)
            return (d1,d2,d3), f
        hp, hc = wf.get_hphc(delta_f=self.df)
        return (hp, hc)

    def get_fd_TDI_GCB(self, wf=None):
        if wf == None:
            wf = waveforms['gcb'](**self.pars)
        f, X, Y, Z = wf.get_fastgb_fd_single(self.dt, oversample=1, 
                                             detector=self.detector)
        if self.TDIchannel == "XYZ":
            return (f,X,Y,Z)#A,E,T = 
        A,E,T = tdi_XYZ2AET(X,Y,Z)
        return (f,A,E,T)

    def get_fd_TDI_EMRI(self, wf=None):
        if wf == None:
            wf = waveforms['emri'](**self.pars)
        tf = np.arange(0, self.Tobs, self.dt)
        X, Y, Z = get_XYZ_td(wf, tf, det=self.detector)
        Xf = np.fft.rfft(X)
        Yf = np.fft.rfft(Y)
        Zf = np.fft.rfft(Z)
        freq = np.fft.rfftfreq(tf.shape[0], d=self.dt)
        if self.TDIchannel == "XYZ":
            return (freq, Xf, Yf, Zf)
        Af, Ef, Tf = tdi_XYZ2AET(Xf,Yf,Zf)
        return (freq, Af, Ef, Tf)

    def get_TDI_wave(self):
        print(waveforms.keys())        
        print(f"Generating TDI responsed {self.source} waveforms")
        
        if self.output is not None:
            outfile = self.output
        else:
            outfile = "output/%s_TDI_%s_signal"%(self.source, self.TDIchannel)
        
        tf = np.arange(0, self.Tobs, self.dt)        
        freq = np.arange(0, self.f_max, self.df)
        f_min = self.df
        print("The output file main name is: ", outfile)

        if self.source == "gcb" or self.source == "GCB":
            wf = waveforms["gcb"](**self.pars)
            freq, X,Y,Z = self.get_fd_TDI_GCB(wf)
        elif self.source == "mbhb" or self.source == "MBHB":
            wf = waveforms['bhb_PhenomD'](**self.pars)
            freq, X,Y,Z = self.get_IMRPhenomD(freq, wf)            
        elif self.source == "sbhb" or self.source == "SBHB":
            wf = waveforms['bhb_EccFD'](**self.pars)
            (X,Y,Z), freq = self.get_EccentricFD(wf)
            f_min = wf.f_min
        elif self.source == "emri" or self.source == "EMRI":
            wf = waveforms['emri'](**self.pars)
            freq, X,Y,Z = self.get_fd_TDI_EMRI(wf)
        elif self.source == "SGWB" or self.source == "sgwb":
            import healpy as hp
            from gwspace.SGWB import SGWB
            wf = SGWB(**self.pars["def"])
            res_signal, freq = wf.get_response_signal(**self.pars["signal"])            
            X = res_signal[:,0,0,0]
            Y = res_signal[:,0,1,1]
            Z = res_signal[:,0,2,2]
            if self.TDIchannel == "AET":
                X,Y,Z = tdi_XYZ2AET(X,Y,Z)
        else:
            pass

        print(f"Save the {self.source} waveform")
        np.save(outfile, np.array([freq, X,Y,Z]))
        
        if self.show_figs:
            plt.figure()
            plt.loglog(freq[1:], np.abs(X[1:]), '--r', label=self.TDIchannel[0])
            plt.loglog(freq[1:], np.abs(Y[1:]), '-.g', label=self.TDIchannel[1])
            plt.loglog(freq[1:], np.abs(Z[1:]), '--b', label=self.TDIchannel[2])

            if self.source == "gcb" or self.source == "GCB":
                plt.xlim(wf.f0 - 1e-5, wf.f0 + 1e-5)
            else:
                plt.xlim(np.max([1e-4, f_min]), np.min([self.f_max, 1]))
            plt.title("Strain of %s"%self.source)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Responsed strain %s"%self.TDIchannel)
            plt.savefig("%s_wave.jpg"%outfile)
            plt.legend(loc="best")

            plt.show()
    
        return

    def get_TDI_data(self):      
        #print(f"Generating TDI responsed {self.source} waveforms")      
        freq = np.arange(0, self.f_max, self.df)        
        Xf = np.zeros_like(freq, dtype=np.complex128)
        Yf = np.zeros_like(freq, dtype=np.complex128)
        Zf = np.zeros_like(freq, dtype=np.complex128)

        for event, pars in self.pars.items():
            print(f"Generate waveform for {event}")
            pars["T_obs"] = self.Tobs
            if event[:3] == "GCB":
                wf = waveforms["gcb"](**pars)
                f, X,Y,Z = self.get_fd_TDI_GCB(wf)
            if event[:3] == "MBH":
                wf = waveforms['bhb_PhenomD'](**pars)
                f, X,Y,Z = self.get_IMRPhenomD(freq, wf)
            if event[:3] == "SBH":
                wf = waveforms['bhb_EccFD'](**pars)
                (X,Y,Z), f = self.get_EccentricFD(wf)
                #f_min = wf.f_min
            if event[:3] == "EMR":
                wf = waveforms['emri'](**pars)
                f, X,Y,Z = self.get_fd_TDI_EMRI(wf)

            #print(f.shape, freq.shape)
            if f.shape[0] >= freq.shape[0]:
                index_n = 0
                index_m = freq.shape[0]
                X = X[:index_m]
                Y = Y[:index_m]
                Z = Z[:index_m]
            else:
                index_n = int(f[0]/self.df)
                index_m = int(f[-1]/self.df)
            Xf[index_n:index_m+1] += X
            Yf[index_n:index_m+1] += Y
            Zf[index_n:index_m+1] += Z
        
        n1, n2, n3 = self.get_noise(return_noise=True)
        if self.TDIchannel == "AET":
            Xf,Yf,Zf = tdi_XYZ2AET(Xf,Yf,Zf)
            Xf += n1
            Yf += n3
            Zf += n2
        elif self.TDIchannel == "XYZ":
            Xf += n1
            Yf += n2
            Zf += n3
        
        if self.output is not None:
            outfile = self.output
        else:
            outfile = "output/%s_TDI_%s_signals_cat"%(self.source, self.TDIchannel)
        print(f"Save the {self.source} waveform")
        np.save(outfile, np.array([freq, Xf,Yf,Zf]))

        if self.show_figs:
            plt.figure()
            plt.loglog(freq[1:], np.abs(Xf[1:]), '--r', label=self.TDIchannel[0])
            plt.loglog(freq[1:], np.abs(Yf[1:]), '-.g', label=self.TDIchannel[1])
            plt.loglog(freq[1:], np.abs(Zf[1:]), '--b', label=self.TDIchannel[2])

            plt.xlim(np.max([1e-4, self.df]), np.min([self.f_max, 1]))
            plt.title("Strain of %s"%self.source)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Responsed strain %s"%self.TDIchannel)
            plt.savefig("%s_wave.jpg"%outfile)
            plt.legend(loc="best")

            plt.show()
    
        return

    def test(self):
        print(r"This is a test")


##=========================================
if __name__ == "__main__":
    if args.generate_catalog:
        input_tps = input("Input the type or types of sources: (using space for separation)")
        tps = input_tps.split()
        input_stars = input("Input the number of source or sources: (using space for separation)")
        nstars = [int(item.strip()) for item in input_stars.split()]
        print(tps)
        print(nstars)
        catalog = generate_catalogue(tps, nstars)
        if args.output is None:
            outfile = "ini_file/catalogue_all.json"
        else:
            outfile = args.output
        dict_to_json(catalog, outfile)
        
        
    gws = ConvertPars(args)
    if gws.check_orbit:
        gws.get_orbits()
    if gws.check_noise:
        gws.get_noise()
    if gws.check_wave:
        gws.get_waveform()
    if gws.check_response:
        gws.get_TDI_wave()
    if gws.check_gendata:
        gws.get_TDI_data()


