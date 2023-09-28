import numpy as np
import healpy as hp
from healpy import Alm
from scipy.special import lpmn, sph_harm
import matplotlib.pyplot as plt
from csgwsim import Noise
from sympy.physics.quantum.cg import CG
from csgwsim import Orbit
from csgwsim import response
import time
from csgwsim.Constants import YRSID_SI, DAY
from csgwsim.Orbit import TianQinOrbit
from SGWB import SGWB


SGWBpars = {"type": "GCB",
            "fmax": 0.2,
            "fmin": 0.001,
            "fn": 200,
            "tsegmid": 5000,
            "Ttot": 63*5000,
            "omega0": 5e-11,
            "alpha": 0.667,
            "nside_in":12,
            "blm_vals":(1.0, 0.75, 0.5, 0.7j, 0.7-0.3j, 1.1j),
            "blmax":2
            }
SGWB_signal = SGWB(fmax=0.2,fmin=0.001,fn=200,tsegmid=5000,Ttot=63*5000,omega0=5e-11,alpha=0.667, nside_in=12,blm_vals=(1.0, 0.75, 0.5, 0.7j, 0.7-0.3j, 1.1j),blmax=2)

hp.mollview(SGWB_signal.signal_in_Gu[0,:],title="The response signal")

tq = Noise.TianQinNoise()
TX, TXY = tq.noise_XYZ(SGWB_signal.frange,unit = "displacement")/(2e8*np.sqrt(3))**2

plt.figure()
plt.loglog(SGWB_signal.frange,TX,label = "XYZ channels noise PSD")
plt.loglog(SGWB_signal.frange,SGWB_signal.signal_SGWB[0,0,:,0],label = "inject signal sample")
plt.xlabel('frequency[Hz]')
plt.ylabel('PSD [Hz]')
plt.legend(loc='best')
plt.show()
# one year SNR for SGWB
SNR = np.sqrt(24*3600*3.64*np.sum(SGWB_signal.signal_SGWB[0,0,:,0]**2/TX**2)/SGWB_signal.frange.size)
print(SNR)

