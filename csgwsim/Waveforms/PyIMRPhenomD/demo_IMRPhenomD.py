"""demo of the IMRPhenomD Python module C 2021 Matthew Digman"""
from time import perf_counter

import numpy as np

from Waveforms.PyIMRPhenomD.IMRPhenomD import AmpPhaseFDWaveform,IMRPhenomDGenerateh22FDAmpPhase
import Waveforms.PyIMRPhenomD.IMRPhenomD_const as imrc

if __name__=='__main__':

    t_start = perf_counter()
    #set the number of frequency pixels
    NF = 10024

    #set some parameters for a test run of a LISA supermassive black hole source
    distance = 1.0e9*imrc.PC_SI/imrc.CLIGHT
    chi1 = 0.1
    chi2 = 0.2

    DF = 6.71057655855477e-2


    m1_SI =  35.0*imrc.MSUN_SI
    m2_SI =  30.0*imrc.MSUN_SI

    freq = np.arange(1,NF+1)*DF

    phic = 0.

    MfRef_in = 0.

    amp_imr = np.zeros(NF)
    phase_imr = np.zeros(NF)
    if imrc.findT:
        time_imr = np.zeros(NF)
        timep_imr = np.zeros(NF)
    else:
        time_imr = np.zeros(0)
        timep_imr = np.zeros(0)


    #the first evaluation of the amplitudes and phase will always be much slower, because it must compile everything
    t0 = perf_counter()
    h22 = AmpPhaseFDWaveform(NF,freq,amp_imr,phase_imr,time_imr,timep_imr,0.,0.)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22,freq,phic,MfRef_in,m1_SI,m2_SI,chi1,chi2,distance*imrc.CLIGHT)
    tf = perf_counter()
    print("compiled in %10.7f seconds"%(tf-t0))


    #run 10000 times with the compiled version to test speed
    t0 = perf_counter()
    n_run = 10000
    for itrm in range(0,n_run):
        IMRPhenomDGenerateh22FDAmpPhase(h22,freq,phic,MfRef_in,m1_SI,m2_SI,chi1,chi2,distance*imrc.CLIGHT)
    tf = perf_counter()
    print("run      in %10.7f seconds"%((tf-t0)/n_run))


    import matplotlib.pyplot as plt
    #plot the quantities of interest for our test case
    plt.loglog(freq,h22.amp)
    plt.ylabel('amplitude')
    plt.xlabel('f')
    plt.show()

    plt.loglog(freq,np.abs(h22.phase))
    plt.ylabel('Phase')
    plt.xlabel('f')
    plt.show()

    #The c version does not get these two quantities directly
    #but it is faster and more accurate to get them directly than to use a numerical derivative
    plt.loglog(freq,np.abs(h22.time))
    plt.ylabel('t(f)')
    plt.xlabel('f')
    plt.show()

    plt.loglog(freq,np.abs(h22.timep))
    plt.ylabel('t\'(f)')
    plt.xlabel('f')
    plt.show()
