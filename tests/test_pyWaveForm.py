#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_pyWaveForm.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:52:12
#==================================

def test_GCB():
    Tobs = 10000  # const.YRSID_SI / 4
    delta_f = 1/Tobs
    delta_T = 1
    f_max = 1/(2*delta_T)

    tf = np.arange(0, Tobs, delta_T)

    print("Testing of GCB waveform")
    GCBpars = {"type": "GCB",
               "Mc": 0.5,
               "DL": 0.3,
               "phi0": 0.0,
               "f0": 0.001,
               "psi": 0.2,
               "iota": 0.3,
               "lambda": 0.4,
               "beta": 1.2,
               }

    print("Mc" in GCBpars.keys())

    GCBwf = WaveForm(GCBpars)
    hpssb, hcssb = GCBwf(tf)

    import matplotlib.pyplot as plt

    plt.plot(tf, hpssb, 'r-', label=r'$h_+^{\rm SSB}$')
    plt.plot(tf, hcssb, 'b--', label=r'$h_\times^{\rm SSB}$')

    plt.xlabel("Time")
    plt.ylabel(r"$h_+, h_\times$")

    plt.legend(loc="best")
    plt.show()

    return


def test_BHB():
    import matplotlib.pyplot as plt

    print("Testing of BHB waveform")
    BHBpars = {"type": "BHB",
               "m1": 3.5e6,
               "m2": 2.1e5,
               "chi1": 0.2,
               "chi2": 0.1,
               "DL": 1e3,
               "phic": 0.0,
               "MfRef_in": 0,
               "psi": 0.2,
               "iota": 0.3,
               "lambda": 0.4,
               "beta": 1.2,
               "tc": 0,
               }

    BHBwf = WaveForm(BHBpars)

    NF = 1024
    freq = 10**np.linspace(-4, 0, NF)

    amp, phase, time, timep = BHBwf.amp_phase(freq)

    plt.figure()
    plt.loglog(freq, amp[(2, 2)])
    plt.ylabel('amplitude')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(phase[(2, 2)]))
    plt.ylabel('Phase')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(time[(2, 2)]))
    plt.ylabel('time')
    plt.xlabel('freq')

    plt.figure()
    plt.loglog(freq, np.abs(timep[(2, 2)]))
    plt.ylabel('dt/df')
    plt.xlabel('freq')

    plt.show()

    return


def test_EMRI():
    print("This is a test of loading EMRI waveform")
    # parameters
    Tobs = 0.3*YRSID_SI  # years
    dt = 15.0  # seconds

    pars = {"type": "EMRI",
            'M': 1e6,
            'a': 0.1,
            'mu': 1e1,
            'p0': 12.0,
            'e0': 0.2,
            'x0': 1.0,
            'qK': 0.2,
            'phiK': 0.2,
            'qS': 0.3,
            'phiS': 0.3,
            'dist': 1.0,
            'Phi_phi0': 1.0,
            'Phi_theta0': 2.0,
            'Phi_r0': 3.0,
            'psi': 0.4,
            'iota': 0.2,
            }

    wf = WaveForm(pars)

    tf = np.arange(0, Tobs, dt)

    hp, hc = wf(tf)

    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(tf[:2000], hp[:2000])
    plt.plot(tf[:2000], hc[:2000])

    plt.show()

    return


if __name__ == '__main__':
    print("This is waveform generation code")
    # test_GCB()
    # test_BHB()
    test_EMRI()
