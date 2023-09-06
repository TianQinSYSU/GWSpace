#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_TDI.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 09:48:18
#==================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    from pyINIDetectors import INITianQin

    print("This is TDI TD response generation code")

    Tobs = 4*DAY  # YRSID_SI / 4
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

    # GCBwf = WaveForm(GCBpars)
    # hpssb, hcssb = GCBwf(tf)

    TQ = INITianQin()
    td = TDResponse(GCBpars, TQ)

    st = time.time()
    yslr_ = td.Evaluate_yslr(tf)
    ed = time.time()

    print("Time cost is %f s for %d points" % (ed-st, tf.shape[0]))

    x, y, z = XYZ_TD(yslr_)

    plt.figure()

    plt.plot(tf, x, '-r')

    plt.figure()
    plt.plot(tf, y, '--b')

    plt.figure()
    plt.plot(tf, z, ':g')

    a, e, t = TDI_XYZ2AET(x, y, z)

    plt.figure()
    plt.plot(tf, a, '-r')

    plt.figure()
    plt.plot(tf, e, '--b')

    plt.figure()
    plt.plot(tf, t, ':g')

    plt.show()
