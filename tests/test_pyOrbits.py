#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_pyOrbits.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:55:52
#==================================

from active_python_path import csgwsim

import numpy as np
GCBWaveform = csgwsim.GCBWaveform
INITianQin = csgwsim.INITianQin
YRSID_SI = csgwsim.Constants.YRSID_SI
DAY = csgwsim.Constants.DAY


if __name__ == '__main__':
    print("Here is the analytical orbits")
    import time

    st = time.time()

    TQ = INITianQin()
    TQOrbit = csgwsim.Orbit(TQ)

    dt = 3600

    tt = np.arange(-5*dt, YRSID_SI+5*dt, dt)
    ret = TQOrbit.get_position(tt)

    ed = time.time()
    print("Time cost for initial position: %f s"%(ed-st))

    # Then one can get the position at any time in one year
    Tobs = YRSID_SI/12
    delta_f = 1/Tobs
    delta_T = 1
    f_max = 1/(2*delta_T)

    tf = np.arange(0, Tobs, delta_T)

    p1L_x = TQOrbit._get_pos['p1L']['x'](tf)
    p1L_y = TQOrbit._get_pos['p1L']['y'](tf)
    p1L_z = TQOrbit._get_pos['p1L']['z'](tf)

    p2L_x = TQOrbit._get_pos['p2L']['x'](tf)

    p3L_x = TQOrbit._get_pos['p3L']['x'](tf)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    plt.plot(tf/DAY, p1L_x, 'r-', label='x')
    plt.plot(tf/DAY, p1L_y, 'g--', label='y')
    plt.plot(tf/DAY, p1L_z, 'b:', label='z')

    plt.plot(tf/DAY, p2L_x, 'b-', label='x2')

    plt.plot(tf/DAY, p3L_x, 'g-', label='x3')

    plt.xlabel('Time [Day]')
    plt.ylabel('pos (x,y,z) [m]')

    plt.legend(loc='best')

    ##==================================
    fig1 = plt.figure()

    st = time.time()

    p0 = TQOrbit.get_pos(tf, pp="n1")

    ed = time.time()
    print("Time cost for generate orbit pos is %f s for %d point"%(ed-st, tf.shape[0]))

    for i in range(3):
        plt.plot(tf/DAY, p0[i])

    ##==================================
    fig2 = plt.figure()

    st = time.time()
    v2 = TQOrbit.get_pos(tf, pp="v2")
    ed = time.time()

    print("Time cost for generat vel if %f s for %d point"%(ed-st, v2.shape[1]))

    for i in range(3):
        plt.plot(tf/DAY, v2[i])

    ##================================
    st = time.time()
    p0 = TQOrbit.get_position_px(tf, pp="p1", get_vel=False)[0]
    ed = time.time()

    print("Time cost for generat vel if %f s for %d point"%(ed-st, p0.shape[1]))

    for i in range(3):
        plt.plot(tf/DAY, p0[i])

    plt.show()
