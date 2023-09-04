#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: pyOrbits.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-01 10:23:11
# ==================================

# import numpy as np
# from scipy.interpolate import InterpolatedUnivariateSpline as Spline
# from Constants import *


class Orbit(object):
    """
    Analytical orbits of spacecrafts
    Note that all the length are in the unit of time [s]
    ----------------------------------------
    Parameters:
    - INI: initial parameters of detectors
    ----------------------------------------
    How to using it:
    ```python
    TQ = INITianQin()
    TQOrbit = Orbit(TQ)

    tt = np.arange(0, YRSID_SI, 3600)
    ret = TQOrbit.get_position(tt)

    # Then one can get the position at any time in one year
    t_new = 1.0
    p0_x = TQOrbit.get_pos['p0']['x'](t_new)
    ```
    """

    def __init__(self, INI):
        self.kappa_0 = INI.kappa_e
        self.armLT = INI.armLength/C_SI  # in time unit [s]
        self.RT = self.armLT/np.sqrt(3)  # in time unit [s]
        self.ecc = INI.ecc
        self.kappa = INI.kappa_0
        self.Omega = INI.Omega
        self.perigee = INI.perigee

        if INI.detector == "TianQin":
            self.phi_s = PI_2-INI.beta_s
            self.theta_s = INI.theta_s

    def get_position_px(self, time, pp="p0", get_vel=False):
        """
        Directly calculate the position or velocity of spacecrafts
        ----------------------------------------------------------
        Parameters:
        - time: single or array of time
        - pp: position ["p0", "p1"]
        """
        if pp == "p0":
            if get_vel:
                p, v = self.position_earth(time)
            else:
                p = self.position_earth(time, get_vel)
        elif pp != "p0":
            if get_vel:
                p, v = self.position_spa_Local(time, pp)
            else:
                p = self.position_spa_Local(time, pp, get_vel)
        if get_vel:
            return p, v
        return p

    def get_position(self, time):
        """
        Calculate the position and do the spline
        -----------------------------------------
        Initial the position and velocity of spacecrafts,
        to get the spline class for position and velocity.
        -----------------------------------------
        Parameters:
        - time: array
        """
        self.time = time
        p0, v0 = self.position_earth(time)
        p1L, p2L, p3L, v1L, v2L, v3L = self.position_spa_Local(time)

        # n1 = (p3L - p2L)/self.armLT
        # n2 = (p1L - p3L)/self.armLT
        # n3 = (p2L - p1L)/self.armLT
        try:
            from scipy.interpolate import InterpolatedUnivariateSpline as Spline
        except:
            print("Spline already defined")

        self._get_pos = {}
        for pp in ["p0", "p1L", "p2L", "p3L", "v0", "v1L", "v2L", "v3L"]:
            self._get_pos[pp] = {}
            for i, x in enumerate(["x", "y", "z"]):
                self._get_pos[pp][x] = Spline(time, eval("%s" % pp)[i])

        return 0

    def get_pos(self, tf, pp='p0'):
        """
        Return the position of different point:
        -----------------------------------
        pp: chose one in 
            ["p0", "p1L", "p2L", "p3L",
            "p1", "p2", "p3", 
            "n1", "n2", "n3", 
            "v0", "v1L", "v2L", "v3L",
            "v1", "v2", "v3"]
        """
        if pp in ["p0", "p1L", "p2L", "p3L", "v0", "v1L", "v2L", "v3L"]:
            x = self._get_pos[pp]["x"](tf)
            y = self._get_pos[pp]["y"](tf)
            z = self._get_pos[pp]["z"](tf)
        elif pp in ["p1", "p2", "p3", "v1", "v2", "v3"]:
            if pp[0] == "p":
                pp0 = "p0"
            else:
                pp0 = "v0"

            pp1 = "%sL"%pp

            x = self._get_pos[pp0]["x"](tf)
            y = self._get_pos[pp0]["y"](tf)
            z = self._get_pos[pp0]["z"](tf)

            x += self._get_pos[pp1]["x"](tf)
            y += self._get_pos[pp1]["y"](tf)
            z += self._get_pos[pp1]["z"](tf)

        elif pp[0] == "n":
            if pp == "n1":
                pr = "p2L"
                ps = "p3L"
            elif pp == "n2":
                pr = "p3L"
                ps = "p1L"
            elif pp == "n3":
                pr = "p1L"
                ps = "p2L"
            x = self._get_pos[pr]["x"](tf)
            y = self._get_pos[pr]["y"](tf)
            z = self._get_pos[pr]["z"](tf)

            x -= self._get_pos[ps]["x"](tf)
            y -= self._get_pos[ps]["y"](tf)
            z -= self._get_pos[ps]["z"](tf)

            x /= self.armLT
            y /= self.armLT
            z /= self.armLT
        else:
            return 0

        return np.array([x, y, z])

    def alpha_earth(self, time):
        """
        the mean orbital ecliptic longitude of the geocenter in the heliocentric-elliptic coordinate system.
        """
        return EarthOrbitOmega_SI*time+self.kappa_0-Perihelion_Ang

    def position_earth(self, time, get_vel=True):
        ecc = EarthEccentricity
        ecc2 = ecc*ecc

        alpha = self.alpha_earth(time)
        csa, sna = np.cos(alpha), np.sin(alpha)

        x = AU_T*(csa-ecc*(1+sna*sna)-1.5*ecc2*csa*sna*sna)
        y = AU_T*(sna+ecc*sna*csa+0.5*ecc2*sna*(1-3*sna*sna))
        z = np.zeros(len(time))

        if get_vel:
            vx = -sna-ecc*(1+2*csa*sna)-1.5*ecc2*sna*(3*csa**2-1)
            vy = csa+ecc*(csa**2-sna**2)+1.5*ecc2*csa*(1-9*sna**2)
            vz = np.zeros(len(time))

            fact = EarthOrbitOmega_SI*AU_T
            vx *= fact
            vy *= fact
            vz *= fact

            return (np.array([x, y, z]), np.array([vx, vy, vz]))
        return np.array([x, y, z])

    def alpha_detecter(self, time, n):
        kappa = 2*PI_3*(n-1)+self.kappa
        return self.Omega*time+kappa-self.perigee

    def position_spa_Local(self, time, pp="all", get_vel=True):
        """
        Calculate the position of spacecrafts
        the orbits surface is (theta_s, phi_s)
        -------------------------------------
        Parameters:
        - time:
        - pp: for the spacecrafts, default is all
        - get_vel: whether to calculate the velocity
        """
        snp, csp = np.sin(self.phi_s), np.cos(self.phi_s)
        snt, cst = np.sin(self.theta_s), np.cos(self.theta_s)

        ndim = len(time)
        if pp == "all":
            ps = [0, 1, 2]
            if get_vel:
                xyz = np.zeros((6, 3, ndim))
            else:
                xyz = np.zeros((3, 3, ndim))
        elif pp != "all":
            ps = [int(pp[1])-1]
            if get_vel:
                xyz = np.zeros((2, 3, ndim))
            else:
                xyz = np.zeros((1, 3, ndim))
        else:
            print("Please input the write number of the spacecraft: p1, p2, p3")
            exit(0)

        for i in range(len(ps)):
            n = ps[i]+1
            alp = self.alpha_detecter(time, n)
            csa = np.cos(alp)
            sna = np.sin(alp)

            xyz[i, 0] = self.RT*(cst*csp*csa-snp*sna)
            xyz[i, 1] = self.RT*(cst*snp*csa+csp*sna)
            xyz[i, 2] = - self.RT*snt*csa

            if get_vel:
                if pp == "all":
                    j = i+2
                else:
                    j = 1

                xyz[j, 0] = self.RT*(-cst*csp*sna-snp*csa)*self.Omega
                xyz[j, 1] = self.RT*(-cst*snp*sna+csp*csa)*self.Omega
                xyz[j, 2] = self.RT*snt*sna*self.Omega

        return xyz


if __name__ == '__main__':
    print("Here is the analytical orbits")
    from pyINIDetectors import *

    import time

    st = time.time()

    TQ = INITianQin()
    TQOrbit = Orbit(TQ)

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
