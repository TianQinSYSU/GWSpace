#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_Orbits.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-09-05 17:55:52
# ==================================

import time
import numpy as np
import matplotlib.pyplot as plt

from gwspace.Constants import YRSID_SI, DAY
from gwspace.Orbit import TianQinOrbit
from gwspace.Orbit import get_pos

st = time.time()
Tobs = YRSID_SI
delta_T = 10
tf = np.arange(0, Tobs, delta_T)

TQOrbit = TianQinOrbit(tf)
ed = time.time()
print(f"Time cost for initial position: {ed-st} s")

plt.figure()
plt.plot(tf/DAY, TQOrbit.orbit_1[0], label='x1')
plt.plot(tf/DAY, TQOrbit.orbit_2[0], label='x2')
plt.plot(tf/DAY, TQOrbit.orbit_3[0], label='x3')
plt.xlabel('Time [Day]')
plt.ylabel('pos (x) [s]')
plt.xlim(90, 115)
plt.ylim(480, 495)
plt.legend()
plt.tight_layout()

plt.figure()
for i in range(3):
    plt.plot(tf/DAY, TQOrbit.p_0[i])
plt.tight_layout()

st = time.time()
x, y, z, L = get_pos(tf, detector="TianQin")
ed = time.time()

print(f"time cost of Cython is {ed-st} s")

plt.figure()
plt.plot(tf/DAY, x[0], label="x1")
plt.plot(tf/DAY, x[1], label="x2")
plt.plot(tf/DAY, x[2], label="x3")

plt.xlabel('Time [Day]')
plt.ylabel('pos (x) [s]')
plt.xlim(90, 115)
plt.ylim(480, 495)
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(tf/DAY, x[0], label="x1")
plt.plot(tf/DAY, y[1], label="y1")
plt.plot(tf/DAY, z[2], label="z1")

plt.xlabel('Time [Day]')
plt.ylabel('pos (x) [s]')
# plt.xlim(90, 115)
# plt.ylim(480, 495)
plt.legend()
plt.tight_layout()

###############################################
dt = 3600
tf = np.arange(0, const.YRSID_SI, dt)

xt_l, yt_l, zt_l, L_l = get_pos(tf, detector="LISA")
xt_tj, yt_tj, zt_tj, L_tj = get_pos(tf, detector="TaiJi")
xt_tq, yt_tq, zt_tq, L_tq = get_pos(tf, detector="TianQin")

def get_centre(x,y,z):
    return np.array([x[0]+x[1]+x[2],y[0]+y[1]+y[2],z[0]+z[1]+z[2]])/3

def get_theta(a, b, la, lb):
    costheta = (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])/la/lb
    theta = np.arccos(costheta)
    return theta

def get_norm(a):
    return np.sqrt(a[0]**2+a[1]**2+a[2]**2)

pos_l = get_centre(xt_l,yt_l,zt_l)
pos_tj = get_centre(xt_tj,yt_tj,zt_tj)
pos_tq = get_centre(xt_tq,yt_tq,zt_tq)

R_l = get_norm(pos_l)
R_tj = get_norm(pos_tj)
R_tq = get_norm(pos_tq)

theta_ltj = get_theta(pos_l, pos_tj, R_l, R_tj)
theta_ltq = get_theta(pos_l, pos_tq, R_l, R_tq)
theta_tjtq = get_theta(pos_tj, pos_tq, R_tj, R_tq)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

tt = tf/const.DAY

ax.plot(tt, theta_ltq/np.pi*180, '--r', label="LISA-TianQin")
ax.plot(tt, theta_tjtq/np.pi*180, '--b', label="TaiJi-TianQin")
ax.plot(tt, theta_ltj/np.pi*180-20, '--g', label=r"(LISA-TaiJi) - 40$^\circ$")

ax.axhline(20, color='gray', ls=':')

ax.set_ylim(17,22)

axins = ax.inset_axes((0.6, 0.1, 0.35, 0.3))
axins.plot(tt, theta_ltq/np.pi*180, '--r')
axins.plot(tt, theta_tjtq/np.pi*180, '--b')
axins.plot(tt, theta_ltj/np.pi*180-20, '--g')
axins.axhline(20, color='gray', ls=':')

axins.set_ylim(20-2.55e-3,20+2.45e-3)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Angle [degree]")

axins.set_xlabel("Time [day]")
axins.set_ylabel("Angle [degree]")

ax.legend(loc='best')

##plt.savefig("../../../TQ-SDS/figs/angle_sdb.pdf", dpi=360)

plt.show()
