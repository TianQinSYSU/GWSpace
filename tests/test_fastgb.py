#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_fastgb.py
# Author: En-Kun Li
# Mail: lienk@mail.sysu.edu.cn
# Created Time: 2023-09-07 10:05:09
# ==================================

import numpy as np
from gwspace import FastGB as FB
import matplotlib.pyplot as plt
from gwspace import libFastGB as _FB


def GenerateFastGB(p, Tobs, Cadence, oversample, TD=False):  # 默认TD = False，即生成频域信号。如果要生成时域信号那么需要标明TD = True
    # unfolding the parameters and producing the list of Params
    Amp = p[0]
    f0 = p[1]
    fdot = p[2]
    iota = p[3]
    psi = p[4]
    phi0 = p[5]
    EclLat = p[6]
    EclLon = p[7]

    Tobs = Tobs
    del_t = Cadence
    Ns = len(Amp)  # 源的个数

    prm = []
    for i in range(Ns):
        tmp = np.array([f0[i], fdot[i], EclLat[i], EclLon[i], Amp[i], iota[i], psi[i], phi0[i]])
        prm.append(tmp)
    # fastB = FB.FastGB("Test", dt=del_t, Tobs=Tobs, orbit="analytic")
    fastB = FB("Test", dt=del_t, Tobs=Tobs, orbit="analytic")

    if TD:
        Xt, Yt, Zt = fastB.TDI(T=Tobs, dt=del_t, simulator='synthlisa', table=prm,
                               algorithm='Michele', oversample=oversample)
        tm = np.arange(len(Xt))*del_t

        return tm, Xt, Yt, Zt

    else:
        Xf, Yf, Zf = fastB.fourier(T=Tobs, dt=del_t, simulator='synthlisa',
                                   table=prm, algorithm='Michele', oversample=oversample)
        return Xf.f, Xf[:], Yf[:], Zf[:]


def test_generate_date():
    f0 = [0.00622028]
    fd = [7.48528554e-16]
    beta = [-0.082205]  # ecliptic latitude [rad]
    Lambda = [2.10225]  # ecliptic longitude [rad]
    Amp = [6.37823e-23]
    iota = [0.645772]
    psi = [2.91617795]
    phi0 = [3.1716561]

    # 两年观测时长，15s采一个点。
    # 在fastGB程序中，Tobs=62914560和dt=15是默认值。
    # 这个15s的采样间隔是适用于LISA的，但实际操作中并不影响，因为它只是一个缺少输入条件时的默认值而已，你可以在这里输入任何你需要的Cadence值。
    # 在这里，仅作为一个初步的例子，我们用了15s的采样间隔。实际天琴的采样间隔可能是1s左右，你可以自行修改成更符合实情的Tobs和Cadence。
    Tobs = 62914560.0
    Cadence = 15.0

    paras = np.array([Amp, f0, fd, iota, psi, phi0, beta, Lambda])

    # 调用函数生成频域信号（如果需要时域，则在括号里再加一个TD = True即可）
    J0806_FD_highos = GenerateFastGB(paras, Tobs, Cadence, 32768)

    # 注意，这里的32768是什么？
    # 它是oversample的值，用于调节对“慢项”的降采样的程度。oversample值必须为2的倍数，否则傅里叶变换时无法用gsl_fft函数。
    # 在这里，原采样点数62914560/15=4194304，而“慢项”采样点数为128*oversample
    # （“128”的值根据Tobs和f0的不同而不同，具体可以本文档最后一部分）
    # 如果不进行降采样，则4194304=128*oversample，算出来oversample=32768刚好也是2的倍数，可以直接用。
    # 这样，就得到了不进行降采样情况下的J0806双星引力波频域波形。

    # 处理得到的数据，方便画图
    J0806_FD_highos = np.array(J0806_FD_highos).T
    J0806_FD_highos[:, 1:] = J0806_FD_highos[:, 1:]*2
    # df_J0806_highos = pd.DataFrame(np.abs(J0806_FD_highos),columns=['f','TDI X (FD)', 'TDI Y (FD)', 'TDI Z (FD)'])

    for i in range(1, 4):
        # plt.subplot(1,3,i)
        plt.plot(J0806_FD_highos[1:, 0], abs(J0806_FD_highos[1:, i]))
        plt.xlim(0.00620, 0.00624)
        plt.xscale('log')
        plt.yscale('log')

    plt.show()


def test_generate_date_with_FB():
    f0 = [0.00622028]
    fd = [7.48528554e-16]
    beta = [-0.082205]  # ecliptic latitude [rad]
    Lambda = [2.10225]  # ecliptic longitude [rad]
    Amp = [6.37823e-23]
    iota = [0.645772]
    psi = [2.91617795]
    phi0 = [3.1716561]

    Tobs = 62914560.0
    Cadence = 15.0

    paras = np.array([f0, fd, beta, Amp, Lambda, iota, psi, phi0])

    print(paras[:, 0])

    # dat = _FB.ComputeXYZ_FD(paras[:,0], 16, Tobs, Cadence)
    M = 1024
    XLS = np.zeros(2*M, 'd')
    YLS = np.zeros(2*M, 'd')
    ZLS = np.zeros(2*M, 'd')

    XSL = np.zeros(2*M, 'd')
    YSL = np.zeros(2*M, 'd')
    ZSL = np.zeros(2*M, 'd')

    _FB.ComputeXYZ_FD(paras[:, 0], M, Tobs, Cadence, XLS, YLS, ZLS, XSL, YSL, ZSL, 8, "TianQin")

    print(XLS.shape)
    print(XLS)

    print("This is dat")

    plt.plot(XLS)

    plt.show()

def transfer_slr(freq, LT, knl, krr, krs):
    return 0.5 * np.sinc(freq*LT * (1-knl)) * np.exp(-1j*freq*(LT+krr+krs))

def _dot_arr(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def _cal_zeta(u,v,n):
    nu = _dot_arr(n,u)
    nv = _dot_arr(n,v)
    zp = nu**2 - nv**2
    zc = 2*nu*nv
    return zp, zc

def _cal_xi(tf,k,r):
    return tf - _dot_arr(k,r)

def get_yslr(tf, freq, u,v,k, wf, detector="TianQin"):
    xt,yt,zt,LT = get_pos(tf, detector)
    ni = lambda i,j: np.array([xt[j]-xt[i], yt[j]-yt[i], zt[j]-zt[i]])/LT    
    n1 = ni(2,1); n2 = ni(0,2); n3 = ni(1,0)
    kn1 = _dot_arr(k,n1)
    kn2 = _dot_arr(k,n2)
    kn3 = _dot_arr(k,n3)

    kr1 = _dot_arr(k, np.array([xt[0],yt[0],zt[0]]))
    kr2 = _dot_arr(k, np.array([xt[1],yt[1],zt[1]]))
    kr3 = _dot_arr(k, np.array([xt[2],yt[2],zt[2]]))

    zp1,zc1 = _cal_zeta(u,v,n1)
    zp2,zc2 = _cal_zeta(u,v,n2)
    zp3,zc3 = _cal_zeta(u,v,n3)

    sinc12 = np.sinc(freq*LT*(1+kn3))
    sinc23 = np.sinc(freq*LT*(1+kn1))
    sinc31 = np.sinc(freq*LT*(1+kn2))

    sinc21 = np.sinc(freq*LT*(1-kn3))
    sinc32 = np.sinc(freq*LT*(1-kn1))
    sinc13 = np.sinc(freq*LT*(1-kn2))

    exp12 = np.exp(-1j*np.pi*freq*(LT+kr1+kr2))
    exp23 = np.exp(-1j*np.pi*freq*(LT+kr2+kr3))
    exp31 = np.exp(-1j*np.pi*freq*(LT+kr3+kr1))

    yslr = {}
    for i in range(4):        
        ts = tf - i*LT
        hp, hc = wf.get_hphc(ts)
        cspsi = np.cos(2*wf.psi) 
        snpsi = np.sin(2*wf.psi)
        hp, hc = (hp*cspsi -hc*snpsi, hp*snpsi+hc*cspsi)

        fact = -1j*np.pi*freq*LT * np.exp(1j*2*np.pi*freq*LT*i)
        
        yslr["%sL"%i] = {
            (1,2): fact*sinc12*exp12, #*(zp3*hp + zc3*hc),
            (2,1): fact*sinc21*exp12*(zp3*hp + zc3*hc),
            (2,3): fact*sinc23*exp23*(zp1*hp + zc1*hc),
            (3,2): fact*sinc32*exp23*(zp1*hp + zc1*hc),
            (3,1): fact*sinc31*exp31*(zp2*hp + zc2*hc),
            (1,3): fact*sinc13*exp31*(zp2*hp + zc2*hc),
        }
    return yslr

class GCBWaveformFD(BasicWaveform):
    
    def __init__(self, mass1, mass2, T_obs, phi0, f0, fdot=None, fddot=None, DL=1., Lambda=None, Beta=None,
                 phi_c=0., tc=0., iota=0., var_phi=0., psi=0, **kwargs):
        BasicWaveform.__init__(self, mass1, mass2, T_obs, DL, Lambda, Beta,
                               phi_c, tc, iota, var_phi, psi, **kwargs)
        self.phi0 = phi0
        self.f0 = f0
        if fdot is None:
            self.fdot = (96/5*PI**(8/3) *
                         (G_SI*self.Mc*MSUN_SI/C_SI**3)**(5/3)
                         * f0**(11/3))
        else:
            self.fdot = fdot
        if fddot is None:
            self.fddot = 11/3*self.fdot**2/f0
        else:
            self.fddot = fddot
        self.amp = 2*(G_SI*self.Mc*MSUN_SI)**(5/3)
        self.amp = self.amp/C_SI**4/(DL*MPC_SI)
        self.amp = self.amp*(PI*f0)**(2/3)

    def get_h_exp(self, tf, freq=None):
        phase = 2*PI*(self.f0+0.5*self.fdot*t +
                      1/6*self.fddot*t*t)*t+self.phi0
        csi = cos(self.iota)
        if freq == None:
            freq = self.f0 + self.fdot*tf
        return self.amp*np.exp(1j*2*np.pi*freq*tf)

if __name__ == "__main__":
    test_generate_date()
    test_generate_date_with_FB()
