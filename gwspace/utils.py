#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: utils.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
# Created Time: 2023-08-11 23:05:45
# ==================================
"""Useful tools for some basic calculations or conversions."""

import numpy as np
from scipy.special import factorial

from gwspace.constants import C_SI, H0_SI, Omega_m_Planck2018


def to_m1m2(m_chirp, eta):
    m1 = m_chirp/(2*eta**(3/5))*(1+(1-4*eta)**0.5)
    m2 = m_chirp/eta**(3/5)-m1
    return m1, m2


def luminosity_distance_approx(z, omega_m=Omega_m_Planck2018):
    """ An analytical approximation of the luminosity distance in flat cosmologies.
     See arxiv:1111.6396 """
    x_z = (1-omega_m)/omega_m/(1+z)**3
    Phix = lambda x: ((1+320*x+0.4415*x*x+0.02656*x**3) / (1+392*x+0.5121*x*x+0.03944*x**2))
    return 2*C_SI/H0_SI*(1+z)/np.sqrt(omega_m)*(
            Phix(0)-Phix(x_z)/np.sqrt(1+z))


def icrs_to_ecliptic(ra, dec, center='bary'):
    """Convert ICRS(Equatorial) to Ecliptic frame.
     https://docs.astropy.org/en/stable/coordinates/index.html
     Reminder: Both dec & latitude range (pi/2, -pi/2) [instead of (0, pi)].
    :param ra: float, right ascension
    :param dec: float, declination
    :param center: {'bary', str}, 'bary' or 'geo'  # Actually it won't have too much difference
    :return: longitude, latitude: float
    """
    from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic, GeocentricTrueEcliptic
    import astropy.units as u

    co = SkyCoord(ra*u.rad, dec*u.rad)
    if center == 'bary':
        cot = co.transform_to(BarycentricTrueEcliptic)
    elif center == 'geo':
        cot = co.transform_to(GeocentricTrueEcliptic)
    else:
        raise ValueError("'center' should be 'bary' or 'geo'")
    return cot.lon.rad, cot.lat.rad  # (Lambda, Beta)


def ssb_to_tianqin_frame(lam, beta):
    """Convert Ecliptic frame to TianQin detector frame.
     Matrix of ecliptic to detector: see Hu et al. https://iopscience.iop.org/article/10.1088/1361-6382/aab52f
     Reminder: Both beta & latitude range (-pi/2, pi/2).
    :param lam: float, ecliptic longitude (Lambda)
    :param beta: float, ecliptic latitude (Beta)
    :return: longitude, latitude: float
    """
    from astropy.coordinates import cartesian_to_spherical

    # from gwspace.constants import J0806_phi, J0806_theta
    # phi_j, theta_j = J0806_phi, np.pi/2 - J0806_theta
    # rz = np.array([[np.sin(phi_j), np.cos(phi_j), 0],
    #                [-np.cos(phi_j), np.sin(phi_j), 0],
    #                [0, 0, 1]])
    # rx = np.array([[1, 0, 0],
    #                [0, np.sin(theta_j), np.cos(theta_j)],
    #                [0, -np.cos(theta_j), np.sin(theta_j)]])
    # trans = rz.dot(rx)
    # trans_inv = np.linalg.inv(trans)

    car = np.array([np.cos(beta)*np.cos(lam), np.cos(beta)*np.sin(lam), np.sin(beta)])
    trans_inv = np.array([[0.8616291604415259,   0.5075383629607039,   0.],
                          [0.04158693653353248, -0.07060060839873289, -0.9966373868180366],
                          [-0.5058317077710601,  0.8587318348686612,  -0.08193850863004093]])
    car_t = cartesian_to_spherical(*trans_inv.dot(car))
    return car_t[2].rad, car_t[1].rad  # lon, lat


def tianqin_frame_to_ssb(lon, lat):
    """Convert TianQin detector frame to Ecliptic frame.
     Matrix of ecliptic to detector: see Hu et al. https://iopscience.iop.org/article/10.1088/1361-6382/aab52f
     Reminder: Both beta & latitude range (-pi/2, pi/2).
    :param lon: float, longitude in detector frame
    :param lat: float, latitude in detector frame
    :return: Lambda, Beta: float
    """
    from astropy.coordinates import cartesian_to_spherical

    car = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    trans = np.array([[0.8616291604415259,  0.04158693653353248, -0.5058317077710601],
                      [0.5075383629607039, -0.07060060839873289,  0.8587318348686612],
                      [0.,                 -0.9966373868180366,  -0.08193850863004093]])
    car_t = cartesian_to_spherical(*trans.dot(car))
    return car_t[2].rad, car_t[1].rad  # (Lambda, Beta)


def dfridr(func, x, h, err=1e-14, *args):
    """
    Parameters:
        func: external function
        x: point or array
        h: initial stepsize
        err: error
    -------------------------------------------------------------------------
    Returns the derivative of a function `func` at a point `x` by Ridders' method
    of polynomial extrapolation. The value `h` is input as an estimated initial
    stepsize; It need not be small, but rather should be an increment in `x` over
    which func changes substantially. An estimate of the error in the derivative
    is returned as err.
    Parameters: Stepsize is decreased by `CON` at each iteration. Max size of
        tableau is set by `NTAB`. Return when error is `SAFE` worse than the best
        so far.
    """
    CON = 1.4
    CON2 = CON*CON
    BIG = 1e30
    NTAB = 10
    SAFE = 2.0

    a = np.zeros((NTAB, NTAB))
    if h == 0:
        raise ValueError('h must be nonzero in dfridr')
        # sys.exit(0)
    hh = h
    a[0, 0] = (func(x+hh, *args)-func(x-hh))/(2.0*hh)
    err = BIG
    for i in range(1, NTAB):
        hh = hh/CON
        a[0, i] = (func(x+hh, *args)-func(x-hh))/(2.0*hh)
        fac = CON2
        for j in range(1, i):
            a[j, i] = (a[j-1, i]*fac-a[j-1, i-1])/(fac-1)
            fac = CON2*fac
            errt = max(np.abs(a[j, -i]-a[j-1, i]), np.abs(a[j, i]-a[j-1, i-1]))
            if errt <= err:
                err = errt
                df = a[j, i]
        if np.abs(a[i, i]-a[i-1, i-1]) >= SAFE*err:
            return df
    return df


def QuadLagrange3(x, y):
    """
    Quadratic Lagrange interpolation polynomial of degree 2
    -------------------------------------------------------
    Parameters:
        x: length of 3
        y: length of 3
    Return:
        res: array with length of 3
    -------------------------------------------
    Reference:
        https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html
        OR
        http://mathonline.wikidot.com/quadratic-lagrange-interpolating-polynomials
    """
    res = np.zeros(3, dtype=y.dtype)
    if (not len(x) == 3) or (not len(y) == 3):
        raise ValueError('Only allows an input length of 3 for x and y.')
    c0 = y[0]/((x[0]-x[1])*(x[0]-x[2]))
    c1 = y[1]/((x[1]-x[0])*(x[1]-x[2]))
    c2 = y[2]/((x[2]-x[0])*(x[2]-x[1]))
    res[0] = c0*x[1]*x[2]+c1*x[2]*x[0]+c2*x[0]*x[1]
    res[1] = -c0*(x[1]+x[2])-c1*(x[2]+x[0])-c2*(x[0]+x[1])
    res[2] = c0+c1+c2
    return res


def QuadLagrangeInterpolate(x, res):
    """
    Quadratic Lagrange Interpolating Polynomials.
    --------------------------------------------------------
    """
    ss = res[0]
    n = len(res)
    for i in range(1, n):
        ss += res[i]*x**i
    return ss


def Factorial(n):
    """ Ref: https://mathworld.wolfram.com/BinomialCoefficient.html """
    if n < 0:
        return np.inf
    else:
        return factorial(n)


def sYlm(s, l, m, theta, phi):
    r""" Spin Weighted Spherical Harmonics: {}_s Y_{lm}(\theta, \phi)

    :param s: spin
    :param l:
    :param m:
    :param theta:
    :param phi:
    """
    if l < abs(s):
        raise ValueError(f"abs spin |s| = {abs(s)} can not be larger than l = {l}")
    if l < abs(m):
        raise ValueError(f"mode |m| = {abs(m)} can not be larger than l = {l}")
    tp = (-1)**(l+m-s)*np.sqrt((2*l+1)/4/np.pi)
    tp *= np.exp(1j*m*phi)
    snt = np.sin(theta/2)
    cst = np.cos(theta/2)
    d1 = Factorial(l+m)*Factorial(l-m)*Factorial(l+s)*Factorial(l-s)
    d1 = np.sqrt(d1)

    def dslm(k):
        d0 = (-1)**(-k)
        d2 = Factorial(l+m-k)*Factorial(l-s-k)*Factorial(k)*Factorial(k+s-m)
        dc = snt**(m-s-2*k+2*l)
        ds = cst**(2*k+s-m)
        d3 = (dc*ds)
        return d0/d2*d3

    k1 = max(0, m-s)
    k2 = min(l+m, l-s)
    tps = 0
    for k in range(k1, k2+1):
        tps += dslm(k)

    return tp*d1*tps


# # FOR ARCHIVE ONLY
# def spin_weighted_spherical_harmonic(s, l, m, theta, phi):
#     # Currently only supports s=-2, l=2,3,4,5 modes
#     from numpy import pi, sqrt, cos, sin, exp
#     func = "SpinWeightedSphericalHarmonic"
#     if l < abs(s):
#         raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |s| <= l\n' % (func, s, l, m))
#     if l < abs(m):
#         raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n' % (func, s, l, m))
#     if not (s == -2):
#         raise ValueError('Error - %s: Invalid mode s=%d - only s=-2 implemented\n' % (func, s))
#
#     fac = {
#         # l=2
#         (2, -2): sqrt(5.0/(64.0*pi))*(1.0-cos(theta))*(1.0-cos(theta)),
#         (2, -1): sqrt(5.0/(16.0*pi))*sin(theta)*(1.0-cos(theta)),
#         (2, 0): sqrt(15.0/(32.0*pi))*sin(theta)*sin(theta),
#         (2, 1): sqrt(5.0/(16.0*pi))*sin(theta)*(1.0+cos(theta)),
#         (2, 2): sqrt(5.0/(64.0*pi))*(1.0+cos(theta))*(1.0+cos(theta)),
#         # l=3
#         (3, -3): sqrt(21.0/(2.0*pi))*cos(theta/2.0)*pow(sin(theta/2.0), 5.0),
#         (3, -2): sqrt(7.0/(4.0*pi))*(2.0+3.0*cos(theta))*pow(sin(theta/2.0), 4.0),
#         (3, -1): sqrt(35.0/(2.0*pi))*(sin(theta)+4.0*sin(2.0*theta)-3.0*sin(3.0*theta))/32.0,
#         (3, 0): (sqrt(105.0/(2.0*pi))*cos(theta)*pow(sin(theta), 2.0))/4.0,
#         (3, 1): -sqrt(35.0/(2.0*pi))*(sin(theta)-4.0*sin(2.0*theta)-3.0*sin(3.0*theta))/32.0,
#         (3, 2): sqrt(7.0/(4.0*pi))*(-2.0+3.0*cos(theta))*pow(cos(theta/2.0), 4.0),
#         (3, 3): -sqrt(21.0/(2.0*pi))*pow(cos(theta/2.0), 5.0)*sin(theta/2.0),
#         # l=4
#         (4, -4): 3.0*sqrt(7.0/pi)*pow(cos(theta/2.0), 2.0)*pow(sin(theta/2.0), 6.0),
#         (4, -3): 3.0*sqrt(7.0/(2.0*pi))*cos(theta/2.0)*(1.0+2.0*cos(theta))*pow(sin(theta/2.0), 5.0),
#         (4, -2): (3.0*(9.0+14.0*cos(theta)+7.0*cos(2.0*theta))*pow(sin(theta/2.0), 4.0))/(4.0*sqrt(pi)),
#         (4, -1): (3.0*(3.0*sin(theta)+2.0*sin(2.0*theta)+7.0*sin(3.0*theta)-7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
#         (4, 0): (3.0*sqrt(5.0/(2.0*pi))*(5.0+7.0*cos(2.0*theta))*pow(sin(theta), 2.0))/16.0,
#         (4, 1): (3.0*(3.0*sin(theta)-2.0*sin(2.0*theta)+7.0*sin(3.0*theta)+7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
#         (4, 2): (3.0*pow(cos(theta/2.0), 4.0)*(9.0-14.0*cos(theta)+7.0*cos(2.0*theta)))/(4.0*sqrt(pi)),
#         (4, 3): -3.0*sqrt(7.0/(2.0*pi))*pow(cos(theta/2.0), 5.0)*(-1.0+2.0*cos(theta))*sin(theta/2.0),
#         (4, 4): 3.0*sqrt(7.0/pi)*pow(cos(theta/2.0), 6.0)*pow(sin(theta/2.0), 2.0),
#         # l= 5
#         (5, -5): sqrt(330.0/pi)*pow(cos(theta/2.0), 3.0)*pow(sin(theta/2.0), 7.0),
#         (5, -4): sqrt(33.0/pi)*pow(cos(theta/2.0), 2.0)*(2.0+5.0*cos(theta))*pow(sin(theta/2.0), 6.0),
#         (5, -3): (sqrt(33.0/(2.0*pi))*cos(theta/2.0)*(17.0+24.0*cos(theta)+15.0*cos(2.0*theta))*pow(sin(theta/2.0), 5.0))/4.0,
#         (5, -2): (sqrt(11.0/pi)*(32.0+57.0*cos(theta)+36.0*cos(2.0*theta)+15.0*cos(3.0*theta))*pow(sin(theta/2.0), 4.0))/8.0,
#         (5, -1): (sqrt(77.0/pi)*(2.0*sin(theta)+8.0*sin(2.0*theta)+3.0*sin(3.0*theta)+12.0*sin(4.0*theta)-15.0*sin(5.0*theta)))/256.0,
#         (5, 0): (sqrt(1155.0/(2.0*pi))*(5.0*cos(theta)+3.0*cos(3.0*theta))*pow(sin(theta), 2.0))/32.0,
#         (5, 1): sqrt(77.0/pi)*(-2.0*sin(theta)+8.0*sin(2.0*theta)-3.0*sin(3.0*theta)+12.0*sin(4.0*theta)+15.0*sin(5.0*theta))/256.0,
#         (5, 2): sqrt(11.0/pi)*pow(cos(theta/2.0), 4.0)*(-32.0+57.0*cos(theta)-36.0*cos(2.0*theta)+15.0*cos(3.0*theta))/8.0,
#         (5, 3): -sqrt(33.0/(2.0*pi))*pow(cos(theta/2.0), 5.0)*(17.0-24.0*cos(theta)+15.0*cos(2.0*theta))*sin(theta/2.0)/4.0,
#         (5, 4): sqrt(33.0/pi)*pow(cos(theta/2.0), 6.0)*(-2.0+5.0*cos(theta))*pow(sin(theta/2.0), 2.0),
#         (5, 5): -sqrt(330.0/pi)*pow(cos(theta/2.0), 7.0)*pow(sin(theta/2.0), 3.0)
#     }.get((l, m), None)
#     if fac is None:
#         raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n' % (func, s, l, m))
#
#     if m == 0:
#         return fac
#     else:
#         return fac*exp(1j*m*phi)


def frequency_noise_from_psd(psd, delta_f, seed=None):
    """ Create noise with a given psd.
    Return noise coloured with the given psd. The return noise
    FrequencySeries has the same length and frequency step as the
    given psd. Note that if unique noise is desired a unique
    seed should be provided.

    :param psd: The noise weighting to color the noise.
    :param delta_f:
    :param seed: in range (0, int) or None
        The seed to generate the noise. If None specified, the seed will not be reset.
    :return: noise: numpy array containing gaussian noise colored by the given psd.
    """
    sigma = (0.5*psd/delta_f)**0.5
    if seed is not None:
        np.random.seed(seed)

    not_zero = (sigma != 0)
    sigma_red = sigma[not_zero]
    noise_re = np.random.normal(0, sigma_red)
    noise_im = np.random.normal(0, sigma_red)

    # rr = lambda sig: np.random.multivariate_normal([0,0], np.eye(2)*sig)
    # noise_re, noise_im = np.array(list(map(rr, sigma_red))).T

    noise_red = noise_re+1j*noise_im

    noise = np.zeros(len(sigma), dtype=np.complex128)
    noise[not_zero] = noise_red

    return noise
