#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: utils.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-11 23:05:45
# ==================================

import numpy as np
from numpy import pi, sqrt, cos, sin, exp
import yaml
import sys, time
# from numpy import pi, conjugate, dot, sqrt, cos, sin, tan, exp, real, imag, arccos, arcsin, arctan, arctan2
# import pyfftw as fft
# import sys

from csgwsim.Constants import C_SI

def sinc(x):
    '''
    return sinc(x)
    -----------------
    be careful that in numpy
    sinc(x) = sin(pi x)/(pi x)
    '''
    return np.sinc(x/np.pi)

class parameter(object):
    """
    Archive the parameters as {value, unit}
    -------
    Parameters:

    """

    def __init__(self, value, unit="1", toSeconds=False):
        self.unit = unit
        if toSeconds:
            self.value = value*ParsValue2Second[unit]()
        else:
            self.value = value

    def __call__(self):
        return self.value


def Length2Second():
    return 1.0/C_SI


ParsValue2Second = {
    "m": Length2Second,
}


## ==========================
def dot_arr(u, v):
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]


def dot_arr_H_arr(u, H, v):
    """
    Calculate u cdot H cdot v
    --------------------------
    Parameter:
    - u: np array with shape of (3, ndim)
    - v: np array with shape of (3, ndim)
    - H: np array with shape of (3, 3)
    ------------------------------------
    Return:
    - ss = n.H.n
    """
    ss = np.zeros_like(u.shape[1])
    for i in range(3):
        for j in range(3):
            ss += u[i]*H[i, j]*v[j]
    return ss

def get_uvk(lambd, beta):
    snl = np.sin(lambd)
    csl = np.cos(lambd)
    snb = np.sin(beta)
    csb = np.cos(beta)

    u = np.array([snl, -csl, 0])
    v = np.array([-snb * csl, - sinb*snl, csb])
    k = np.array([-csb * csl, -snl*csb, -snb])
    return (u,v,k)

def polarization_tensor(u, v):
    e_p = np.zeros((3,3))
    e_c = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            e_p[i,j] = u[i] * u[j] - v[i] * v[j]
            e_c[i,j] = u[i] * v[j] + v[i] * u[j]

    return (e_p, e_c)


def cal_zeta(u, v, nl):
    """
    Calculate xi^+ and xi^x
    ------------------------------
    Parameters
    ----------
    - u, v: polarization coordinates
    - nl: unit vector from sender to receiver

    Return
    ------
    - n otimes (\epsilon_+ \epsilon_x) otimes n
    """
    xi_p = (dot_arr(u, nl))**2-(dot_arr(v, nl))**2
    xi_c = 2*dot_arr(u, nl)*dot_arr(v, nl)
    return xi_p, xi_c


def to_m1m2(m_chirp, eta):
    m1 = m_chirp/(2*eta**(3/5))*(1+(1-4*eta)**0.5)
    m2 = m_chirp/eta**(3/5)-m1
    return m1, m2


## ==========================

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

    co = SkyCoord(ra * u.rad, dec * u.rad)
    if center == 'bary':
        cot = co.transform_to(BarycentricTrueEcliptic)
    elif center == 'geo':
        cot = co.transform_to(GeocentricTrueEcliptic)
    else:
        raise ValueError("'center' should be 'bary' or 'geo'")
    return cot.lon.rad, cot.lat.rad  # (Lambda, Beta)


def deg2rad(deg):
    """
    Convert degree to radian
    """
    return deg/180*np.pi


def rad2deg(rad):
    """
    Convert radian to degree
    """
    return rad/np.pi*180


def yaml_readinpars(filename):
    """
    Read in parameters to dict from filename
    """
    with open(filename, 'r', encoding='utf-8') as fp:
        temp = yaml.load(fp.read(), Loader=yaml.Loader)
    return temp


def readvalue(pars, so, s1):
    """
    Convert str in dict to float
    Convert angles in degree to radian
    Convert length to SI unit [m]
    """
    ss = pars[so][s1]['value']
    try:
        unit = pars[so][s1]['unit']
    except:
        unit = None

    # print('the unit of %s is %s'%(ss, unit))

    if type(ss) is list:
        fs = np.array([float(s) for s in ss])
    else:
        fs = float(ss)

    # degree to radian
    if unit == 'degree' or unit == 'deg':
        fs = deg2rad(fs)
    elif (unit is None) or unit == 'radian' or unit == 'm' or unit == 's':
        return fs
    else:
        fs *= eval(unit)

    return fs


def readvalue_dict(pars, so):
    """
    Convert str in dict to float
    Convert angles in degree to radian
    Convert length to SI unit [m]
    """
    ss = pars[so]['value']
    try:
        unit = pars[so]['unit']
    except:
        unit = None

    # print('the unit of %s is %s'%(ss, unit))

    if type(ss) is list:
        fs = np.array([float(s) for s in ss])
    else:
        fs = float(ss)

    # degree to radian
    if unit == 'degree' or unit == 'deg':
        fs = deg2rad(fs)
    elif unit == None or unit == 'radian' or unit == 'm' or unit == 's':
        return fs
    else:
        fs *= eval(unit)

    return fs


## =======================================
def sinc(x):
    return np.sin(x)/x


def VectorDirectProduct(u, v):
    """
    Vector direct product
    ---------------------
    Parameter:
        u: array like of m dimension
        v: array like of n dimension
    Return:
        u v: matrix of dimension (m,n)
    """
    # m = np.shape(u)[0]
    # n = np.shape(v)[0]
    # uv = np.zeros((m,n))
    # for i in range(m):
    #    for j in range(n):
    #        uv[i,j] = u[i] * v[j]
    # return uv
    return np.outer(u, v)


def DoubleContraction(A, B):
    """
    Double Contraction
    ------------------
    Parameter:
        A: matrix, order is equal to or bigger than two
        B: matrix, order is equal to or bigger than two
        A and B must have the same dimension (m,n)
    Return:
        A:B = \sum_{ij} A_{ij}B^{ij}
        one number
    """
    # m,n = np.shape(A)
    # ss = 0
    # for i in range(m):
    #    for j in range(n):
    #        ss += A[i,j] * B[i,j]
    # return ss
    return np.sum(A*B)


"""
def FourierTransformData(x, dt, wis=None):
    '''
    Fourier transform data
    ---------------------------
    Parameters:
        x: the input data
        dt: the step in the reference array
    ---------------------------------------
    Return:
        transform of the data
    '''
    N = len(x)
    yt = fft.empty_aligned(N, dtype="float64")
    yf = fft.empty_aligned(int(N/2+1), dtype="complex128")
    fft_object = fft.FFTW(yt, yf, flags=('FFTW_ESTIMATE', ))

    yt = np.copy(x)
    yf = np.copy(fft_object(yt*dt))

    return (yf)
"""


## =======================================
def Rotation_3D(axis, theta):
    """
    Rotation around the axis of theta.
    -----------------------------------
    Parameters:
        axis: the axis of the index, number of 0,1,2 or x,y,z
        theta: rotation of angle
    Return:
        3x3 matrix
    """
    if axis == 2 or axis == 'y':
        ii = -1
    elif axis == 'x':
        ii = 0
    elif axis == 'z':
        ii = 2
    else:
        ii = axis

    R = np.eye(3)
    R[ii-1, ii-1] = np.cos(theta)
    R[ii+1, ii+1] = np.cos(theta)
    R[ii+1, ii-1] = np.sin(theta)
    R[ii-1, ii+1] = -np.sin(theta)
    return R


## =======================================
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


## =======================================
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


def QuadLagrangeInterpolat(x, res):
    """
    Quadratic Lagrange Interpolating Polynomials.
    --------------------------------------------------------
    """
    ss = res[0]
    n = len(res)
    for i in range(1, n):
        ss += res[i]*x**i
    return ss


# =============================================
# Currently only supports s=-2, l=2,3,4,5 modes
def SpinWeightedSphericalHarmonic(s, l, m, theta, phi):
    func = "SpinWeightedSphericalHarmonic"
    # Sanity checks
    if l < abs(s):
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |s| <= l\n' % (func, s, l, m))
    if l < abs(m):
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n' % (func, s, l, m))
    if not (s == -2):
        raise ValueError('Error - %s: Invalid mode s=%d - only s=-2 implemented\n' % (func, s))

    fac = {
        # l=2
        (2, -2): sqrt(5.0/(64.0*pi))*(1.0-cos(theta))*(1.0-cos(theta)),
        (2, -1): sqrt(5.0/(16.0*pi))*sin(theta)*(1.0-cos(theta)),
        (2, 0): sqrt(15.0/(32.0*pi))*sin(theta)*sin(theta),
        (2, 1): sqrt(5.0/(16.0*pi))*sin(theta)*(1.0+cos(theta)),
        (2, 2): sqrt(5.0/(64.0*pi))*(1.0+cos(theta))*(1.0+cos(theta)),
        # l=3
        (3, -3): sqrt(21.0/(2.0*pi))*cos(theta/2.0)*pow(sin(theta/2.0), 5.0),
        (3, -2): sqrt(7.0/(4.0*pi))*(2.0+3.0*cos(theta))*pow(sin(theta/2.0), 4.0),
        (3, -1): sqrt(35.0/(2.0*pi))*(sin(theta)+4.0*sin(2.0*theta)-3.0*sin(3.0*theta))/32.0,
        (3, 0): (sqrt(105.0/(2.0*pi))*cos(theta)*pow(sin(theta), 2.0))/4.0,
        (3, 1): -sqrt(35.0/(2.0*pi))*(sin(theta)-4.0*sin(2.0*theta)-3.0*sin(3.0*theta))/32.0,
        (3, 2): sqrt(7.0/(4.0*pi))*(-2.0+3.0*cos(theta))*pow(cos(theta/2.0), 4.0),
        (3, 3): -sqrt(21.0/(2.0*pi))*pow(cos(theta/2.0), 5.0)*sin(theta/2.0),
        # l=4
        (4, -4): 3.0*sqrt(7.0/pi)*pow(cos(theta/2.0), 2.0)*pow(sin(theta/2.0), 6.0),
        (4, -3): 3.0*sqrt(7.0/(2.0*pi))*cos(theta/2.0)*(1.0+2.0*cos(theta))*pow(sin(theta/2.0), 5.0),
        (4, -2): (3.0*(9.0+14.0*cos(theta)+7.0*cos(2.0*theta))*pow(sin(theta/2.0), 4.0))/(4.0*sqrt(pi)),
        (4, -1): (3.0*(3.0*sin(theta)+2.0*sin(2.0*theta)+7.0*sin(3.0*theta)-7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
        (4, 0): (3.0*sqrt(5.0/(2.0*pi))*(5.0+7.0*cos(2.0*theta))*pow(sin(theta), 2.0))/16.0,
        (4, 1): (3.0*(3.0*sin(theta)-2.0*sin(2.0*theta)+7.0*sin(3.0*theta)+7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
        (4, 2): (3.0*pow(cos(theta/2.0), 4.0)*(9.0-14.0*cos(theta)+7.0*cos(2.0*theta)))/(4.0*sqrt(pi)),
        (4, 3): -3.0*sqrt(7.0/(2.0*pi))*pow(cos(theta/2.0), 5.0)*(-1.0+2.0*cos(theta))*sin(theta/2.0),
        (4, 4): 3.0*sqrt(7.0/pi)*pow(cos(theta/2.0), 6.0)*pow(sin(theta/2.0), 2.0),
        # l= 5
        (5, -5): sqrt(330.0/pi)*pow(cos(theta/2.0), 3.0)*pow(sin(theta/2.0), 7.0),
        (5, -4): sqrt(33.0/pi)*pow(cos(theta/2.0), 2.0)*(2.0+5.0*cos(theta))*pow(sin(theta/2.0), 6.0),
        (5, -3): (sqrt(33.0/(2.0*pi))*cos(theta/2.0)*(17.0+24.0*cos(theta)+15.0*cos(2.0*theta))*pow(sin(theta/2.0), 5.0))/4.0,
        (5, -2): (sqrt(11.0/pi)*(32.0+57.0*cos(theta)+36.0*cos(2.0*theta)+15.0*cos(3.0*theta))*pow(sin(theta/2.0), 4.0))/8.0,
        (5, -1): (sqrt(77.0/pi)*(2.0*sin(theta)+8.0*sin(2.0*theta)+3.0*sin(3.0*theta)+12.0*sin(4.0*theta)-15.0*sin(5.0*theta)))/256.0,
        (5, 0): (sqrt(1155.0/(2.0*pi))*(5.0*cos(theta)+3.0*cos(3.0*theta))*pow(sin(theta), 2.0))/32.0,
        (5, 1): sqrt(77.0/pi)*(-2.0*sin(theta)+8.0*sin(2.0*theta)-3.0*sin(3.0*theta)+12.0*sin(4.0*theta)+15.0*sin(5.0*theta))/256.0,
        (5, 2): sqrt(11.0/pi)*pow(cos(theta/2.0), 4.0)*(-32.0+57.0*cos(theta)-36.0*cos(2.0*theta)+15.0*cos(3.0*theta))/8.0,
        (5, 3): -sqrt(33.0/(2.0*pi))*pow(cos(theta/2.0), 5.0)*(17.0-24.0*cos(theta)+15.0*cos(2.0*theta))*sin(theta/2.0)/4.0,
        (5, 4): sqrt(33.0/pi)*pow(cos(theta/2.0), 6.0)*(-2.0+5.0*cos(theta))*pow(sin(theta/2.0), 2.0),
        (5, 5): -sqrt(330.0/pi)*pow(cos(theta/2.0), 7.0)*pow(sin(theta/2.0), 3.0)
    }.get((l, m), None)
    if fac is None:
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n'%(func, s, l, m))

    # Result
    if m == 0:
        return fac
    else:
        return fac*exp(1j*m*phi)


## factorial
def Factorial(n):
    """
    ------------------------------------------------------
    Refs:
    https://mathworld.wolfram.com/BinomialCoefficient.html
    """
    if n < 0:
        return np.inf
    elif n == 0:
        return 1
    return n*Factorial(n-1)


## binominal coefficient
def BinomialCoefficient(n, k):
    """
    Binomial Coefficient
    ---------------------
    $$
    \binom{n}{k} =
        \begin{cases}
            \frac{n!}{k! (n-k)!} & \text{ for } 0 \leq k < n \\
            0 & \text{otherwise}
        \end{cases}
    $$
    ---------------------------------------------------------
    Refs:
    https://mathworld.wolfram.com/BinomialCoefficient.html
    """
    if 0 <= k <= n:
        return Factorial(n)/Factorial(k)/Factorial(n-k)
    # else:
    #    print("%d is not less than %d, or %d is smaller than 0\n"%(k, n, k))
    return 0


## spin-weighted spherical harmonics
def sYlm(s, l, m, theta, phi):
    """
    Spin Weighted Spherical Harmonics
    ---------------------------------
    the spin-weight
    {}_s Y_{lm}(\theta, \phi)
    -->
    Parameters:
        s: spin
        l:
        m:
        theta:
        phi:

    """
    if l < abs(s): print(f"Error - abs spin |s| = {abs(s)} can not be larger than l = {l}")
    if l < abs(m): print(f"Error - mode |m| = {abs(m)} can not be larger than l = {l}")
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


##+====================================
def epsilon(i, j, k):
    """
    epsilon tensor or the permutation symbol or the Levi-Civita symbol
    -------------------------------------------------------------------
    Parameters:
        i,j,k: three int values

    Return:
        epsilon_ijk =  0 if any two labels are the same
                    =  1 if i,j,k is an even permutation of 1,2,3
                    = -1 if i,j,k is an odd permutation of 1,2,3
    Refs:
    https://mathworld.wolfram.com/PermutationSymbol.html
    """
    if i == j or j == k or k == i:
        return 0
    ss = 0
    if j < i: ss += 1
    if k < j: ss += 1
    if k < i: ss += 1
    sp = ss % 2
    if sp == 0:
        return 1
    return -1

class countdown(object):
    def __init__(self,totalitems,interval=10000):
        self.t0    = time.time()
        self.total = totalitems
        self.int   = interval

        self.items = 0
        self.tlast = self.t0
        self.ilast = 0

        self.longest = 0

    def pad(self,outstring):
        if len(outstring) < self.longest:
            return outstring + " " * (self.longest - len(outstring))
        else:
            self.longest = len(outstring)
            return outstring

    def status(self,local=False,status=None):
        self.items = self.items + 1

        if self.items % self.int != 0:
            return

        t = time.time()

        speed      = float(self.items) / (t - self.t0)
        localspeed = float(self.items - self.ilast) / (t - self.tlast)

        if self.total != 0:
            eta      = (self.total - self.items) / speed
            localeta = (self.total - self.items) / localspeed
        else:
            eta, localeta = 0, 0

        self.tlast = t
        self.ilast = self.items

        print(self.pad("\r%d/%d done, %d s elapsed (%f/s), ETA %d s%s" % (self.items,self.total,t - self.t0,
                                                                          localspeed if local else speed,
                                                                          localeta if local else eta,
                                                                          ", " + status if status else "")   ), end=' ')
        sys.stdout.flush()

    def end(self,status=None):
        t = time.time()

        speed = self.items / (t - self.t0)

        print(self.pad("\r%d finished, %d s elapsed (%d/s)%s" % (self.items,t - self.t0,speed,", " + status if status else "")))
        sys.stdout.flush()


class FrequencyArray(np.ndarray):
    """
    Class to manage array in frequency domain based numpy array class
    """
    def __new__(subtype,data,dtype=None,copy=False,df=None,kmin=None):
        """
        ...
        @param data is ... [required]
        @param dtype is ... [default: None]
        @param copy is ... [default: None]
        @param df is ... [default: None]
        @param kmin is ... [default: None]
        @return
        """
        # make sure we are working with an array, copy the data if requested,
        # then transform the array to our new subclass
        subarr = np.array(data,dtype=dtype,copy=copy)
        subarr = subarr.view(subtype)

        # get df and kmin preferentially from the initialization,
        # then from the data object, otherwise set to None
        subarr.df   = df   if df   is not None else getattr(data,'df',  None)
        subarr.kmin = int(kmin) if kmin is not None else getattr(data,'kmin',0)

        return subarr


    def __array_wrap__(self,out_arr,context=None):
        """
        ...
        @param out_arr is ... [required]
        @param context is ... [default: None]
        @return
        """
        out_arr.df, out_arr.kmin = self.df, self.kmin

        return np.ndarray.__array_wrap__(self,out_arr,context)


    def __getitem__(self,key):
        """
        ...
        @param key is ... [required]
        @return
        """
        return self.view(np.ndarray)[key]


    def __getslice__(self,i,j):
        """
        ...
        @param i is ... [required]
        @param j is ... [required]
        @return
        """
        return self.view(np.ndarray)[i:j]

    # def __array_finalize__(self,obj):
    #    if obj is None: return
    #    self.df   = getattr(obj,'df',  None)
    #    self.kmin = getattr(obj,'kmin',None)


    def __repr__(self):
        """
        ...
        @return
        """
        if self.df is not None:
            return 'Frequency array (f0=%s,df=%s): %s' % (self.kmin * self.df,self.df,self)
        else:
            return str(self)


    def __add__(self,other):
        """
        Combine two FrequencyArrays into a longer one by adding intermediate zeros if necessary
        @param other is ... [required]
        @return
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            beg = min(self.kmin,other.kmin)
            end = max(self.kmin + len(self),other.kmin + len(other))

            ret = np.zeros(end-beg,dtype=np.find_common_type([self.dtype,other.dtype],[]))

            ret[(self.kmin  - beg):(self.kmin  - beg + len(self))]   = self
            ret[(other.kmin - beg):(other.kmin - beg + len(other))] += other

            return FrequencyArray(ret,kmin=beg,df=self.df)

        # fall back to simple arrays (may waste memory)
        return np.ndarray.__add__(self,other)


    def __sub__(self,other):
        """
        ...
        same behavior as __add__: TO DO -- consider restricting the result to the extend of the first array
        @param other is ... [required]
        @return
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            beg = min(self.kmin,other.kmin)
            end = max(self.kmin + len(self),other.kmin + len(other))

            ret = np.zeros(end-beg,dtype=np.find_common_type([self.dtype,other.dtype],[]))

            ret[(self.kmin  - beg):(self.kmin  - beg + len(self))]   = self
            ret[(other.kmin - beg):(other.kmin - beg + len(other))] -= other

            return FrequencyArray(ret,kmin=beg,df=self.df)

        # fall back to simple arrays (may waste memory)
        return np.ndarray.__sub__(self,other)


    def rsub(self,other):
        """
        Restrict the result to the extent of the first array (useful, e.g., for logL over frequency-limited data)
        @param other is ... [required]
        @return
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            if other.kmin >= self.kmin + len(self) or self.kmin >= other.kmin + len(other):
                return self
            else:
                beg = max(self.kmin,other.kmin)
                end = min(self.kmin + len(self),other.kmin + len(other))

                ret = np.array(self,copy=True,dtype=np.find_common_type([self.dtype,other.dtype],[]))
                ret[(beg - self.kmin):(end - self.kmin)] -= other[(beg - other.kmin):(end - other.kmin)]

                return FrequencyArray(ret,kmin=self.kmin,df=self.df)

        return np.ndarray.__sub__(self,other)

    def __iadd__(self,other):
        """
        The inplace add and sub will work only if the second array is contained in the first one
        also there may be problems with upcasting
        @param other is ... [required]
        @return
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            if (self.kmin <= other.kmin) and (self.kmin + len(self) >= other.kmin + len(other)):
                np.ndarray.__iadd__(self[(other.kmin - self.kmin):(other.kmin - self.kmin + len(other))],other[:])
                return self

        # fall back to simple arrays
        np.ndarray.__iadd__(self,other)
        return self

    def __isub__(self,other):
        """
        ...
        @param other is ... [required]
        @return
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            if (self.kmin <= other.kmin) and (self.kmin + len(self) >= other.kmin + len(other)):
                np.ndarray.__isub__(self[(other.kmin - self.kmin):(other.kmin - self.kmin + len(other))],other[:])
                return self

        # fall back to simple arrays
        np.ndarray.__isub__(self,other)
        return self


    def __mul__(self,other):
        """
        In multiplication, we go for the intersection of arrays (not their union!)
        no intersection return a scalar 0
        @param other is ... [required]
        @return
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            beg = max(self.kmin,other.kmin)
            end = min(self.kmin + len(self),other.kmin + len(other))

            if beg >= end:
                return 0.0
            else:
                ret = np.array(self[(beg - self.kmin):(end - self.kmin)],copy=True,dtype=np.find_common_type([self.dtype,other.dtype],[]))
                ret *= other[(beg - other.kmin):(end - other.kmin)]

                return FrequencyArray(ret,kmin=beg,df=self.df)

        # fall back to simple arrays (may waste memory)
        return np.ndarray.__mul__(self,other)


    def __div__(self,other):
        """
        In division, it's OK if second array is larger, but not if it's smaller (which implies division by zero!)
        @param other is ... [required]
        @return
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            if (other.kmin > self.kmin) or (other.kmin + len(other) < self.kmin + len(self)):
                raise ZeroDivisionError
            else:
                ret = np.array(self,copy=True,dtype=np.find_common_type([self.dtype,other.dtype],[]))
                ret /= other[(self.kmin - other.kmin):(self.kmin - other.kmin + len(self))]

            return FrequencyArray(ret,kmin=self.kmin,df=self.df)

        # fall back to simple arrays
        return np.ndarray.__div__(self,other)


    @property
    def f(self):
        """
        Return the reference frequency array
        """
        return np.linspace(self.kmin * self.df,(self.kmin + len(self) - 1) * self.df,len(self))


    @property
    def fmin(self):
        """
        Return the minimum frequency
        """
        return self.kmin * self.df


    @property
    def fmax(self):
        """
        Return the maximal frequency
        """
        return (self.kmin + len(self)) * self.df


    def ifft(self,dt):
        """
        ...
        @param dt is ...
        """
        #n = int(1.0/(dt*self.df))
        n = round(1.0/(dt * self.df)) 
        # by liyn (in case the int() function would cause loss of n)

        ret = np.zeros(int(n/2+1),dtype=self.dtype)
        ret[self.kmin:self.kmin+len(self)] = self[:]
        ret *= n                                        # normalization, ehm, found empirically

        return np.fft.irfft(ret)


    def restrict(self,other):
        """
        Restrict the array to the dimensions of the second, or to dimensions specified as (kmin,len)
        @param Other array
        """
        if isinstance(other,FrequencyArray) and (self.df == other.df):
            kmin, length = other.kmin, len(other)
        elif isinstance(other,(list,tuple)) and len(other) == 2:
            kmin, length = other
        else:
            raise TypeError

        # no need to restrict anything?
        if kmin == self.kmin and length == len(self):
            return other

        ret = FrequencyArray(np.zeros(length,dtype=self.dtype),kmin=kmin,df=self.df)

        beg = max(self.kmin,kmin)
        end = min(self.kmin + len(self),kmin + length)

        ret[(beg - kmin):(end - kmin)] = self[(beg - self.kmin):(end - self.kmin)]

        return ret


    def pad(self,leftpad=1,rightpad=1):
        """
        Pad the array on both sides
        @param leftpad is ...
        @param rightpad is ...
        """
        return self.restrict((self.kmin - int(leftpad)*len(self),int(1+leftpad+rightpad)*len(self)))



# a = FrequencyArray([1,2,3,4,5],kmin=1)
# b = FrequencyArray([1,2,3,4],kmin=2)

# print 2 * a, type(a), (2*a).kmin

def wrapper(*args, **kwargs):
    """Function to convert array and C/C++ class arguments to ptrs

    This function checks the object type. If it is a cupy or numpy array,
    it will determine its pointer by calling the proper attributes. If you design
    a Cython class to be passed through python, it must have a :code:`ptr`
    attribute.

    If you use this function, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    args:
        *args (list): list of the arguments for a function.
        **kwargs (dict): dictionary of keyword arguments to be converted.

    returns:
        Tuple: (targs, tkwargs) where t indicates target (with pointer values
            rather than python objects).

    """
    # declare target containers
    targs = []
    tkwargs = {}

    # args first
    for arg in args:
        # numpy arrays
        if isinstance(arg, np.ndarray):
            targs.append(arg.__array_interface__["data"][0])
            continue

        try:
            # cython classes
            targs.append(arg.ptr)
            continue
        except AttributeError:
            # regular argument
            targs.append(arg)

    # kwargs next
    for key, arg in kwargs.items():
        if isinstance(arg, np.ndarray):
            # numpy arrays
            tkwargs[key] = arg.__array_interface__["data"][0]
            continue

        try:
            # cython classes
            tkwargs[key] = arg.ptr
            continue
        except AttributeError:
            # other arguments
            tkwargs[key] = arg

    return (targs, tkwargs)


def pointer_adjust(func):
    """Decorator function for cupy/numpy agnostic cython

    This decorator applies :func:`few.utils.utility.wrapper` to functions
    via the decorator construction.

    If you use this decorator, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper

if __name__ == '__main__':
    Larm = parameter(value=10, unit="m", toSeconds=True)
    print("Unit of Larm is %s" % Larm.unit)
    print("value of Larm is %e" % Larm.value)
