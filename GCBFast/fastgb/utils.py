#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: utils.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 20:52:21
#==================================

import numpy as np
import sys, time

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
        n = int(1.0/(dt*self.df))

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
