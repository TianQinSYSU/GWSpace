#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: utils.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 20:52:21
#==================================

import numpy as np

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


def simplesnr(f,h,i=None,years=1.0,noisemodel='SciRDv1',includewd=None):
    """
    TODO To be described
    @param other is the other TDI data
    """
    if i == None:
     h0 = h * np.sqrt(16.0/5.0)    # rms average over inclinations
    else:
     h0 = h * np.sqrt((1 + np.cos(i)**2)**2 + (2.*np.cos(i))**2)

    snr = h0 * np.sqrt(years * 365.25*24*3600) / np.sqrt(lisasens(f,noisemodel,years))
    #print "snr = ", snr, np.sqrt(years * 365.25*24*3600), h0 * np.sqrt(years * 365.25*24*3600) , np.sqrt(lisasens(f,noisemodel,years))
    return snr


