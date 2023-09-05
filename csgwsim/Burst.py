#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: Burst.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-31 16:36:40
# ==================================

import numpy as np
# from Constants import *


class BurstWaveform(object):
    """
    A sin-Gaussian waveforms for pooly modelled burst source
    --------------------------------------------------------
    """

    def __init__(self, amp, tau, fc, tc=0):
        self.amp = amp
        self.tau = tau
        self.fc = fc
        self.tc = tc

    def __call__(self, tf):
        t = tf-self.tc
        h = (2/np.pi)**0.25 * self.tau**(-0.5) * self.amp
        h *= np.exp(- (t/self.tau)**2)*np.exp(2j*np.pi*self.fc*t)
        return h.real, h.imag


