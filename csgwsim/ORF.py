#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: ORF.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-15 14:52:04
#==================================

import numpy as np
from utils import get_uvk


def transfer_single_arm(f, kn, kpr, kps, LT):
    return (0.5 * np.sinc(np.pi * f *LT*(1-kn)) * 
            np.exp(-1j * np.pi * f* (LT + kpr + kps)) )

class OverlapReductionFunc(object):
    '''
    Calculate ORF
    '''

