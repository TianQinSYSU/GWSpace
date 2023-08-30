#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: EMRI.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-28 13:43:30
#==================================

import numpy as np
import numpy as np
from Constants import *
from Waveforms.PyIMRPhenomD import IMRPhenomD as pyIMRD
from Waveforms.PyIMRPhenomD import IMRPhenomD_const as pyimrc

import sys, os
try:
    import bbh
except:
    bbh_path = './Waveforms/bbh'
    abs_bbh_path = os.path.abspath(bbh_path)
    
    if abs_bbh_path not in sys.path:
        sys.path.append(abs_bbh_path)

import bbh



