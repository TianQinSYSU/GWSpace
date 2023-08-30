#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: bbh.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-25 21:47:25
#==================================

import numpy as np
import copy

import pystruct as pystruct
import pyconstants as pyconstants
import pyIMRPhenomD as pyIMRPhenomD
import pyIMRPhenomHM as pyIMRPhenomHM

list_approximants_bbh = ['IMRPhenomD', 'IMRPhenomHM']

if __name__ == '__main__':
    print("This is some Phenom waveform")
