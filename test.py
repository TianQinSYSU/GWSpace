#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-25 10:54:38
#==================================

import sys
from utils import yaml_readinpars

from Waveforms.bbh.bbh import *



argv = sys.argv[1:]

print(f"The input file is {argv}")

pars = yaml_readinpars(argv[0])

print(f"GW source type is {pars['type']}")
print(f"{pars['iota'] * 3/3.0}")

print(f"type of pars['m1'] is {type(pars['m1'])}")
print(type(pars['m1']))
