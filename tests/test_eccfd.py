#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_eccfd.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-14 19:49:41
#==================================

import os, sys
sys.path.append("/home/ekli/Documents/csgwd_BackUP/EccentricFD/build/lib.linux-x86_64-3.9")

from eccentric_fd.eccentric_fd import *

if __name__ == '__main__':
    from time import time, strftime
    para = {'delta_f': 0.0001,
            'f_final': 1,
            'f_lower': 0.01,
            'mass1': 10 * MSUN_SI,
            'mass2': 10 * MSUN_SI,
            'inclination': 0.23,
            'eccentricity': 0.4,
            'long_asc_nodes': 0.23,
            'coa_phase': 0,
            'distance': 100 * MPC_SI,
            'obs_time': 365*24*3600}
    start_time = time()
    print(strftime("%Y-%m-%d %H:%M:%S"))
    h_ap = gen_ecc_fd_amp_phase(**para)
    h_ap_h = gen_ecc_fd_and_phase(**para)
    hp, hc = gen_ecc_fd_waveform(**para)
    print(strftime("%Y-%m-%d %H:%M:%S"), f'Finished in {time() - start_time: .5f}s', '\n')
