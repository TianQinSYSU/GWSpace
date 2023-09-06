#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_pyINIDetectors.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:54:36
#==================================

from active_python_path import csgwsim as gws


if __name__ == '__main__':
    print("This is initial pars for detectors")
    tq = gws.INITianQin()
    print(f"Armlength for TianQin is: {tq.armLength}")
