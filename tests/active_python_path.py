#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: active_python_path.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 10:14:13
#==================================

import os, sys

try:
    import csgwsim
except ImportError:
    csgwsimpath = '../build/lib'
    abs_csgwsimpath = os.path.abspath(csgwsimpath)

    if abs_csgwsimpath not in sys.path:
        sys.path.append(abs_csgwsimpath)

    import csgwsim
