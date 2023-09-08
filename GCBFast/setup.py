#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: setup.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 19:43:34
#==================================

import numpy
from Cython.Build import cythonize
from distutils.core import setup, Extension
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib
import sys

argv_replace = []
for arg in sys.argv:
    if arg.startswith('--with-gsl='):
        gsl_prefix = arg.split('=', 1)[1]
    else:
        argv_replace.append(arg)
sys.argv = argv_replace

lib_gsl_dir = gsl_prefix+"/lib"
include_gsl_dir = gsl_prefix+"/include"

# Update these paths accordingly
c_include_dirs = ['include', include_gsl_dir, numpy.get_include()]
c_sources = ['src/spacecrafts.c', 'src/GB.c']

extensions = [
    Extension("fastgb._FastGB", ["src/_FastGB.pyx"] + c_sources,
              include_dirs=c_include_dirs,
              extra_compile_args = ["-std=c99", "-O3"],
              libraries=['gsl', 'gslcblas', 'm'],
              library_dirs=[lib_gsl_dir],
              )
]

setup(
        name = "fastgb",
        version = '0.0.1',
        ext_modules = cythonize(extensions),
        author = '',
        #py_modules = ['fastgb'],
        packages = ['fastgb'],
)

