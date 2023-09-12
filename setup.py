#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: setup.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 19:43:34
#==================================

import numpy
from setuptools import find_namespace_packages, find_packages
from Cython.Build import cythonize
from distutils.core import setup, Extension
import sys

argv_replace = []

gsl_prefix = '/usr'

for arg in sys.argv:
    if arg.startswith('--with-gsl='):
        gsl_prefix = arg.split('=', 1)[1]
    else:
        argv_replace.append(arg)

sys.argv = argv_replace

lib_gsl_dir = gsl_prefix+"/lib"
include_gsl_dir = gsl_prefix+"/include"

# Update these paths accordingly

# extensions
gcbpath = "./"
fastgb_ext = Extension("csgwsim._FastGB",
        sources = [
            gcbpath+"/src/_FastGB.pyx",
            gcbpath+"/src/spacecrafts.c",
            gcbpath+"/src/GB.c",
            ],
        include_dirs = [gcbpath+"/include", include_gsl_dir, numpy.get_include()],
        extra_compile_args = ["-std=c99", "-O3"],
        libraries=['gsl', 'gslcblas', 'm'],
        library_dirs=[lib_gsl_dir],
        )

csgwpath = "csgwsim"

# add all extensions
extensions = []
extensions.append(fastgb_ext)


setup(
        name = "fastgb",
        version = '0.0.1',
        ext_modules = cythonize(extensions),
        author = '',
        packages = find_packages(exclude=('tests', 'docs')),
  )

