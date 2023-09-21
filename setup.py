#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: setup.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-06 19:43:34
# ==================================

import numpy as np
from setuptools import find_packages
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
code_lib = 'csgwsim'


def func_ext(name, src):
    return Extension(
        code_lib+"."+name,
        sources=src,
        include_dirs=["include",
                      include_gsl_dir, np.get_include()],
        extra_compile_args=["-std=c99", "-O3"],
        libraries=['gsl', 'gslcblas', 'm'],
        library_dirs=[lib_gsl_dir],
    )


fastgb_ext = func_ext("libFastGB",
                      src=[
                          "src/_FastGB.pyx",
                          "src/spacecrafts.c",
                          "src/GB.c",
                      ])

eccfd_ext = func_ext('libEccFD',  # name of the lib
                     src=['src/InspiralEccentricFD.c',
                          'src/InspiralEccentricFDBasic.c',
                          'src/InspiralOptimizedCoefficientsEccentricityFD.c',
                          ])

imrphd_ext = func_ext('pyIMRPhenomD',
                      src=[
                          'src/pyIMRPhenomD.pyx',
                          'src/IMRPhenomD.c',
                          'src/IMRPhenomD_internals.c',
                      ])

# add all extensions
extensions = []
extensions.append(fastgb_ext)
extensions.append(eccfd_ext)
extensions.append(imrphd_ext)

setup(
    name='csgwsim',
    version='0.0.1',
    ext_modules=cythonize(extensions),
    author='',
    packages=find_packages(exclude=('tests', 'docs')),
)
