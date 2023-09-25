#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: setup.py
# Author: En-Kun Li, Han Wang
# Mail: lienk@mail.sysu.edu.cn, wanghan657@mail2.sysu.edu.cn
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
code_lib = 'gwspace'


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

# translate the Constants.h to Constants.py
fp_const_h = "./include/Constants.h"
fp_const_py = "./gwspace/Constants.py"

with open(fp_const_h, "r") as fp_in:
    with open(fp_const_py, "w") as fp_out:
        lines = fp_in.readlines()
        for line in lines:
            if (len(line.split())) >= 3:
                if line.split()[0] == "#define":
                    try:
                        _ = float(line.split()[2])
                        string_out = line.split()[1] + " = " + line.split()[2] + "\n"
                        fp_out.write(string_out)

                    except ValueError as e:
                        continue

setup(
    name='gwspace',
    version='0.0.1',
    ext_modules=cythonize(extensions),
    author='En-Kun Li, Han Wang',
    version = '0.0.1',
    packages=find_packages(exclude=('tests', 'docs')),
)
