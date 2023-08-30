#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: setup.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-25 11:27:36
#==================================

from __future__ import print_function

import sys, os, subprocess, shutil

from distutils.errors import DistutilsError
from distutils.command.clean import clean as _clean

from setuptools.command.install import install as _install
from setuptools import Extension, setup, Command
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools import find_packages

from Cython.Build import cythonize

import numpy

# Classifiers for distribution of package
classifiers = [
    'Programming Language :: Python',
    'Programming Language :: Cython',
    'Programming Language :: Python :: %s'%(sys.version_info.major),
    'Programming Language :: Python :: %s'%(sys.version.split()[0]),
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Astronomy',
    'Topic :: Scientific/Engineering :: Physics',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
]

requires = []
setup_requires = ['numpy',]
install_requires = setup_requires + [
                      'cython',
                      'scipy',
                      'matplotlib',
                      'h5py',
                      'astropy'
                      ]

# do the actual work of building the package
#VERSION = get_version_info()

argv_replace = []
constants = ''
gsl_prefix = '/usr'
force_recompile=False
for arg in sys.argv:
    if arg.startswith('--with-gsl='):
        gsl_prefix = arg.split('=', 1)[1]
    elif arg=='--force':
        force_recompile = True
    else:
        argv_replace.append(arg)
sys.argv = argv_replace

# Extensions
lib_gsl_dir = gsl_prefix+"/lib"
include_gsl_dir = gsl_prefix+"/include"
# Note: we use only one list of include dirs with everything needed in it
#extra_compile_args = [Dconstants]
extra_link_args= []

extra_compile_args = ['-std=c99']
all_include_dirs = [numpy.get_include(), include_gsl_dir, "./",
                    "./include/"]
all_libraries = ["gsl", "gslcblas"]
all_library_dirs = [lib_gsl_dir]
def func_extension(ext, sources=[]):
    return Extension(ext,
              sources=sources,
              include_dirs=all_include_dirs,
              language="c",
              extra_compile_args=extra_compile_args,
              #extra_link_args=extra_link_args,
              libraries=all_libraries,
              library_dirs=all_library_dirs)

INC   = "./include/"
SRC   = "./src/"
PYSRC = "./python/"

extensions = [
    func_extension("pyconstants",
      sources=[PYSRC+"pyconstants.pyx"]),
    func_extension("pystruct",
      sources=[PYSRC+"pystruct.pyx",
      SRC+"struct.c"]),
    func_extension("pyIMRPhenomD",
      sources=[PYSRC+"pyIMRPhenomD.pyx",
      SRC+"IMRPhenomD.c",
      SRC+"IMRPhenomD_internals.c",
      SRC+"IMRPhenomUtils.c",
      SRC+"IMRPhenomInternalUtils.c",
      SRC+"struct.c"]),
    func_extension("pyIMRPhenomHM",
      sources=[PYSRC+"pyIMRPhenomHM.pyx",
      SRC+"IMRPhenomHM.c",
      SRC+"IMRPhenomD_internals.c",
      SRC+"IMRPhenomUtils.c",
      SRC+"IMRPhenomInternalUtils.c",
      SRC+"RingdownCW.c",
      SRC+"struct.c"]),
    #func_extension("pyEOBNRv2HMROM",
    #  sources=[PYSRC+"pyEOBNRv2HMROM.pyx",
    #  "EOBNRv2HMROM.c",
    #  "EOBNRv2HMROMstruct.c",
    #  "struct.c"])
]
ext_modules = cythonize(extensions,force=force_recompile)


setup(
    name = 'bhb',
    version = '0.1.0',
    description = 'Library for different type BHB waveform.',
    #long_description = open('descr.rst').read(),
    author = 'ekli and ... ...',
    author_email = 'lienk@mail.sysu.edu.cn',
    keywords = ['gravitational waves'],
    setup_requires = setup_requires,
    install_requires = install_requires,
    packages = find_packages(),
    ext_modules = ext_modules,
    classifiers = classifiers,
    #py_modules = ['bbh'],
)


