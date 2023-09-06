#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: setup.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:31:48
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
    'Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Astronomy',
    'Topic :: Scientific/Engineering :: Physics',
]

# Install requires
requires = []
setup_requires = ['numpy',]
install_requires = setup_requires + [
                      'cython',
                      'scipy',
                      'matplotlib',
                      'h5py',
                      'astropy',
                      'pyyaml',
                      'setuptools',
                      'numba',
                      ]

# Readin args

argv_replace = []
constants = ''
gsl_prefix = '/usr'
compile_bhb = False
compile_few = False
compile_eccFD = False
force_recompile=False
for arg in sys.argv:
    if arg.startswith('--with-gsl='):
        gsl_prefix = arg.split('=', 1)[1]
    elif arg.startswith('--with-lapack='):
        lapack_prefix = arg.split('=', 1)[1]
    elif arg.startswith('--compile-bhb'):
        compile_bhb = True
    elif arg.startswith('--compile-FEW'):
        compile_few = True
    elif arg.startswith('--compile-eccFD'):
        compile_eccFD = True
    elif arg=='--force':
        force_recompile = True
    else:
        argv_replace.append(arg)
sys.argv = argv_replace

# Extensions
lib_gsl_dir = gsl_prefix+"/lib"
include_gsl_dir = gsl_prefix+"/include"
lib_lapack_dir = lapack_prefix+"/lib"
include_lapack_dir = lapack_prefix+"/include"


with open('README.md') as fp:
    readme = fp.read()

with open('LICENSE') as fp:
    license = fp.read()

setup(
    name='csgwsim',
    version='0.0.1',
    description='Code for Space Gravitational Wave detector Simulation',
    long_description=readme,
    author='ekli, Han Wang, Hong-Yu Chen, Chang-Qing Ye, Xiang-Yu Lyu and ...',
    author_email='lienk@mail.sysu.edu.cn',
    keywords = ['TianQin', 'LISA', 'gravitational waves'],
    url='https://github.com/....',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

