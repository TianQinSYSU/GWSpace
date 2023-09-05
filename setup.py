#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: setup.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 17:31:48
#==================================

from setuptools import setup, find_packages

with open('README.md') as fp:
    readme = fp.read()

with open('LICENSE') as fp:
    license = fp.read()

setup(
    name='csgwsim',
    version='0.0.1',
    description='Code for Space Gravitational Wave detector Simulation',
    long_description=readme,
    author='ekli, Han Wang, Hong-Yu Chen, Chang-Qing Ye, Xiang-Yu Lyu',
    author_email='lienk@mail.sysu.edu.cn',
    url='https://github.com/....',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

