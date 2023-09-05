#!/usr/bin/bash
#-*- coding: utf-8 -*-  
#==================================
# File Name: install_build.sh
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-02-08 10:08:10
#==================================

rm -rf cmake-build-debug
mkdir cmake-build-debug
cd cmake-build-debug
cmake ..
make
