#!/bin/bash
#-*- coding: utf-8 -*-  
#==================================
# File Name: install_local.sh
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-08-26 09:44:07
#==================================

echo "====================================="
echo "Compile the bbh waveforms"
cd Waveforms/bbh
rm -r build
rm *.so
python setup.py build
cp build/lib.*/*.so ./

echo "Done the compile of bbh waveform"
cd ../../

echo "====================================="
echo "Compile the FastEMRIWaveforms"
cd Waveforms/FastEMRIWaveforms
rm -r build
rm *.so

python setup.py build --gsl /opt/gsl-2.7.1 --lapack /opt/lapack-3.11.0
cp build/lib.*/*.so ./

echo "Done the compile of the bbh waveform"
cd ../../
