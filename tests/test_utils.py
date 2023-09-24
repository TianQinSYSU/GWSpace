#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: test_utils.py
# Author: En-Kun Li
# Mail: lienk@mail.sysu.edu.cn
# Created Time: 2023-09-06 09:49:53
# ==================================

import time
from gwspace.utils import spin_weighted_spherical_harmonic, sYlm


def test_sYlm():
    print("This is a test of SpinWeightedSphericalHarmonic func")
    print("Note that SpinWeightedSphericalHarmonic func can only take s=-2")

    theta = 0.3
    phi = 0.1

    s = -2

    for l in range(2, 6):
        for m in range(-l, l+1):
            st1 = time.time()
            y1 = spin_weighted_spherical_harmonic(s, l, m, theta, phi)
            ed1 = time.time()

            st2 = time.time()
            y2 = sYlm(s, l, m, theta, phi)
            ed2 = time.time()

            print(r"[%2d]sYlm[(%2d,%2d)] = %s <---> %s" % (s, l, m, y1, y2))
            print("-"*50)
            print(r"Time cost for 1 and 2 is %s & %s" % (ed1-st1, ed2-st2))
            print("="*50)


if __name__ == "__main__":
    test_sYlm()
