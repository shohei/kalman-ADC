#!/usr/bin/env python
# -*- coding: utf-8 -*-
#filename_setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx
import numpy

setup(
    name = 'resample',
    ext_modules = [Extension('resample', ['resample.pyx'])],
    cmdclass = { 'build_ext': build_pyx },
    include_dirs = [numpy.get_include()],
)
