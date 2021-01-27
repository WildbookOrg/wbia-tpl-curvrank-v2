# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension('stitch', sources=['stitch.pyx'], libraries=['m'])  # Unix-like specific
]

setup(name='stitch', ext_modules=cythonize(ext_modules))
