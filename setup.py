#!/usr/bin/env python
import glob
import os.path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


USE_CYTHON = os.path.exists('.use_cython')


if USE_CYTHON:
    extensions = [
        Extension("*", ["structs/*.pyx"]),
        Extension("*", ["oversegmenters/*.pyx"]),
    ]
    extensions = cythonize(extensions)
else:
    extensions = [
        Extension("oversegmenters.watershed_util", ["oversegmenters/watershed_util.cpp"]),
        Extension("structs.dtypes", ["structs/dtypes.cpp"]),
    ]


setup(
    ext_modules = extensions
)
