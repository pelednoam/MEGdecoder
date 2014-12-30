'''
Created on Dec 12, 2013

@author: noampeled
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("distanceUtils", ["distanceUtils.pyx"])]
)
