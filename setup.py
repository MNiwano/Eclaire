#! /usr/bin/env python
#-*- encoding:utf-8 -*-

from setuptools import setup
from setuptools import find_packages

desc='''
This package provides some useful classes and functions
in astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.
'''

requires=['numpy','astropy']
try:
    import cupy
except:
    requires.append('cupy')

with open('eclair/common.py') as f:
    exec(f.read())
    assert __version__

setup(
    name='eclair',
    packages=['eclair'],
    version=__version__,
    description='Eclair: CUDA-based Library for Astronomical Image Reduction',
    long_description=desc,

    author='Masafumi Niwano',
    author_email='niwano@hp.phys.titech.ac.jp',
    url='https://github.com/MNiwano/Eclair',

    install_requires=requires,
    keywords = ['astronomy', 'science', 'fits', 'GPU', 'CUDA'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU'
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering'
    ]
)