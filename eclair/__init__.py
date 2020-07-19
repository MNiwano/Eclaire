# -*- coding: utf-8 -*-
'''
Eclair
======

Eclair: CUDA-based Library for Astronomical Image Reduction

This package provides some useful classes and functions
in astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.

This module requires
    1. NVIDIA GPU
    2. CUDA
    3. NumPy, Astropy and CuPy
'''

__all__ = [
    '__version__', 'set_dtype', 'reduction', 'FitsContainer',
    'imalign', 'imcombine', 'fixpix'
]

from .common import __version__

from .util import set_dtype, reduction

from .io import FitsContainer

from .align import imalign

from .stats import sigma_clipped_stats, imcombine

from .fix import fixpix