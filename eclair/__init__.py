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

import cupy as cp

import common
from common import __version__

from kernel import reduction_kernel

from io import FitsContainer

from align import imalign

from stats import imcombine

from fix import fixpix

def set_dtype(dtype):
    '''
    Change the default dtype used by all functions
    and classes in this package.

    Parameters
    ----------
    dtype : str or dtype
    '''
    common.dtype = dtype

def reduction(image,bias,dark,flat,out=None,dtype=None):
    '''
    This function is equal to the equation:
    out = (image - bias - dark) / flat, but needs less memory.
    Therefore, each inputs must be broadcastable shape.

    Parameters
    ----------
    image : ndarray
    bias : ndarray
    dark : ndarray
    flat : ndarray
    out : cupy.ndarray, default None
        Alternate output array in which to place the result. The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    dtype : str or dtype, default 'float32'
        dtype of array used internally
        If None, this value will be usually "float32", 
        but this can be changed with eclair.set_dtype.
        If the input dtype is different, use a casted copy.
    
    Returns
    -------
    out : cupy.ndarray
    '''
    dtype = common.judge_dtype(dtype)

    image = cp.asarray(image,dtype=dtype)
    bias  = cp.asarray(bias,dtype=dtype)
    dark  = cp.asarray(dark,dtype=dtype)
    flat  = cp.asarray(flat,dtype=dtype)

    if out is None:
        out = cp.empty(
            cp.broadcast(image,bias,dark,flat).shape,
            dtype=dtype
        )
    
    reduction_kernel(image,bias,dark,flat,out)

    return out
