# -*- coding: utf-8 -*-
'''
Eclair
======

Eclair: CUDA-based Library for Astronomical Image Reduction

This module provides some useful classes and functions
in astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.

This module requires
    1. NVIDIA GPU
    2. CUDA
    3. NumPy, Astropy and CuPy
'''

__all__ = ['reduction', 'FitsContainer', 'ImAlign',
    'imalign', 'imcombine', 'fixpix']

from cupy import asarray

from param import __version__, dtype

from kernel import reduction_kernel

from fitscontainer import FitsContainer

from align import ImAlign, imalign

from combine import imcombine

from fix import fixpix

def reduction(image,bias,dark,flat,out=None,dtype=dtype):
    '''
    This function is equal to the equation:
    result = (image - bias - dark) / flat, but needs less memory.
    Therefore, each inputs must be broadcastable shape.

    Parameters
    ----------
    image : cupy.ndarray
    bias : cupy.ndarray
    dark : cupy.ndarray
    flat : cupy.ndarray
    out : cupy.ndarray, default None
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape and dtype as the
        expected output.
    dtype : str or dtype (NumPy or CuPy), default 'float32'
        dtype of array used internally
        If the input dtype is different, use a casted copy.
    
    Returns
    -------
    result : cupy.ndarray
    '''
    image = asarray(image,dtype=dtype)
    bias  = asarray(bias,dtype=dtype)
    dark  = asarray(dark,dtype=dtype)
    flat  = asarray(flat,dtype=dtype)

    if out is None:
        result = reduction_kernel(image,bias,dark,flat)
    else:
        result = reduction_kernel(image,bias,dark,flat,out)

    return result