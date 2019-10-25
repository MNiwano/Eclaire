# -*- coding: utf-8 -*-

from itertools  import product

import cupy     as cp

from param  import dtype
from kernel import (
    fix_kernel,
    conv_kernel,
)

#############################
#   fixpix
#############################

def fixpix(data,mask,dtype=dtype,overwrite_input=False):
    '''
    fill the bad pixel with mean of surrounding pixels

    Parameters
    ----------
    data : 3-dimension cupy.ndarray
        An array of images stacked along the 1st axis
    mask : 2-dimension cupy.ndarray
        An array indicates bad pixel positions
        The shape of mask must be same as image.
        The value of bad pixel is 1, and the others is 0.
    dtype : str or dtype, default 'float32'
        dtype of array used internally
        If the input dtype is different, use a casted copy.
    overwrite_input : bool, default False
        If True, input data is overwrited, but VRAM is saved.

    Returns
    -------
    fixed : 3-dimension cupy.ndarray
        An array of images fixed bad pixel
    '''
    y_len1, x_len1 = data.shape[1:]
    y_len2, x_len2 = mask.shape
    if not(y_len1==y_len2 and x_len1==x_len2):
        raise ValueError('shape differs between data and mask')

    data = cp.asarray(data,dtype=dtype)
    mask = cp.asarray(mask,dtype=dtype)

    filt = 1 - mask[cp.newaxis,:,:]
    if overwrite_input:
        fixed = data.view()
    else:
        fixed = data.copy()
    cp.multiply(fixed,filt,out=fixed)
    while not filt.all():
        dconv = convolve(fixed,dtype=dtype)
        nconv = convolve(filt,dtype=dtype)
        fix_kernel(fixed, filt, dconv, nconv, fixed)
        cp.sign(nconv,out=filt)

    return fixed

def convolve(data,dtype=dtype):
    y_len, x_len = data.shape[1:]
    xy_len = x_len * y_len

    conv = cp.empty_like(data)
    
    conv_kernel(data,x_len,y_len,xy_len,conv)
    
    return conv