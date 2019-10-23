# -*- coding: utf-8 -*-

from itertools  import product

import cupy     as cp

from param  import dtype
from kernel import (
    judge_kernel,
    fix_kernel,
    conv_kernel,
)

#############################
#   fixpix
#############################

def fixpix(data,mask,dtype=dtype,memsave=False):
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
    memsave : bool, default False
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

    tmpm = mask[cp.newaxis,:,:]
    if memsave:
        fixed = data.view()
    else:
        fixed = data.copy()
    while tmpm.sum():
        filt  = 1 - tmpm
        dconv = convolve(fixed,filt,dtype=dtype)
        nconv = convolve(filt,1.0,dtype=dtype)
        zeros = judge_kernel(nconv)
        fix_kernel(fixed, tmpm, dconv, nconv, zeros, fixed)
        tmpm  = zeros

    return fixed

def convolve(data,filt,dtype=dtype):
    nums, y_len0, x_len0 = data.shape
    x_len1 = x_len0 + 2
    y_len1 = y_len0 + 2
    xy_len = x_len0 * y_len0

    conv = cp.zeros([nums,y_len1,x_len1],dtype=dtype)
    
    conv_kernel(data,filt,x_len0,y_len0,x_len1,y_len1,xy_len,conv)
    
    return conv[:,1:-1,1:-1]