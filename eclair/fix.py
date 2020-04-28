# -*- coding: utf-8 -*-

import cupy     as cp

from common import judge_dtype
from kernel import (
    elementwise_not,
    checkfinite,
    fix_core,
    conv_kernel,
)

#############################
#   fixpix
#############################

def fixpix(data,mask,out=None,dtype=None,fix_NaN=False):
    '''
    fill the bad pixel with mean of surrounding pixels

    Parameters
    ----------
    data : ndarray
        An array of image
        If a 3D array containing multiple images,
        the images must be stacked along the 1st dimension (axis=0).
    mask : ndarray
        An array indicates bad pixel positions
        The shape must be same as image.
        The value of bad pixel is nonzero, and the others is 0.
        If all pixels are bad, raise ValueError.
    out : cupy.ndarray, default None
        Alternate output array in which to place the result. The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    dtype : str or dtype, default 'float32'
        dtype of array used internally
        If None, this value will be usually "float32", 
        but this can be changed with eclair.set_dtype.
        If the input dtype is different, use a casted copy.
    fix_NaN : bool, default False
        If true, fix NaN pixel even if it's not bad pixel in the mask.

    Returns
    -------
    fixed : ndarray
        An array of images fixed bad pixel

    Notes
    -----
    NaN is ignored in interpolation calculations,
    but is not fixed if fix_NaN is False.
    '''
    dtype = judge_dtype(dtype)
    if data.shape[-2:] != mask.shape[-2:]:
        raise ValueError('shape differs between data and mask')
    elif mask.all():
        raise ValueError('No available pixel')

    data = cp.array(data,dtype=dtype,copy=False,ndmin=3)
    mask = cp.array(mask,dtype=dtype,copy=False,ndmin=3)

    convolution = lambda data,out:conv_kernel(data,out)

    if out is None:
        out = cp.empty_like(data)
    cp.copyto(out,data)

    filt = elementwise_not(mask)
    if fix_NaN:
        filt = checkfinite(data,filt)

    out *= filt

    dconv = cp.empty_like(out)
    nconv = cp.empty_like(filt)
    
    while not filt.all():
        convolution(out,dconv)
        convolution(filt,nconv)
        fix_core(out,filt,dconv,nconv,out)
        cp.sign(nconv,out=filt)
    
    return cp.squeeze(out)