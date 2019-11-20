# -*- coding: utf-8 -*-

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
    data : ndarray
        An array of image
        If a 3D array containing multiple images,
        the images must be stacked along the 1st dimension (axis=0).
    mask : ndarray
        An array indicates bad pixel positions
        The shape must be same as image.
        The value of bad pixel is Nonzero, and the others is 0.
        If all pixels are bad, raise ValueError.
    dtype : str or dtype, default 'float32'
        dtype of array used internally
        If the input dtype is different, use a casted copy.
    overwrite_input : bool, default False
        If True, input data is overwrited, but VRAM is saved.

    Returns
    -------
    fixed : ndarray
        An array of images fixed bad pixel

    Notes
    -----
    NaN is ignored in interpolation calculations, but is not fixed.
    '''
    if data.shape[-2:] != mask.shape[-2:]:
        raise ValueError('shape differs between data and mask')
    elif mask.all():
        raise ValueError('No available pixel')

    data = cp.array(data,dtype=dtype,copy=False,ndmin=3)
    mask = cp.array(mask,dtype=dtype,copy=False,ndmin=3)

    y_len, x_len = data.shape[-2:]
    convolution = lambda data,out:conv_kernel(data,x_len,y_len,out)

    if overwrite_input:
        fixed = data
    else:
        fixed = data.copy()

    filt = 1 - mask
    fixed *= filt

    dconv = cp.empty_like(data)
    nconv = cp.empty_like(filt)
    
    while not filt.all():
        convolution(fixed,dconv)
        convolution(filt,nconv)
        fix_kernel(fixed,filt,dconv,nconv,fixed)
        cp.sign(nconv,out=filt)
    
    return cp.squeeze(fixed)