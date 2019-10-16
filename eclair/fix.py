# -*- coding: utf-8 -*-

from itertools  import product

import cupy     as cp

from param  import dtype
from kernel import (
    judge_kernel,
    fix_kernel,
)

#############################
#   fixpix
#############################

def fixpix(data,mask,memsave=False):
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
    memsave : bool, default False
        If True, input data is overwrited, but VRAM is saved.

    Returns
    -------
    fixed : 3-dimension cupy.ndarray (dtype float32)
        An array of images fixed bad pixel
    '''
    
    tmpm  = mask[cp.newaxis,:,:]
    if memsave:
        fixed = data.view()
    else:
        fixed = data.copy()
    while tmpm.sum():
        filt   = 1 - tmpm
        fixed *= filt
        dconv  = convolve(fixed)
        nconv  = convolve(filt)
        zeros  = judge_kernel(nconv)
        fixed += fix_kernel(tmpm, dconv, nconv, zeros)
        tmpm   = zeros

    return fixed

def convolve(data):
    nums, y_len, x_len = data.shape
    conv = cp.zeros([nums,2+y_len,2+x_len],dtype=dtype)
    for x,y in product(range(3),repeat=2):
        conv[:,y:y+y_len,x:x+x_len] += data

    return conv[:,1:-1,1:-1]