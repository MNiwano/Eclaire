# -*- coding: utf-8 -*-

import cupy as cp

from .util import (
    judge_dtype,
    elementwise_not,
    ternary_operation,
    checkfinite,
)

#############################
#   fixpix
#############################

def fixpix(data,mask,out=None,dtype=None,fix_NaN=False):
    '''
    fill the bad pixel with linear interpolation using surrounding pixels

    Parameters
    ----------
    data : array-like
        An array of image
        If a 3D array containing multiple images,
        the images must be stacked along the 1st dimension (axis=0).
    mask : array-like
        An array indicates bad pixel positions
        The shape must be same as image.
        The value of bad pixel is nonzero, and the others is 0.
        If all pixels are bad, raise ValueError.
    out : cupy.ndarray, default None
        Alternate output array in which to place the result. The default
        is None; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    dtype : str or dtype, default None
        dtype of ndarray used internally
        If None, use eclair.common.default_dtype.
        If the input dtype is different, use a casted copy.
    fix_NaN : bool, default False
        If true, fix NaN pixel even if it's not specified
        as bad pixel in mask

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

    data = cp.asarray(data,dtype=dtype)
    mask = cp.asarray(mask,dtype=dtype)

    dshape = data.shape
    imshape = data.shape[-2:]
    if imshape != mask.shape[-2:]:
        raise ValueError('shape differs between data and mask')
    if mask.all(axis=(-2,-1)).any():
        raise ValueError('No available pixel')

    if out is None:
        out = data.copy()
    else:
        cp.copyto(out,data)
    
    tout = cp.array(out,copy=False,ndmin=3)
    filt = cp.array(mask,dtype='uint8')
    cp.logical_not(filt,out=filt)

    ternary_operation(filt,tout,cp.nan,tout)

    dconv = cp.empty_like(tout)
    nconv = cp.empty_like(filt)

    for _ in range(max(imshape)):
        count_up(filt,nconv)
        conv_kernel(tout,nconv,dconv)
        fix_core(filt,dconv,fix_NaN,tout)
        if nconv.all():
            break
        else:
            cp.sign(nconv,out=filt)

    return out

count_up = cp.ElementwiseKernel(
    in_params='raw I input',
    out_params='I output',
    operation='''
    int ly = input.shape()[0];
    int lx = input.shape()[1];
    int i_y = i / lx;
    int i_x = i % lx;
    int idx[2];
    int *t_y = idx, *t_x = idx+1;
    int s_x = max(i_x-1,0), e_x = min(i_x+2,lx);
    int s_y = max(i_y-1,0), e_y = min(i_y+2,ly);
    I tmp = 0;
    for (*t_y=s_y; *t_y<e_y; (*t_y)++) {
        for (*t_x=s_x; *t_x<e_x; (*t_x)++) {
            tmp += input[idx];
        }
    }
    output = tmp;
    ''',
    name='count_up'
)

conv_kernel = cp.ElementwiseKernel(
    in_params='raw T input, I n',
    out_params='T output',
    operation='''
        int ly = input.shape()[1];
        int lx = input.shape()[2];
        int iyz = i / lx;
        int i_x = i % lx;
        int i_y = iyz % ly;
        int i_z = iyz / ly;
        int idx[3] = {i_z};
        int *t_y = idx+1, *t_x = idx+2;
        int s_x = max(i_x-1,0), e_x = min(i_x+2,lx);
        int s_y = max(i_y-1,0), e_y = min(i_y+2,ly);
        int fn, c = 0;
        T val, tmp, summed = 0;
        for (*t_y=s_y; *t_y<e_y; (*t_y)++) {
            tmp = 0;
            for (*t_x=s_x; *t_x<e_x; (*t_x)++) {
                val = input[idx];
                c += fn = isfinite(val);
                tmp += (fn? val : 0);
            }
            summed += tmp;
        }
        T corr = (e_x - s_x)*(e_y - s_y) / c;
        output = corr*summed / n;
    ''',
    name='convolution'
)

fix_core = cp.ElementwiseKernel(
    in_params='I f, T d, bool fn',
    out_params='T z',
    operation='''
        char flag = f & (!fn | isfinite(z));
        z = (flag? z : d);
    ''',
    name='fix_core'
)