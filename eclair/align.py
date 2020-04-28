# -*- coding: utf-8 -*-

import sys
from itertools  import product

if sys.version_info.major == 2:
    from future_builtins import zip, map

import numpy as np
import cupy  as cp

from common import null, judge_dtype

from kernel import (
    neighbor_core,
    linear_core,
    poly_core,
    solve_tridiag,
    spline_core,
)

#############################
#   imalign
#############################

class Align:

    interp_choices = ['spline3','poly3','linear','neighbor']
    
    def __init__(self,x_len,y_len,interp='spline3',dtype=None):
        self.x_len = x_len
        self.y_len = y_len
        self.dtype = judge_dtype(dtype)
        if   interp == 'spline3':
            self.vecx = mkvec(x_len,self.dtype)
            self.vecy = mkvec(y_len,self.dtype)
            self.shift = self.__spline
        elif interp == 'poly3':
            self.__polyinit()
            self.shift = self.__poly
        elif interp == 'linear':
            self.shift = self.__linear
        elif interp == 'neighbor':
            self.shift = self.__neighbor
        else:
            raise ValueError('"{}" is not inpremented'.format(interp))
        
    def __call__(self,data,shifts,progress=null,args=()):

        data   = cp.asarray(data,dtype=self.dtype)
        shifts = np.asarray(cp.asnumpy(shifts),dtype=self.dtype)

        nums, y_len, x_len = data.shape
        if (y_len,x_len) != (self.y_len,self.x_len):
            message = 'shape of images is differ from {}'
            raise ValueError(message.format((self.y_len,self.x_len)))
        elif nums != len(shifts):
            raise ValueError('data and shifts do not match')

        xy_i = np.floor(shifts).astype(int)
        xy_d = shifts - xy_i

        xy_i   -= xy_i.min(axis=0)
        x_u,y_u = xy_i.max(axis=0)

        aligned = cp.empty([nums,y_len-y_u,x_len-x_u],dtype=self.dtype)
        for i,((ix,iy),dxy,frame) in enumerate(zip(xy_i,xy_d,data)):
            shifted = self.shift(frame,*dxy)
            aligned[i] = shifted[y_u-iy:y_len-iy, x_u-ix:x_len-ix]
            progress(i,*args)

        return aligned

    def __neighbor(self,data,dx,dy):
        shifted = cp.empty_like(data)
        neighbor_core(data,dx,dy,shifted)
        return shifted

    def __linear(self,data,dx,dy):
        shifted = self.__neighbor(data,dx,dy)
        linear_core(data,dx,dy,shifted[1:,1:])
        return shifted
    
    def __poly(self,data,dx,dy):
        shifted = self.__linear(data,dx,dy)

        ex = 1-dx
        ey = 1-dy
        shift_vector = cp.array(
            [ex**j * ey**i for i,j in product(range(4),repeat=2)],
            dtype=self.dtype
        )
        shift_vector.dot(self.mat,out=shift_vector)
        shift_mat = shift_vector.reshape(4,4)

        poly_core(data,shift_mat,shifted[2:-1,2:-1])
        
        return shifted

    def __spline(self,data,dx,dy):
        shifted = self.__neighbor(data,dx,dy)
        tmpd = cp.empty([self.x_len-1,self.y_len],dtype=self.dtype)
        spline1d(data.T,dx,self.vecx,tmpd)
        spline1d(tmpd.T,dy,self.vecy,shifted[1:,1:])
        return shifted

    def __polyinit(self):
        mat = np.empty([16,16],dtype=self.dtype)
        for y,x,k,l in product(range(4),repeat=4):
            mat[y*4+x,k*4+l] = (x-1)**l * (y-1)**k
        self.mat = cp.array(np.linalg.inv(mat),dtype=self.dtype)

def imalign(data,shifts,interp='spline3',dtype=None):
    '''
    Stack the images with aligning their relative positions,
    and cut out the overstretched area
    Parameters
    ----------
    data : 3D ndarray
        An array of images stacked along the 1st dimesion (axis=0)
    shifts : 2D ndarray
        An array of relative positions of images in units of pixel
        Along the 1st axis, values of each images must be the same order
        as the 1st axis of "data".
        Along the 2nd axis, the 1st item is interpreted as 
        the value of X, the 2nd item as the value of Y.
    interp : {'spline3', 'poly3', 'linear', 'neighbor'}, default 'spline3'
        Subpixel interpolation algorithm in subpixel image shift
            spline3  - bicubic spline
            poly3    - 3rd order interior polynomial
            linear   - bilinear
            neighbor - nearest neighbor
    dtype : str or dtype, default None
        dtype of array used internally
        If None, this value will be usually "float32", 
        but this can be changed with eclair.set_dtype.
        If the dtype of input array is different, use a casted copy.
    
    Returns
    -------
    aligned : 3D cupy.ndarray
        An array of images aligned and stacked along the 1st axis
    '''
    y_len, x_len = data.shape[-2:]
    func = Align(x_len=x_len,y_len=y_len,interp=interp,dtype=dtype)

    return func(data,shifts)

def mkvec(n,dtype):
    asarray = lambda x:cp.asarray(x,dtype=dtype)

    vec1 = cp.full(n-2,4,dtype=dtype)
    vec2 = cp.ones(n-3,dtype=dtype)

    for i in range(n-3):
        vec2[i] /= vec1[i]
        vec1[i+1] -= vec2[i]

    return asarray(vec1), asarray(vec2)

def spline1d(data,d,vec,out):
    v1, v2 = vec
    u = cp.zeros_like(data)
    u[1:-1] = (data[2:]-data[1:-1])-(data[1:-1]-data[:-2])
    solve_tridiag(v1,v2,u[1:-1],size=u.shape[1])
    spline_core(u,data,1-d,out)