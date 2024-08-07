# -*- coding: utf-8 -*-

import sys
from itertools  import product

if sys.version_info.major == 2:
    from future_builtins import zip, map

import numpy as np
import cupy  as cp

from .util import judge_dtype
#
#############################
#   imalign
#############################

class Shift:
    
    def __init__(self,x_len,y_len,interp='spline3',boundary='neighbor',
        constant=0,dtype=None):
        self.x_len = x_len
        self.y_len = y_len
        self.dtype = judge_dtype(dtype)
        self.const = constant

        errmsg = '"{}" is not inpremented'

        if   interp == 'spline3':
            self.vec = (
                mkvec(x_len,self.dtype),
                mkvec(y_len,self.dtype)
            )
            self.interp = self.spline3
        elif interp == 'poly3':
            mat = np.empty([16,16],dtype=self.dtype)
            arange = np.arange(4)
            v = (arange-1).reshape(-1,1) ** arange
            np.stack(
                [np.outer(vy,vx).ravel() for vy,vx in product(v,repeat=2)],
                out=mat
            )
            self.mat = cp.array(np.linalg.inv(mat),dtype=self.dtype)
            self.interp = self.poly3
        elif interp == 'linear':
            self.interp = self.linear
        elif interp == 'neighbor':
            self.interp = self.neighbor
        else:
            raise NotImplementedError(errmsg.format(interp))

        if   boundary=='neighbor':
            self.bound = self.neighbor
        elif boundary=='constant':
            self.bound = self.constant
        else:
            raise NotImplementedError(errmsg.format(boundary))

    def constant(self,data,dx,dy):
        return cp.full_like(data,self.const,dtype=self.dtype)

    def neighbor(self,data,dx,dy):
        shifted = cp.empty_like(data)
        nearest_neighbor(data,dx,dy,shifted)
        return shifted

    def linear(self,data,dx,dy):
        shifted = self.bound(data,dx,dy)
        bilinear(data,dx,dy,shifted[1:,1:])
        return shifted
    
    def poly3(self,data,dx,dy):
        shifted = self.linear(data,dx,dy)

        ex = (1 - dx) ** np.arange(4)
        ey = (1 - dy) ** np.arange(4)
        shift_vector = cp.asarray(np.outer(ey,ex).ravel(),dtype=self.dtype)
        shift_vector.dot(self.mat,out=shift_vector)
        shift_mat = shift_vector.reshape(4,4)
        
        polynomial(data,shift_mat,shifted[2:-1,2:-1])

        return shifted

    def spline3(self,data,dx,dy):
        vx, vy = self.vec
        shifted = self.bound(data,dx,dy)
        tmpd = cp.empty([self.x_len-1,self.y_len],dtype=self.dtype)
        spline1d(data.T,dx,vx,tmpd)
        spline1d(tmpd.T,dy,vy,shifted[1:,1:])
        return shifted

def imshift(data,shift,interp='spline3',boundary='neighbor',dtype=None):
    '''
    Shift an image

    Parameters
    ----------
    data : 2D ndarray
        An array of image.
    shifts : array-like
        A relative position of shifted image in units of pixel.
        The 1st item is interpreted as the value of X,
        the 2nd item as the value of Y.
    interp : str, default 'spline3'
        Subpixel interpolation algorithm in subpixel image shift.
        The choices are: 'spline3' (bicubic spline), 
        'poly3' (3rd order interior polynomial), 'linear' (bilinear), 
        and 'neighbor' (nearest neighbor).
    boundary : str, default 'neighbor'
        The choices are: 'neighbor', 'constant'.
    dtype : str or dtype, default None
        dtype of array used internally.
        If None, use eclair.common.default_dtype.
        If the input dtype is different, use a casted copy.
    
    Returns
    -------
    shifted : 2D cupy.ndarray
        An array of the shifted image
    '''
    dtype = judge_dtype(dtype)
    data = cp.asarray(data,dtype=dtype)
    shift = np.asarray(cp.asnumpy(shift),dtype=dtype)

    try:
        shift = shift.reshape(2)
    except ValueError:
        raise ValueError('shift must be a sequence consisting of 2 values')
    try:
        y_len, x_len = shape = np.array(data.shape)
    except ValueError:
        raise ValueError('data must have 2 dimensions')

    shifter = Shift(
        x_len=x_len, y_len=y_len, interp=interp,
        boundary=boundary, dtype=dtype
    )
    shifted = shifter.interp(data,*shift)

    return shifted


def imalign(
        data, shifts, interp='spline3', boundary='neighbor',
        trimimages=True, dtype=None):
    '''
    Stack the images with aligning their relative positions

    Parameters
    ----------
    data : 3D ndarray
        An array of images stacked along the 1st dimesion (axis=0).
    shifts : 2D ndarray
        An array of relative positions of images in units of pixel.
        Along the 1st axis, values of each images must be the same order
        as the 1st axis of "data".
        Along the 2nd axis, the 1st item is interpreted as 
        the value of X, the 2nd item as the value of Y.
    interp : str, default 'spline3'
        Subpixel interpolation algorithm in subpixel image shift.
        The choices are: 'spline3' (bicubic spline), 
        'poly3' (3rd order interior polynomial), 'linear' (bilinear), 
        and 'neighbor' (nearest neighbor).
    boundary : str, default 'neighbor'
        The choices are: 'neighbor', 'constant'.
    trimimages : bool, default True
        If True, the output images will be trimmed to include only the region
        over which they all overlap.
    dtype : str or dtype, default None
        dtype of array used internally.
        If None, use eclair.common.default_dtype.
        If the input dtype is different, use a casted copy.
    
    Returns
    -------
    aligned : 3D cupy.ndarray
        An array of images aligned and stacked along the 1st axis
    '''
    dtype = judge_dtype(dtype)
    data = cp.asarray(data,dtype=dtype)
    shifts = np.asarray(cp.asnumpy(shifts),dtype=dtype)

    try:
        nums, y_len, x_len = shape = np.array(data.shape)
    except ValueError:
        raise ValueError('data must have 3 dimensions')
    
    if nums != len(shifts):
        raise ValueError('data and shifts do not match')

    shift = Shift(
        x_len=x_len, y_len=y_len, interp=interp,
        boundary=boundary, dtype=dtype
    ).interp

    xy_i = np.floor(shifts).astype(int)
    xy_d = shifts - xy_i

    xy_i   -= xy_i.min(axis=0)
    x_u,y_u = xy_u = xy_i.max(axis=0)

    if trimimages:
        shape[1:] -= xy_u[::-1]
        copy_ = (
            lambda dst,src,ix,iy :
            cp.copyto(dst,src[y_u-iy:y_len-iy, x_u-ix:x_len-ix])
        )
    else:
        shape[1:] += xy_u[::-1]
        copy_ = (
            lambda dst,src,ix,iy :
            cp.copyto(dst[iy:iy+y_len,ix:ix+x_len],src)
        )

    aligned = cp.full(shape,cp.nan,dtype=dtype)
    for ixy,dxy,src,dst in zip(xy_i,xy_d,data,aligned):
        shifted = shift(src,*dxy)
        copy_(dst,shifted,*ixy)

    return aligned

def mkvec(n,dtype):
    asarray = lambda x:cp.asarray(x,dtype=dtype)

    vec1 = np.full(n-2,4,dtype=dtype)
    vec2 = np.ones(n-3,dtype=dtype)

    for i in range(n-3):
        vec2[i] /= vec1[i]
        vec1[i+1] -= vec2[i]

    return asarray(vec1), asarray(vec2)

def spline1d(data,d,vec,out):
    v1, v2 = vec
    u = cp.zeros_like(data)
    v_vector(data,u[1:-1])
    solve_tridiag(v1,v2,u[1:-1],size=u.shape[1])
    spline_core(u,data,1-d,out)

nearest_neighbor = cp.ElementwiseKernel(
    in_params='raw T input, T dx, T dy',
    out_params='T output',
    operation='''
        int rows = input.shape()[0];
        int cols = input.shape()[1];
        int ix = i % cols - (dx>=0.5);
        int iy = i / cols - (dy>=0.5);
        int idx[] = {
            max(0,min(rows-1,iy)),
            max(0,min(cols-1,ix))
        };
        output = input[idx];
    ''',
    name='nearest_neighbor'
)

bilinear = cp.ElementwiseKernel(
    in_params='raw T x, T dx, T dy',
    out_params='T z',
    operation='''
        typedef unsigned int uint;
        uint cols = x.shape()[1] - 1;
        uint ix = i % cols;
        uint iy = i / cols;
        uint i0[] = {iy, ix};
        uint i1[] = {iy, ix+1};
        uint i2[] = {iy+1, ix};
        uint i3[] = {iy+1, ix+1};
        T ex = 1-dx, ey = 1-dy;
        z = dy*(dx*x[i0] + ex*x[i1]) + ey*(dx*x[i2] + ex*x[i3]);
    ''',
    name='bilinear'
)

polynomial = cp.ElementwiseKernel(
    in_params='raw T input, raw T mat',
    out_params='T output',
    operation='''
        typedef unsigned int uint;
        uint width = input.shape()[1] - 3;
        uint i_x = i % width, i_y = i / width;
        uint idx1[2], idx2[2];
        uint *y1 = idx1, *x1 = idx1+1;
        uint *y2 = idx2, *x2 = idx2+1;
        output = 0;
        for (*y1=0, *y2=i_y; *y1<4; (*y1)++, (*y2)++) {
            T tmp = 0;
            for (*x1=0, *x2=i_x; *x1<4; (*x1)++, (*x2)++) {
                tmp = fma(mat[idx1],input[idx2],tmp);
            }
            output += tmp;
        }
    ''',
    name='polynomial'
)

v_vector = cp.ElementwiseKernel(
    in_params='raw T input',
    out_params='T output',
    operation='''
        unsigned int cols = input.shape()[1];
        T d0 = input[i];
        T d1 = input[i+cols];
        T d2 = input[i+2*cols];
        output = (d2-d1) - (d1-d0);
    ''',
    name='v_vector'
)

solve_tridiag = cp.ElementwiseKernel(
    in_params='raw T vec1, raw T vec2',
    out_params='raw T data',
    operation='''
        typedef unsigned int uint;
        uint h = data.shape()[0];
        uint idx[2] = {0, i};
        uint *j = idx;

        T tmp = (data[idx] /= vec1[*j]);

        // Forward elimination
        for ((*j)++; *j<h; (*j)++) {
            data[idx] -= tmp;
            tmp = (
                data[idx] /= vec1[*j]
            );
        }

        // Backward substitution
        uint c;
        for ((*j)--, c=0; c<h; (*j)--,c++) {
            tmp = (
                data[idx] -= tmp * vec2[*j]
            );
        }
    ''',
    name='solve_tridiag'
)

spline_core = cp.ElementwiseKernel(
    in_params='raw T u, raw T y, T d',
    out_params='T z',
    operation='''
        unsigned int i2 = i + u.shape()[1];
        T u1 = u[i], u2 = u[i2];
        T y1 = y[i], y2 = y[i2];
        T a1 = (y2-y1) - (2*u1+u2);
        T a2 = 3 * u1;
        T a3 = u2 - u1;
        z = y1 + d*(a1 + d*(a2 + d*a3))
    ''',
    name='spline'
)