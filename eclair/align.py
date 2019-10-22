
# -*- coding: utf-8 -*-

from itertools  import product

import numpy as np
import cupy  as cp

from param import dtype

from kernel import (
    neighbor_kernel,
    linear_kernel,
    spline_kernel,
)

#############################
#   imalign
#############################

class ImAlign:
    '''
    Generate imalign function
    Instance of this class can be used as function to align images.

    Attributes
    ----------
    x_len, y_len : int
        Shape of images to align
    interp : {'spline3', 'poly3', 'linear', 'neighbor'}
        Subpixel interpolation algorithm in subpixel image shift
         spline3  - 3rd order spline interpolation
         poly3    - 3rd order polynomial interpolation
         linear   - linear interpolation
         neighbor - nearest neighbor
    '''
    
    def __init__(self,x_len,y_len,interp='spline3',dtype=dtype):
        '''
        Parameters
        ----------
        x_len, y_len : int
            Shape of images to align
        interp : {'spline3', 'poly3', 'linear', 'neighbor'}, default 'spline3'
            Subpixel interpolation algorithm in subpixel image shift
             spline3 - bicubic spline
             poly3   - 3rd order interior polynomial
             linear  - bilinear
             neighbor - nearest neighbor
        '''
        self.x_len  = x_len
        self.y_len  = y_len
        self.interp = interp
        if   interp == 'spline3':
            self.shift = self.__spline
            self.matx = Ms(x_len)
            if y_len == x_len:
                self.maty = self.matx.view()
            else:
                self.maty = Ms(y_len)
        elif interp == 'poly3':
            self.shift = self.__poly
            self.mat = Mp()
        elif interp == 'linear':
            self.shift = self.__linear
        elif interp == 'neighbor':
            self.shift = self.__neighbor
        else:
            raise ValueError('"%s" is not inpremented'%interp)
        
    def __call__(self,data,shifts,reject=False,baseidx=None,tolerance=None,
                 selected=None,progress=lambda *args:None,args=()):
        '''
        self.__call__(*args,**kwargs) <==> self(*args,**kwargs)

        Stack the images with aligning their relative positions,
        and cut out the overstretched area

        Parameters
        ----------
        data : 3-dimension cupy.ndarray
            An array of images stacked along the 1st axis
            If the shape of image is differ from attributes x_len, y_len,
            ValueError is raised.
        shifts : 2-dimension numpy.ndarray
            An array of relative positions of images in units of pixel
            Along the 1st axis, values of each images must be the same order
            as the 1st axis of "data".
            Along the 2nd axis, the 1st item is interpreted as 
            the value of X, the 2nd item as the value of Y.
        reject : bool, default False
            If True, reject too distant image.
            Then, you must input baseidx, tolerance and selected.
        baseidx : int, default None
            Index of base image
            If you set reject True, set also this parameter.
        tolerance : int or float, default None
            Maximum distance from base image, in units of pixel
            If you set reject True, set also this parameter.
        selected : variable referencing list object, default None
            List for storing indices of selected images
            If you set reject True, set also this parameter.
        progress : function, default lambda *args:None
            Function to be executed simultaneously with aligning
            and given arguments (i, *args), where i is index of image
            If you want to do something simultaneously with aligning,
            input as a function. (e.g. logging, showing progress bar, etc)
        args : tupple
            arguments given additionally to the progress function

        Returns
        -------
        align : 3-dimension cupy.ndarray
            An array of images aligned and stacked along the 1st axis
        '''
        nums, y_len, x_len = data.shape
        if (y_len,x_len) != (self.y_len,self.x_len):
            message = 'shape of images is differ from (%d,%d)'
            raise ValueError(message%(self.y_len,self.x_len))
        if nums != len(shifts):
            raise ValueError('data and shifts do not match')

        lowlim = np.floor(shifts.min(axis=0))

        shifts = shifts - lowlim
        x_u,y_u = np.floor(shifts.max(axis=0)).astype('int')
    
        xy_i = np.floor(shifts).astype('int')
        xy_d = shifts - xy_i

        iterator = zip(xy_i,xy_d,data)
        if reject:
            if isinstance(baseidx,int):
                raise ValueError('baseidx is invalid')
            elif isinstance(tolerance,(int,float)):
                raise ValueError('tolerance is invalid')
            elif isinstance(selected,list):
                raise ValueError('selected is invalid')
            else:
                norm  = np.linalg.norm(shifts-shifts[baseidx,:],axis=1)
                flags = (norm <= tolerance)
                nums  = flags.sum()
                selected += list(np.where(flags)[0])
                iterator = (i for i,f in zip(iterator,flags) if f)
                
        aligned = cp.empty([nums,y_len-y_u,x_len-x_u],dtype=dtype)
        for i,((ix,iy),(dx,dy),layer) in enumerate(iterator):
            aligned[i] = self.shift(layer,dx,dy)[
                y_u-iy : y_len-iy,
                x_u-ix : x_len-ix
            ]
            progress(i,*args)

        return aligned

    def __neighbor(self,data,dx,dy):
        shifted = cp.empty_like(data,dtype=dtype)
        neighbor_kernel(data,dx,dy,self.x_len,shifted)
        return shifted

    def __linear(self,data,dx,dy):
        shifted = self.__neighbor(data,dx,dy)
        linear_kernel(data,dx,dy,self.x_len,shifted[1:,1:])
        return shifted
    
    def __poly(self,data,dx,dy):
        x_len = self.x_len-3
        y_len = self.y_len-3

        shifted = self.__linear(data,dx,dy)

        shift_vector = cp.empty([16],dtype=dtype)
        stack = cp.empty([16,y_len,x_len],dtype=dtype)
        for i, j in product(range(4),repeat=2):
            shift_vector[i*4+j] = (1-dx)**i * (1-dy)**j
            stack[i*4+j,:,:]  = data[i:i+y_len,j:j+x_len]

        tmpmat = cp.dot(shift_vector,self.mat)
        shifted[2:-1,2:-1] = cp.tensordot(tmpmat,stack,1)

        return shifted

    def __spline(self,data,dx,dy):
        shifted = self.__neighbor(data,dx,dy)
        tmpd = cp.empty([self.x_len-1,self.y_len],dtype=dtype)
        self.__spline1d(data.T,dx,self.matx,tmpd)
        self.__spline1d(tmpd.T,dy,self.maty,shifted[1:,1:])
        return shifted

    def __spline1d(self,data,d,mat,out):
        v = data[2:,:]+data[:-2,:]-2*data[1:-1,:]
        u = cp.zeros_like(data,dtype=dtype)
        cp.dot(mat,v,out=u[1:-1,:])
        spline_kernel(u,data,1-d,out.shape[-1],out)

def imalign(data,shifts,interp='spline3',reject=False,baseidx=None,
            tolerance=None,selected=None):
    '''
    Stack the images with aligning their relative positions,
    and cut out the overstretched area
    This function uses class eclair.ImAlign internally.

    Refer to eclair.ImAlign for documentation of parameters and return.

    See also
    --------
    eclair.ImAlign : Class to generate imalign function
    '''
    y_len, x_len = data.shape[1:]
    func = ImAlign(x_len=x_len,y_len=y_len,interp=interp)

    return func(data,shifts,baseidx=baseidx,reject=reject,
                tolerance=tolerance,selected=selected)

def Mp():
    Mp = np.empty([16,16],dtype=dtype)
    for y,x,k,l in product(range(4),repeat=4):
        Mp[y*4+x,k*4+l] = (x-1)**k * (y-1)**l
    Mp = cp.array(np.linalg.inv(Mp))

    return Mp

def Ms(ax_len):
    Ms = 4 * np.identity(ax_len-2,dtype=dtype)
    Ms[1:  , :-1] += np.identity(ax_len-3,dtype=dtype)
    Ms[ :-1,1:  ] += np.identity(ax_len-3,dtype=dtype)
    Ms = cp.array(np.linalg.inv(Ms))

    return Ms