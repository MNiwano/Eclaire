# -*- coding: utf-8 -*-

from __future__ import division

from itertools  import product, count
from os.path    import basename
import warnings, sys

from astropy.io   import fits
from astropy.time import Time
import numpy    as np
import cupy     as cp

from .io     import mkhdu
from .util   import (
    judge_dtype,
    elementwise_not,
    checkfinite,
    replace_kernel,
)

#############################
#   imcombine
#############################

class SigClip:
    def __init__(self, reduce='mean', center='mean', axis=None,
        default=cp.nan, dtype=None, returnmask=False):

        if isinstance(axis,int) or axis is None:
            self.axis = axis
        else:
            self.axis = tuple(axis)
        self.default = default
        self.dtype   = judge_dtype(dtype)
        self.rtnmask = returnmask

        errormsg = '{0} is not impremented as {1}'

        if   center == 'mean':
            self.center = lambda mean,*args:mean
        elif center == 'median':
            self.center = lambda mean,*args:self.median(*args,keepdims=True)
        else:
            raise NotImplementedError(errormsg.format(center,'center'))

        if   reduce == 'mean':
            self.reduce = self.mean
        elif reduce == 'median':
            self.reduce = self.median
        else:
            raise NotImplementedError(errormsg.format(combine,'reduce'))

    def __call__(self,data,iters=5,width=3.0,
                    weights=None,mask=None,keepdims=False):

        data = cp.asarray(data,dtype=self.dtype)

        filt = cp.ones_like(data)
        if mask is not None:
            mask = cp.asarray(mask,dtype=self.dtype)
            elementwise_not(cp.broadcast_to(mask,data.shape),filt)
        if weights is not None:
            weights = cp.asarray(weights,dtype=self.dtype)
            try:
                filt *= weights
            except ValueError:
                if isinstance(self.axis,int):
                    ndim = data.ndim
                    axis = self.axis % ndim
                    if weights.size == data.shape[axis]:
                        filt *= weights.reshape(
                            -1 if i==axis else 1 for i in range(ndim)
                        )
                    else:
                        raise ValueError(
                            'length of weights must be same as'
                            ' the length of data along specified axis.'
                        )
                else:
                    raise ValueError(
                        'If weights and data are not broadcastable, '
                        'axis must be specified as int.'
                    )

        checkfinite(data,filt,filt)

        iterator = count() if (iters is None) else range(iters)

        csum = check_sum(filt,axis=self.axis)
        for _ in iterator:
            self.updatefilt(data,filt,width)
            tsum = check_sum(filt,axis=self.axis)
            if all_equal(csum,tsum):
                break
            else:
                csum = tsum

        if self.rtnmask:
            result = elementwise_not(filt,filt)
        else:
            result = self.reduce(data,filt,keepdims=keepdims)

        return result

    def updatefilt(self,data,filt,width):
        mean  = self.mean(data,filt,keepdims=True)
        sigma = self.sigma(data,mean,filt,keepdims=True)
        cent  = self.center(mean,data,filt)

        sigma *= width

        updatefilt_core(data,cent,sigma,filt)
    
    def sigma(self,data,mean,filt,keepdims=False):
        fnum = filt.sum(axis=self.axis,keepdims=keepdims)
        fsqm = weightedvar(data,mean,filt,axis=self.axis,keepdims=keepdims)
        nonzero_division(fsqm,fnum,0,fsqm)
        return cp.sqrt(fsqm,out=fsqm)

    def mean(self,data,filt,keepdims=False):
        fnum = filt.sum(axis=self.axis,keepdims=keepdims)
        fsum = weightedsum(data,filt,axis=self.axis,keepdims=keepdims)
        return nonzero_division(fsum,fnum,self.default,fsum)
    
    def median(self,data,filt,keepdims=False):
        nums = check_sum(filt,axis=self.axis,keepdims=keepdims)

        data, filt, axis = self.__reshape(data,filt)

        dmin, dmax = -cp.inf, cp.inf

        l = data.shape[axis]
        i = (l - 1) // 2
        fth = [i,min(i+1,l-1)]
        odd, eve = (dmin,dmax) if l%2 else (dmax,dmin)

        tmpd = elementwise_not(filt)
        tmpd.cumsum(axis=axis,out=tmpd)

        sweep_out(data,filt,tmpd,odd,eve,tmpd)
        tmpd.partition(fth,axis=axis)

        indice = lambda i:tuple(
            i if j==axis else slice(None) for j in range(tmpd.ndim)
        )
        d0, d1 = (tmpd[indice(f)] for f in fth)
        result = median_core(nums,d0,d1,self.default)

        return result

    def __reshape(self,data,filt):
        ndim = data.ndim
        if isinstance(self.axis,int):
            axis = self.axis % ndim
            reshape = lambda array: array
        elif self.axis is None:
            axis = 0
            reshape = cp.ravel
        else:
            move_axes = sorted(ax%ndim for ax in self.axis)
            axis = move_axes[0]
            naxs = len(move_axes)

            axes = sorted(set(range(ndim)) - set(move_axes))
            axes[axis:axis] = move_axes

            shape = [data.shape[ax] for ax in axes]
            shape[axis:axis+naxs] = [-1]

            reshape = (
                lambda array:
                array.transpose(*axes).reshape(*shape)
            )

        data = reshape(data)
        filt = reshape(filt)

        return data, filt, axis

def sigma_clipped_stats(data,axis=None,keepdims=False,mask=None,weights=None,
    reduce='mean',center='mean',iters=5,width=3.0,dtype=None,returnmask=False):
    '''
    Calculate sigma-clipped statistics on the provided data

    Parameters
    ----------
    data : array-like
    axis : int or sequence of int, default 0
        The axis or axes along which to sigma clip the data.
        If None, then the flattened data will be used.
    keepdims : bool, default False
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
    mask : array-like, default None
        A boolean mask associated with the values in data,
        where a True value indicates the corresponding element of data
        is masked. Masked pixels are excluded when computing the statistics.
    weights : array-like, default None
        array of weights associated with the values in data.
        This must be a broadcastable array with data.
        If axis is specified as an int, this can be a 1D array with
        the same length as data along specified axis.
    reduce : {'mean','median'}, default 'mean'
        reduce method after clipping
    center : {'mean','median'}, default 'mean'
        derivation method of clipping center
    iters : int, default 3
        A number of sigmaclipping iterations
        If None, clip until convergence is achieved.
    width : float or int, default 3.0
        A clipping width in sigma units
    dtype : str or dtype, default 'float32'
        dtype of array used internally
        If None, use eclair.common.default_dtype.
        If the input dtype is different, use a casted copy.
    returnmask : bool, default False
        If True, returns an array which indicates
        the corresponding value of data is clipped or not.
        At this time, reduction after clipping is not performed.

    Returns
    -------
    result : cupy.ndarray
        If returnmask is False, this is sigma-clipped mean or median.
        If returnmask is True, this is an array which indicates
        the corresponding value of data is clipped or not.
        This array has the same shape as data, and is 1 where
        the value is clipped and 0 where not.

    Notes
    -----
    If the number of values along specified axis is greater than 2^64,
    this function does not work properly.
    This is becuase whether clipping has converged or not is determined
    by comparing the number of remaining values before and after clipping
    and the number is stored as uint64.
    '''
    sigclip = SigClip(
        axis=axis, dtype=dtype,
        reduce=reduce, center=center, returnmask=returnmask
    )

    result = sigclip(
        data,iters=iters,width=width,weights=weights,mask=mask,
        keepdims=keepdims
    )

    return result

def imcombine(data,name=None,list=None,header=None,weights=None,mask=None,
    combine='mean',center='mean',iters=5,width=3.0,dtype=None,**kwargs):
    '''
    Combine images and optionally write to FITS file

    Parameters
    ----------
    data : 3D ndarray
        An array of images stacked along the 1st dimension (axis=0)
    name : str, default None
        A path of output FITS file
        If given, write the result to a FITS file.
        Whether path like object is supported depends on
        the version of Python and Astropy.
    list : array-like, default None
        Names of image to combine
        These are written to the header.
        If the string is path-like, only basename is used.
    header : astropy.io.fits.Header, default None
        The header associated with data.
        This is used only when the name is given.
        If None, a header of the appropriate type is created
        for the supplied data.
    weights : array-like, default None
        array of weights associated with the values in data.
        This must be a broadcastable array with data, or a 1D array.
        If 1D, the length of this must be the same as the number of images.
    mask : ndarray, default None
        A boolean mask associated with the values in data,
        where a True value indicates the corresponding element of data
        is masked. Masked pixels are excluded when computing the statistics.
    combine : {'mean', 'median'}, default 'mean'
        An algorithm to combine
    center : {'mean', 'median'}, default 'mean'
        An algorithm to get center value
    iters : int, default 3
        A number of sigmaclipping iterations
        If None, clip until convergence is achieved.
    width : int or float, default 3.0
        A clipping width in sigma units
    dtype : str or dtype, default 'float32'
        dtype of array used internally
        If None, use eclair.common.default_dtype.
        If the input dtype is different, use a casted copy.
    kwargs : keywards arguments
        Additional keywards arguments given to writeto method of HDU object
    
    Returns
    -------
    combined : 2D cupy.ndarray

    See Also
    --------
    sigma_clipped_stats : Calculate sigma-clipped statistics
    '''

    combined = sigma_clipped_stats(
        data, axis=0, mask=mask, weights=weights, dtype=dtype,
        reduce=combine, center=center, iters=iters, width=width
    )

    if name is not None:
        if header is None:
            header = fits.Header()
        else:
            header = fits.Header(header)

        ldata = len(data)
        if list is not None:
            llist = len(list)
            if llist != ldata:
                warnings.warn(
                    'Number of items is different between list and data'
                )
            if llist <= 999:
                key = 'IMCMB{:03d}'
            else:
                key = 'IMCMB{:03X}'
                msg = "IMCMB keys are written in hexadecimal."
                header.append('COMMENT',msg)
            for i,f in enumerate(list,1):
                header[key.format(i)] = basename(f)
        header['NCOMBINE'] = ldata
        
        hdu = mkhdu(combined,header=header)
        
        hdu.writeto(name,**kwargs)

        print(
            'Combine: {0:d} frames, Output: {1}'.format(ldata,basename(name))
        )
    
    return combined

check_sum = cp.ReductionKernel(
    in_params='T x',
    out_params='uint64 z',
    map_expr='(x!=0)',
    reduce_expr='a+b',
    post_map_expr='z=a',
    identity='0',
    name='check_sum'
)

weightedsum = cp.ReductionKernel(
    in_params='T x, T f',
    out_params='T y',
    map_expr='f * (f ? x : 0)',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    name='weightedsum'
)

weightedvar = cp.ReductionKernel(
    in_params='T x, T m, T f',
    out_params='T y',
    map_expr='square(x,m,f)',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    preamble='''
        template <typename T>
        __device__ T square(T x, T m, T f) {
            T dev = x-m;
            T var = dev*dev;
            return f * (f ? var : 0);
        }
    ''',
    name='weightedvar'
)

all_equal = cp.ReductionKernel(
    in_params='T x, T y',
    out_params='uint8 z',
    map_expr='x==y',
    reduce_expr='a & b',
    post_map_expr='z=a',
    identity='1',
    name='all_equal'
)

nonzero_division = cp.ElementwiseKernel(
    in_params='T x, T n, T d',
    out_params='T z',
    operation='''
        int f = (n==0);
        T q = x / (n+f);
        z = (f ? d : q);
    ''',
    name='nonzero_division'
)

sweep_out = cp.ElementwiseKernel(
    in_params='T data, T filt, T cum, T odd, T eve',
    out_params='T output',
    operation='''
        int isodd = (int)cum % 2;
        T tmp;
        if (filt!=0) {
            tmp = data;
        } else if (isodd) {
            tmp = odd;
        } else {
            tmp = eve;
        }
        output = tmp;
    ''',
    name='sweep_out'
)

median_core = cp.ElementwiseKernel(
    in_params='uint64 n, raw T x1, raw T x2, T d',
    out_params='T output',
    operation='''
        int isz = (n==0);
        int ise = 1 - n%2;
        T d1 = x1[i], d2 = x2[i];
        T q = (d1 + (ise ? d2 : d1)) / 2;
        output = (isz ? d : q);
    ''',
    name='median_core'
)

updatefilt_core = cp.ElementwiseKernel(
    in_params='T data, T cen, T lim',
    out_params='T filt',
    operation='''
        T dev = data-cen;
        filt *= (dev*dev <= lim*lim);
    ''',
    name='updatefilt'
)