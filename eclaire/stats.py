# -*- coding: utf-8 -*-

from __future__ import division

from itertools  import product, repeat
from os.path    import basename
import warnings
import sys

from astropy.io   import fits
from astropy.time import Time
import numpy as np
import cupy  as cp

from .io   import mkhdu
from .util import (
    judge_dtype,
    elementwise_not,
    checkfinite,
    replace_kernel,
    ternary_operation
)

#############################
#   imcombine
#############################

class SigClip:

    def __init__(
        self, reduce='mean', center='mean', axis=None,
        default=cp.nan, dtype=None):

        if isinstance(axis,int) or axis is None:
            self.axis = axis
        else:
            self.axis = tuple(axis)
        self.default = default
        self.dtype   = judge_dtype(dtype)

        errormsg = '{0} is not impremented as {1}'

        callback = {
            'mean': lambda mean,*args,**kwargs: mean,
            'median': lambda mean,*args,**kwargs: self.median(*args,**kwargs),
        }

        try:
            self.center = callback[center]
        except KeyError as e:
            raise NotImplementedError(errormsg.format(center,'center'))

        try:
            self.reduce = callback[reduce]
        except KeyError as e:
            raise NotImplementedError(errormsg.format(reduce,'reduce'))

    def __call__(
            self, data, iters=5, width=3.0,
            weights=None, mask=None, keepdims=False):

        data = cp.asarray(data,dtype=self.dtype)

        filt = cp.full_like(data,True,dtype=bool)
        if mask is not None:
            mask = cp.broadcast_to(cp.asarray(mask),data.shape)
            cp.logical_not(mask,out=filt)

        if weights is None:
            weights = cp.asarray(1,dtype=self.dtype)
        else:
            weights = cp.asarray(weights,dtype=self.dtype)
            try:
                weights = cp.broadcast_to(weights,data.shape)
            except ValueError:
                if isinstance(self.axis,int):
                    ndim = data.ndim
                    axis = self.axis % ndim
                    if weights.size == data.shape[axis]:
                        weights = weights.reshape(
                            *(-1 if i==axis else 1 for i in range(ndim))
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

        checkfinite(data,weights,filt)

        if iters is None:
            iterator = repeat(None)
        else:
            try:
                iterator = repeat(None,int(iters))
            except ValueError as e:
                ValueError('iter must be integer')

        csum = check_sum(filt,axis=self.axis)
        tsum = cp.empty_like(csum)
        for _ in iterator:
            self.updatefilt(data,filt,weights,width)
            check_sum(filt,tsum,axis=self.axis)
            if cp.array_equal(csum,tsum):
                break
            else:
                tsum, csum = csum, tsum
        del csum, tsum

        mean = self.mean(data,filt,weights,keepdims=True)
        stddev = self.sigma(data,filt,weights,mean,keepdims=keepdims)
        reduced = self.reduce(mean,data,filt,weights,keepdims=keepdims)
        mask = cp.logical_not(filt,out=filt)

        return reduced.reshape(stddev.shape), stddev, mask

    def updatefilt(self,data,filt,weights,width):
        mean  = self.mean(data,filt,weights,keepdims=True)
        sigma = self.sigma(data,filt,weights,mean=mean,keepdims=True)
        cent  = self.center(mean,data,filt,weights,keepdims=True)

        sigma *= width

        updatefilt_core(data,cent,sigma,filt)
    
    def sigma(self,data,filt,weights,mean,keepdims=False):
        fnum = weightedsum(
            1, filt, weights, axis=self.axis, keepdims=keepdims
        )
        fsqm = weightedvar(
            data, mean, filt, weights, axis=self.axis, keepdims=keepdims
        )
        nonzero_division(fsqm,fnum,0,fsqm)
        return cp.sqrt(fsqm,out=fsqm)

    def mean(self,data,filt,weights,keepdims=False):
        fnum = weightedsum(
            1, filt, weights, axis=self.axis, keepdims=keepdims
        )
        fsum = weightedsum(
            data, filt, weights, axis=self.axis, keepdims=keepdims
        )
        return nonzero_division(fsum,fnum,self.default,fsum)

    def median(self,data,filt,weights,keepdims=False):

        tmpd = cp.where(filt,data,np.nan)

        result = cp.nanmedian(
            tmpd, axis=self.axis, overwrite_input=True, keepdims=keepdims
        )

        return result

def sigma_clipped_stats(data,axis=None,keepdims=False,mask=None,weights=None,
    reduce='mean',center='mean',iters=5,width=3.0,dtype=None):
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
        If None, use eclaire.common.default_dtype.
        If the input dtype is different, use a casted copy.
    
    Returns
    -------
    reduced : cupy.ndarray
        sigma-clipped mean or median
    stddev : cupy.ndarray
        sigma-clipped standard deviation
    mask : cupy.ndarrray
        an array which indicates the corresponding value
        of data is clipped or not.
        This array has the same shape as data, and is True where
        the value is clipped and False where not.

    Notes
    -----
    If the number of values along specified axis is greater than 2^64,
    this function does not work properly.
    This is becuase convergence of clipping is determined
    by comparing the number of remaining values before and after clipping
    and the number is stored as uint64.
    '''
    sigclip = SigClip(
        axis=axis, dtype=dtype, reduce=reduce, center=center
    )

    result = sigclip(
        data,iters=iters,width=width,weights=weights,mask=mask,
        keepdims=keepdims
    )

    return result

def imcombine(data,name=None,list=None,header=None,weights=None,mask=None,
    combine='mean',center='mean',iters=5,width=3.0,dtype=None,exthdus=[],
    **kwargs):
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
    
    combined, *_ = sigma_clipped_stats(
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
                comment = 'IMCMB keys are written in hexadecimal.'
                header.append('COMMENT',comment)
            for i,f in enumerate(list,1):
                header[key.format(i)] = basename(f)
        header['NCOMBINE'] = ldata
        
        hdu = mkhdu(combined,header=header)

        hdul = fits.HDUList(
            [hdu] + exthdus
        )
        hdul.writeto(name,**kwargs)
        print(
            'Combine: {:d} frames, Output: {}'.format(ldata,basename(name))
        )
    
    return combined

check_sum = cp.ReductionKernel(
    in_params='T x',
    out_params='uint64 z',
    map_expr='(x!=0)',
    reduce_expr='a+b',
    reduce_type='unsigned long long',
    post_map_expr='z=a',
    identity='0',
    name='check_sum'
)

weightedsum = cp.ReductionKernel(
    in_params='T x, I f, T w',
    out_params='T y',
    map_expr='w * (f? x : 0)',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    name='weightedsum'
)

weightedvar = cp.ReductionKernel(
    in_params='T x, T m, I f, T w',
    out_params='T y',
    map_expr='square(x,m,f,w)',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    preamble='''
    template <typename T, typename I>
    __device__ T square(T x, T m, I f, T w) {
        T dev = x-m;
        T var = dev*dev;
        return w * (f? var : 0);
    }
    ''',
    name='weightedvar'
)

nonzero_division = cp.ElementwiseKernel(
    in_params='T x, T n, T d',
    out_params='T z',
    operation='''
        char f = (n==0);
        T q = x / n;
        z = (f? d : q);
    ''',
    name='nonzero_division'
)

updatefilt_core = cp.ElementwiseKernel(
    in_params='T data, T cen, T lim',
    out_params='I filt',
    operation='''
        filt &= (abs(data-cen) <= lim);
    ''',
    name='updatefilt'
)