# -*- coding: utf-8 -*-

from __future__ import division

from itertools  import product
from os.path    import basename
import warnings

from astropy.io   import fits
from astropy.time import Time
import numpy    as np
import cupy     as cp

from common import judge_dtype
from io     import mkhdu
from kernel import (
    elementwise_not,
    checkfinite,
    filteredsum,
    filteredvar,
    default_mean,
    replace_kernel,
    filtered_median,
    updatefilt_core,
)

#############################
#   imcombine
#############################

times = np.zeros(2,dtype=float)

class SigClip:
    def __init__(self, combine='mean', center='mean', axis=0, default=0,
        dtype=None, returnmask=False):

        self.axis    = axis
        self.default = default
        self.dtype   = judge_dtype(dtype)
        self.rtnmask = returnmask

        errormsg = '{0} is not impremented as {1}'

        if   center == 'mean':
            self.center = lambda mean,*args:mean
        elif center == 'median':
            self.center = lambda mean,*args:self.median(*args,keepdims=True)
        else:
            raise ValueError(errormsg.format(center,'center'))

        if   combine == 'mean':
            self.combine = self.mean
        elif combine == 'median':
            self.combine = self.median
        else:
            raise ValueError(errormsg.format(combine,'combine'))

    def __call__(self,data,iter=3,width=3.0,mask=None,keepdims=False):
        data = cp.asarray(data,dtype=self.dtype)

        if mask is None:
            filt = cp.ones_like(data)
        else:
            mask = cp.array(mask,dtype=self.dtype)
            filt = elementwise_not(mask,mask)
        
        checkfinite(data,filt,filt)

        for _ in range(iter):
            filt = self.updatefilt(data,filt,width)

        if self.rtnmask:
            return elementwise_not(filt,filt)

        result = self.combine(data,filt,keepdims=keepdims)

        return result

    def updatefilt(self,data,filt,width):
        mean  = self.mean(data,filt,keepdims=True)
        sigma = self.sigma(data,mean,filt)
        cent  = self.center(mean,data,filt)
        
        updatefilt_core(data,filt,cent,sigma,width,filt)
        
        return filt
    
    def sigma(self,data,mean,filt):
        fnum = filt.sum(axis=self.axis,keepdims=True)
        fsqm = filteredvar(data,mean,filt,axis=self.axis,keepdims=True)
        cp.divide(fsqm,fnum,out=fsqm)
        return cp.sqrt(fsqm,out=fsqm)

    def mean(self,data,filt,keepdims=False):
        fnum = filt.sum(axis=self.axis,keepdims=keepdims)
        fsum = filteredsum(data,filt,axis=self.axis,keepdims=keepdims)
        return default_mean(fsum,fnum,self.default,fsum)
    
    def median(self,data,filt,keepdims=False):
        if self.axis != 0:
            msg = 'Only axis=0 is supported for median'
            raise NotImplementedError(msg)

        nums = filt.sum(axis=0,keepdims=keepdims)

        tmpd = cp.where(filt,data,data.max(axis=0))
        tmpd.sort(axis=0)

        result = filtered_median(tmpd,nums,self.default,nums)
        
        return result

def imcombine(data,name=None,list=None,header=None,combine='mean',center='mean',
    iter=3,width=3.0,mask=None,dtype=None,memsave=False,**kwargs):
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
    combine : {'mean', 'median'}, default 'mean'
        An algorithm to combine images
    center : {'mean', 'median'}, default 'mean'
        An algorithm to get center value
    iter : int, default 3
        A number of sigmaclipping iterations
    width : int or float, default 3.0
        A clipping width in sigma units
    dtype : str or dtype, default 'float32'
        dtype of array used internally
        If None, this value will be usually "float32", 
        but this can be changed with eclair.set_dtype.
        If the input dtype is different, use a casted copy.
    mask : ndarray, default None
        array indicating which elements of data are masked for calculation.
        The value must be 0 for elements to use or nonzero to ignore.
    memsave : bool, default False
        If True, split data and process it serially.
        Then, VRAM is saved, but speed may be slower.
    kwargs : keywards arguments
        These are given to writeto method of HDU object
    '''

    sigclip = SigClip(combine,center,dtype=dtype)

    nums, y_len, x_len = data.shape
    if memsave:
        lengthes = y_len//2, x_len//2
        combined = cp.empty([y_len,x_len],dtype=dtype)
        slices = tuple(
            (slice(l),slice(l,None)) for l in lengthes
        )
        for yslice, xslice in product(*slices):
            if mask is None:
                tmpm = None
            else:
                tmpm = mask[:,yslice,xslice]
            combined[yslice,xslice] = sigclip(
                data[:,yslice,xslice],iter,width,mask=tmpm
            )
    else:
        combined = sigclip(data,iter,width,mask=mask)

    if name is not None:
        if header is None:
            header = fits.Header()

        if list is not None:
            if len(list) != nums:
                warnings.warn(
                    'Number of items is different between list and data'
                )
            if len(list) <= 999:
                key = 'IMCMB{:03d}'
            else:
                key = 'IMCMB{:03X}'
                msg = "IMCMB keys are written in hexadecimal."
                header.append('COMMENT',msg)
            for i,f in enumerate(list,1):
                header[key.format(i)] = basename(f)
        header['NCOMBINE'] = nums
        
        hdu = mkhdu(combined,header=header)
        
        hdu.writeto(name,**kwargs)

        print('Combine: {0:d} frames, Output: {1}'.format(nums,name))

    return combined