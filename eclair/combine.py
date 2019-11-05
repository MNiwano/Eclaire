# -*- coding: utf-8 -*-

from itertools  import product
from os.path    import basename
import warnings

from astropy.io   import fits
from astropy.time import Time
import numpy    as np
import cupy     as cp

from param  import dtype
from io     import fitswriter
from kernel import (
    nonzerosum_kernel,
    filterdsum_kernel,
    filterdvar_kernel,
    replace_kernel,
    median_kernel,
    updatefilt_kernel,
)

#############################
#   imcombine
#############################

class SigClip:
    def __init__(self,combine='mean',center='mean',
            axis=0,dtype=dtype,returnfilter=False):

        self.axis    = axis
        self.dtype   = dtype
        self.rtnfilt = returnfilter

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

    def __call__(self,data,iter=3,width=3.0,filter=None):
        data = cp.asarray(data,dtype=self.dtype)

        if filter is None:
            filt = cp.ones_like(data)
        else:
            filt = cp.asarray(filter,dtype=self.dtype)

        for _ in range(iter):
            filt = self.genfilt(data,filt,width)

        if self.rtnfilt:
            return filt

        result = self.combine(data,filt)

        return result

    def genfilt(self,data,filt,width):
        mean  = self.mean(data,filt,keepdims=True)
        sigma = self.sigma(data,mean,filt)

        cent = self.center(mean,data,filt)
        
        updatefilt_kernel(data,filt,cent,sigma,width,filt)
        
        return filt
    
    def sigma(self,data,mean,filt):
        num = nonzerosum_kernel(filt,axis=self.axis,keepdims=True)
        sqm = filterdvar_kernel(data,mean,filt,axis=self.axis,keepdims=True)
        cp.divide(sqm,num,out=sqm)
        return cp.sqrt(sqm,out=sqm)

    def mean(self,data,filt,keepdims=False):
        num = nonzerosum_kernel(filt,axis=self.axis,keepdims=keepdims)
        sum = filterdsum_kernel(data,filt,axis=self.axis,keepdims=keepdims)
        return cp.divide(sum,num,out=sum)
    
    def median(self,data,filt,keepdims=False):
        y_len, x_len = data.shape[1:]
        nums = filt.sum(axis=0,keepdims=keepdims)

        tmpd = replace_kernel(data,filt,data.max(axis=0))
        tmpd.sort(axis=0)

        result = median_kernel(tmpd,nums,y_len*x_len,nums)
        
        return result

def imcombine(name,data,list=None,header=None,combine='mean',center='mean',
        iter=3,width=3.0,dtype=dtype,filter=None,memsave=False,overwrite=False):
    '''
    Calculate sigma-clipped mean or median of images,
    and write to FITS file

    Parameters
    ----------
    name : str
        A path of output FITS file
        Whether path like object is supported depends on
        the version of Python and Astropy.
    data : 3D ndarray
        An array of images stacked along the 1st dimension (axis=0)
    list : array-like, default None
        Names of image to combine
        These are written to the header.
        If the string is pathlike, only basename is used.
    header : astropy.io.fits.Header, default None
        Items of this header are appended to the header of the output file.
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
        If the input dtype is different, use a casted copy.
    filter : ndarray, default None
        array indicating which elements of data are used for calculation.
        The value must be 1 for elements to use and 0 for elements to ignore.
    memsave : bool, default False
        If True, split data and process it serially.
        Then, VRAM is saved, but speed may be slower.
    overwrite : bool, default False
        If True, overwrite the output file if it exists.
        Raises an IOError if False and the output file exists.
    '''
    sigclip = SigClip(combine,center,dtype=dtype)
    
    if header is None:
        header = fits.Header()

    nums, y_len, x_len = data.shape
    if memsave:
        lengthes = int(y_len/2), int(x_len/2)
        combined = cp.empty([y_len,x_len],dtype=dtype)
        slices = tuple((slice(l),slice(l,None)) for l in lengthes)
        for yslice, xslice in product(*slices):
            if filter is None:
                filt = None
            else:
                filt = filter[:,yslice,xslice]
            combined[yslice,xslice] = sigclip(
                data[:,yslice,xslice],iter,width,filter=filt
            )
    else:
        combined = sigclip(data,iter,width,filter=filter)

    if list:
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

    fitswriter(name,combined,header=header,overwrite=overwrite)

    print('Combine: {0:d} frames, Output: {1}'.format(nums,name))