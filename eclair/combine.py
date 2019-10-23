# -*- coding: utf-8 -*-

from itertools  import product
from os.path    import basename
from warnings   import warn
from copy       import copy
import time

from astropy.io import fits
import numpy    as np
import cupy     as cp

from param import dtype, origin

from kernel import (
    filterdsum_kernel,
    filterdstd_kernel,
    median_kernel,
    clip_kernel,
    replace_kernel,
)

#############################
#   imcombine
#############################

class SigClip:
    def __init__(self,combine='mean',center='mean',
        axis=0,dtype=dtype,returnfilter=False):
        self.combine = combine
        self.center  = center
        self.axis    = axis
        self.dtype   = dtype
        self.rtnfilt = returnfilter

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

        if self.combine == 'mean':
            result = self.mean(data,filt)
        else:
            result = self.median(data,filt)

        return result

    def genfilt(self,data,filt,width):
        mean  = self.mean(data,filt,keepdims=True)
        sigma = self.sigma(data,mean,filt)
        if self.center == 'mean':
            cent = mean.view()
        else:
            cent = self.median(data,filt)
        
        clip_kernel(data,filt,cent,sigma,width,filt)
        
        return filt
    
    def sigma(self,data,mean,filt):
        num = filt.sum(axis=self.axis,keepdims=True)
        replace_kernel(num,1,num)
        sqm = filterdstd_kernel(data,mean,filt,axis=self.axis,keepdims=True)
        return cp.sqrt(sqm/num)

    def mean(self,data,filt,keepdims=False):
        num = filt.sum(axis=self.axis,keepdims=keepdims)
        replace_kernel(num,1,num)
        sum = filterdsum_kernel(data,filt,axis=self.axis,keepdims=keepdims)
        return sum/num
    
    def median(self,data,filt):
        y_len, x_len = data.shape[1:]
        nums = filt.sum(axis=0)
        even = 1-(nums%2)

        tmpf = cp.zeros_like(data)
        tmpd = median_kernel(data,filt,data.max(axis=0))
        tmpd.sort(axis=0)

        z1 = np.ceil((nums/2 - 1).get()).astype(int)
        x1, y1 = np.meshgrid(np.arange(x_len),np.arange(y_len))

        y2, x2 = np.where(even.get())
        z2 = z1[y2, x2] + 1

        tmpf[z1.ravel(),y1.ravel(),x1.ravel()] = 1
        tmpf[z2,y2,x2] = 1

        return self.mean(tmpd,tmpf)

def imcombine(name,data,list=None,header=None,combine='mean',center='mean',
        iter=3,width=3.0,dtype=dtype,filter=None,memsave=False,overwrite=False):
    '''
    Calculate sigma-clipped mean or median of images,
    and write to FITS file

    Parameters
    ----------
    name : str
        A name of output FITS file
    data : 3-dimension cupy.ndarray
        An array of images stacked along the 1st axis
    list : array-like, default None
        Names of images combined
        These are written to the header.
    header : astropy.io.fits.Header, default None
        A header for output FITS file
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
    memsave : bool, default False
        If True, divide data and calculate it serially.
        Then, VRAM is saved, but speed may be slower.
    overwrite : bool, default False
        If True, overwrite the output file if it exists.
        Raises an IOError if False and the output file exists.
    '''

    for v, k in zip((combine,center),('combine','center')):
        if v not in ('mean','median'):
            raise ValueError('"%s" is not impremented as %s'%(v,k))
    sigclip = SigClip(combine,center,dtype=dtype)

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

    now_ut = time.strftime('%Y/%m/%dT%H:%M:%S',time.gmtime())

    hdu = fits.PrimaryHDU(
        data=combined.get(),
        header=copy(header),
    )

    hdu.header.insert(6,('DATE',now_ut,'Date FITS file was generated'))
    if list:
        if len(list) != nums:
            warn('Number of items is different between list and data')
        for i,f in enumerate(list,1):
            hdu.header['IMCMB%03d'%i] = basename(f)
    hdu.header['NCOMBINE'] = nums
    hdu.header.append(origin)

    hdu.writeto(name,overwrite=overwrite)

    print('Combine: %d frames, Output: %s'%(nums,name))