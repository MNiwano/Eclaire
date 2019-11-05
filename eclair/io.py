# -*- coding: utf-8 -*-

import sys
from os.path     import isfile
from collections import namedtuple

if sys.version_info.major == 2:
    from itertools import izip as zip

from astropy.io   import fits
from astropy.time import Time
import cupy  as cp
import numpy as np

from param import dtype, origin, null

_list = list

def fitsloader(name,hdu_index=0,dtype=dtype,xp=cp,
        xmin=0,xmax=None,ymin=0,ymax=None):
    '''
    load FITS header and data

    Parameters
    ----------
    name : str
        FITS file path
        Whether path like object is supported depends on
        the version of Python and Astropy.
    hdu_index : int, default 0
        HDU index to read in HDU List
    dtype : str or dtype, default 'float32'
        dtype of the return array
    xp : module object of numpy or cupy, default cupy
        Whether the return array is numpy.ndarray or cupy.ndarray
    xmin : int, default 0
    xmax : int, default None
    ymin : int, default 0
    yamx : int, default None
        return the area [ymin:ymax, xmin:xmax] of the array.

    Returns
    -------
    header : astropy.io.fits.Header
        FITS header
    data : ndarray
        FITS image data
    '''
    with fits.open(name) as hdul:
        hdu = hdul[hdu_index]

        header = hdu.header
        data = xp.asarray(hdu.data[ymin:ymax,xmin:xmax],dtype=dtype)

        try:
            header['CRPIX1'] -= xmin
            header['CRPIX2'] -= ymin
        except:
            pass
        
    return header, data

def fitswriter(name,data,header=None,overwrite=False):
    '''
    Create FITS file with given header and data

    Parameters
    ----------
    name : str
        FITS file path
    data : ndarray
        Image data array of FITS file
    header : astropy.io.fits.Header, default None
        Header of FITS file
    overwrite : bool, default False
        If True, overwrite the output file if it exists.
        Raises an IOError if False and the output file exists.
    '''

    hdu = fits.PrimaryHDU(data=cp.asnumpy(data))
    now = Time.now().isot

    hdu.header.append(origin)
    hdu.header.append(
        ('DATE',now,'Date FITS file was generated')
    )
    if header is not None:
        hdu.header.extend(header)

    hdu.writeto(name,overwrite=overwrite)

class FitsContainer:
    '''
    container class for storing FITS name, header, data

    Attributes
    ----------
    list : array-like
        List of FITS file paths
    header : list
        List of FITS header
        Before calling load method, it is a list of None.
    data : 3D cupy.ndarray
        array of image data stacked along 1st axis
        Before calling load method, it is a list of None.
    slices : dict
        When executing the load method,
        read the area [ymin:ymax, xmin:xmax] of the FITS data array.
    dtype : str or dtype
        dtype of cupy.ndarray

    Notes
    -----
    The methods implemented by this class interpret the attributes
    list, header, and data are sorted in the same order.
    '''

    def __init__(self,list,dtype=dtype):
        '''
        Parameters
        ----------
        list : iterable
            List of FITS file paths
            load() method refer this list to load FITS file.
            Whether path like object is supported depends on 
            the version of Python and Astropy.
        dtype : str or dtype, default 'float32'
            dtype of cupy.ndarray
        '''
        self.list  = _list(list)
        self.dtype = dtype

        self.slices = dict(xmin=0,xmax=None,ymin=0,ymax=None)
        
        self.header = [None] * len(list)
        self.data   = self.header

    def __getitem__(self,idx):
        '''
        self.__getitem__(idx) <==> self[idx]

        Parameters
        ----------
        idx : int
        
        Returns
        -------
        name : str
            FITS file name
        header : astropy.io.fits.Header
            FITS header
        data : cupy.ndarray
            FITS data 
        '''
        return self.list[idx], self.header[idx], self.data[idx]
    
    def __iter__(self):
        '''
        self.__iter__() <==> iter(self)

        The return of the iterator is the same as __getitem__
        '''
        return zip(self.list,self.header,self.data)

    def __len__(self):
        '''
        self.__len__() <==> len(self)

        If the number of list or header or data does not match,
        raise ValueError.
        '''
        n_l = len(self.list)
        n_h = len(self.header)
        n_d = len(self.data)
        if (n_l != n_h) or (n_l != n_d):
            msg = 'the number of list or header or data does not match'
            raise ValueError(msg)
        return n_l

    def setslice(self,xmin=0,xmax=None,ymin=0,ymax=None):
        '''
        Set data slice range

        Parameters
        ----------
        xmin : int, default 0
        xmax : int, default None
        ymin : int, default 0
        ymax : int, default None
        '''
        self.slices = dict(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)

    def clip(self,indices):
        '''
        clip out items in attributes list, header, data
        that are not in given indices

        Parameters
        ----------
        indices : array-like
            indices of items to leave
        '''
        indices = sorted(indices)

        list = self.list
        head = self.header
        data = self.data
        self.list   = [list[i] for i in indices]
        self.header = [head[i] for i in indices]
        for i,j in enumerate(indices):
            data[i] = data[j]
        self.data = data[:len(indices)]

    def load(self,hdu_index=0,check_exists=False,func=null,args=()):
        '''
        Load containts of FITS in the list

        Parameters
        ----------
        hdu_index : int, default 0
            HDU index to read in HDU List
        check_exists : bool, default False
            Before reading, check the existence of the file in list
            and delete the non-existing file from the list

        Notes
        -----
        The existence check of the file is performed by os.path.isfile.
        So it cannot detect the file that exists but has invalid contents.
        '''
        loader = lambda x:fitsloader(x,dtype=self.dtype,xp=np,**self.slices)
        if check_exists:
            self.list = filter(isfile,self.list)
        iterator = enumerate(self.list)
        try:
            i,f = next(iterator)
            head, data = loader(f)

            y_len, x_len = data.shape
            array = cp.empty([len(self.list),y_len,x_len],dtype=dtype)

            self.header[i] = head
            array[i].set(data)
            func(i,*args)
            for i,f in iterator:
                head, data = loader(f)
                self.header[i] = head
                array[i].set(data)
                func(i,*args)

            self.data = array
        except StopIteration:
            raise ValueError('list is empty')

    def write(self,outlist,overwrite=False,func=null,args=()):
        '''
        make FITS file with storing FITS header and data

        Parameters
        ----------
        outlist : array-like
            list of output FITS paths
            raise ValueError if the length of this list is different
            from the number of images.
        overwrite : bool, default False
            If True, overwrite the output file if it exists.
            Raises an IOError if False and the output file exists.
        '''
        if len(outlist) != len(self):
            msg = 'the length of outlist differs from the number of images'
            raise ValueError(msg)
        for i,(o,head,data) in enumerate(zip(outlist,self.header,self.data)):
            fitswriter(o,data,header=head,overwrite=overwrite)
            func(i,*args)