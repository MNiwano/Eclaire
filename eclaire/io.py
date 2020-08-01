# -*- coding: utf-8 -*-

import sys
import copy
import time

if sys.version_info.major == 2:
    from future_builtins import zip, map
    from collections     import Sized, Iterable
    import __builtin__ as builtins
else:
    from collections.abc import Sized, Iterable
    import builtins

from astropy.io   import fits
from astropy.time import Time
import cupy  as cp
import numpy as np

from .common import __version__, __update__
from .util   import judge_dtype

origin = (
    'Eclair v{version} {date}'.format(version=__version__,date=__update__),
    'FITS file originator'
)

null1 = lambda x:x
null2 = lambda *args:args

class FitsContainer(object):
    '''
    Class for storing multiple FITS data and performing SIMD processing.
    
    Notes
    -----
    The methods implemented by this class interpret the attributes
    list, header, and data are sorted in the same order.
    '''

    def __init__(self,list=[],header=[],data=[],dtype=None):
        '''
        Parameters
        ----------
        list : array-like
            List of FITS file names
        header : array-like
            List of FITS header
        data : array-like
            Array of image data stacked along 1st axis
        dtype : str or dtype
            dtype of ndarray.
            If None, use eclair.common.default_dtype.
        '''
        self.dtype = judge_dtype(dtype)
        
        self.list   = list
        self.header = header
        self.data   = data

    @property
    def list(self):
        '''
        List of FITS file names
        '''
        return self.__list
    
    @list.setter
    def list(self,seq):
        self.__list = builtins.list(seq)

    @property
    def header(self):
        '''
        List of FITS header
        '''
        return self.__header
    
    @header.setter
    def header(self,seq):
        self.__header = builtins.list(seq)

    @property
    def data(self):
        '''
        Array of image data stacked along 1st axis
        '''
        return self.__data

    @data.setter
    def data(self,array):
        self.__data = cp.asarray(array,dtype=self.dtype)

    def __getitem__(self,idx):
        '''
        self.__getitem__(idx) <==> self[idx]
        '''
        list_  = self.list
        header = self.header
        data   = self.data
        if isinstance(idx,int):
            return list_[idx], header[idx], data[idx]
        elif idx is Ellipsis or isinstance(idx,slice):
            view = copy.copy(self)
            view.list   = list_[idx]
            view.header = header[idx]
            view.data   = data[idx]
            return view
        elif isinstance(idx,Iterable):
            indices = builtins.list(idx)
            copied = copy.copy(self)
            if (all(isinstance(i,bool) for i in indices)
                    and len(indices)==len(self)):
                copied.list   = (f for i,f in zip(indices,list_) if i)
                copied.header = (h for i,h in zip(indices,header) if i)
                copied.data   = data[indices]
            elif all(isinstance(i,int) for i in indices):
                copied.list   = (list_[i] for i in indices)
                copied.header = (header[i] for i in indices)
                copied.data   = data[indices]
            else:
                raise ValueError('given sequence is invalid')
            return copied
        else:
            raise TypeError('Invalid index')
    
    def __iter__(self):
        '''
        self.__iter__() <==> iter(self)
        '''
        return zip(self.list,self.header,self.data)

    def __len__(self):
        '''
        self.__len__() <==> len(self)
        '''
        n_l = len(self.list)
        n_h = len(self.header)
        n_d = len(self.data)
        if (n_l == n_h) and (n_l == n_d):
            return n_l
        else:
            msg = 'the number of list or header or data does not match'
            raise ValueError(msg)

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
        self.list   = (list[i] for i in indices)
        self.header = (head[i] for i in indices)
        for i,j in enumerate(indices):
            data[i] = data[j]
        self.data = data[:len(indices)]

    def extend(self,*args):
        '''
        Add containsts of the other instances to self.

        Parameters
        ----------
        *args : FitsContainer or its subclass
            instances which merges with self
        '''
        if not args:
            return self
        
        if not all(isinstance(arg,FitsContainer) for arg in args):
            raise TypeError('Only support FitsContainer or its subclasses')

        self.list = sum(
            (arg.list for arg in args),
            self.list
        )
        self.header = sum(
            (arg.header for arg in args),
            self.header
        )

        try:
            self.data = cp.concatenate(
                [self.data] + [arg.data for arg in args],
                axis=0
            )
        except ValueError:
            raise ValueError('data shape mismatch')

        return self

    def from_iterator(self,iter,length=None,mapping=None,wrapper=None):
        '''
        Load headers and data from iterator

        Parameters
        ----------
        iter : iterable
            An iterator that returns header and data
            This must return a tuple (header, data).
            'header' and 'data' must be instances of astropy.io.fits.Header
            and 2d-numpy.ndarray, respectively.
        length : int, default None
            A length of iter. If iter does not support a __len__ method,
            this value must be specified.
        mapping : callable, default None
            If not None, pass the return value of iter (header, data) to
            this object and use its return as (header, data).
            It must be a callable object which receives returns of iter
            as arguments with unpacking, and returns the same format tuple
            after some processing.
        wrapper : callable, default None
            A wrapper of iterator
            If not None, use iterwrap(iter) as iterator.
            It must be a callable object that takes iter as an argument
            and returns an iterable which returns values in the same format
            as iter.

        Notes
        -----
        'length' is used in the initialization of a cupy.ndarray which stores
        the data. The reason why the ndarray is not initialized after reading
        all containts of iter is that it generates latency in host memory
        allocation.
        '''
        if length is None:
            length = len(iter)
        else:
            length = int(length)

        if mapping is None:
            mapping = null2
        if wrapper is None:
            wrapper = null1

        iterator = builtins.iter(
            wrapper(mapping(*args) for args in iter)
        )

        try:
            header, data = next(iterator)
        except StopIteration:
            raise ValueError('iterator is empty')
        else:
            tmp_head = []
            tmp_data = cp.empty([length] + list(data.shape),dtype=self.dtype)
            happend = tmp_head.append

            array_iter = builtins.iter(tmp_data)
            target = next(array_iter)

            happend(header)
            target.set(data)

            for (header, data), target in zip(iterator,array_iter):
                happend(header)
                target.set(data)

            try:
                next(iterator)
            except StopIteration:
                actual_length = len(tmp_head)
                if actual_length < length:
                    tmp_data = tmp_data[:actual_length]
            else:
                raise ValueError('length is shorter than iter')

            self.header = tmp_head
            self.data   = tmp_data

    def from_files(self,files,hdu_index=0,**kwargs):
        '''
        Load headers and data from FITS files

        Parameters
        ----------
        files : array-like
            A sequence of FITS file path
            Whether path-like objects are supported depends on
            the version of Python and Astropy.
        hdu_index : int, default 0
            Index of target HDU in HDUList
        kwargs : keyword arguments
            Additional keyword arguments given to from_iterator

        See Also
        --------
        from_iterator : Load headers and data from iterator
        '''
        self.list = files

        iterable = (
            fitsloader(x,dtype=self.dtype,xp=np,hdu_index=hdu_index)
                for x in self.list
        )

        self.from_iterator(iterable,length=len(self.list),**kwargs)

    def from_hduls(self,hduls,hdu_index=0,**kwargs):
        '''
        Load headers and data from a sequence of HDUList

        Parameters
        ----------
        hduls : array-like
            A sequence of HDUList
        hdu_index : int, default 0
            Index of target HDU in HDUList
        kwargs : keyword arguments
            Additional keyword arguments given to from_iterator

        See Also
        --------
        from_iterator : Load headers and data from iterator
        '''
        hduls = builtins.list(hduls)
        if not all(isinstance(hdul,fits.HDUList) for hdul in hduls):
            raise TypeError('Input must be a sequence of HDUList')

        self.list = [hdul.filename() for hdul in hduls]

        iterable = (
            hdu_splitter(x[hdu_index],dtype=self.dtype,xp=np) for x in hduls
        )

        self.from_iterator(iterable,length=len(hduls),**kwargs)

    def from_hdus(self,hdus,**kwargs):
        '''
        Load headers and data from a sequence of HDU

        Parameters
        ----------
        hdus : array-like
            A sequence of HDU
        kwargs : keyword arguments
            Additional keyword arguments given to from_iterator

        See Also
        --------
        from_iterator : Load headers and data from iterator
        '''
        hdus = builtins.list(hdus)

        nums = len(hdus)    
        self.list = [None] * nums

        iterable = (
            hdu_splitter(hdu,dtype=self.dtype,xp=np) for hdu in hdus
        )

        self.from_iterator(iterable,length=nums,**kwargs)

    def write(self,outlist,mapping=None,wrapper=None,**kwargs):
        '''
        make FITS file with storing FITS header and data

        Parameters
        ----------
        outlist : array-like
            list of output FITS paths
            raise ValueError if the length of this list is different
            from the number of images.
        kwargs : keyward arguments
            These are given to writeto method of HDU object
        '''
        if mapping is None:
            mapping = null2
        if wrapper is None:
            wrapper = null1

        outlist = builtins.list(outlist)
        iterator = wrapper(
            mapping(*args) for args in zip(self.header,self.data)
        )
        if len(outlist) != len(self):
            msg = 'the length of outlist differs from the number of images'
            raise ValueError(msg)
        for (head, data), o in zip(iterator,outlist):
            hdu = mkhdu(data,header=head)
            hdu.writeto(o,**kwargs)

def hdu_splitter(hdu,dtype=None,xp=cp):
    '''
    Split HDU object into header and data

    Parameters
    ----------
    hdu : astropy HDU object
    dtype : str or dtype, default None
        dtype of array used internally
        If None, use eclair.common.default_dtype.
        If the input dtype is different, use a casted copy.
    xp : module object of numpy or cupy, default cupy
        Whether the return value is numpy.ndarray or cupy.ndarray
    
    Returns
    -------
    header : astropy.io.fits.Header
    data : ndarray
    '''
    header = hdu.header
    data = xp.asarray(hdu.data,dtype=judge_dtype(dtype))

    return header, data

def mkhdu(data,header=None,hdu_class=fits.PrimaryHDU):
    '''
    Creates HDU from given data

    Parameters
    ----------
    data : ndarray
    header : astropy.io.fits.Header, default None
        The header associated with data.
        If None, a header of the appropriate type is created
        for the supplied data.
    hdu_class : class, default astropy.io.fits.PrimaryHDU
        HDU class

    Returns
    -------
    hdu : HDU object

    Notes
    -----
    This function is intended for use inside eclair.
    It may be better to call astropy HDU class dilectly.
    '''
    hdu = hdu_class(data=cp.asnumpy(data))
    now = Time.now().isot

    hdu.header['ORIGIN'] = origin
    hdu.header['DATE']   = (now,'Date HDU was created')
    if header is not None:
        hdu.header.extend(header)
        
    return hdu

def fitsloader(name,hdu_index=0,**kwargs):
    '''
    Read FITS file and get header and data.

    Parameters
    ----------
    name : str
        path of FITS file
        Whether path-like objects are supported depends on
        the version of Python and Astropy.
    hdu_index : int, default 0
        Index in HDUList of HDU
    kwargs : keyward arguments
        Additional keyword arguments to pass to the hdu_splitter function
        See also hdu_splitter.

    Returns
    -------
    header : astropy.io.fits.Header
    data : ndarray
    '''
    with fits.open(name) as hdul:
        result = hdu_splitter(hdul[hdu_index],**kwargs)
        
    return result