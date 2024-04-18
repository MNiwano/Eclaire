# -*- coding: utf-8 -*-

import sys
import io
import copy
import time
import itertools
import warnings
from pathlib import Path

if sys.version_info.major == 2:
    from future_builtins import zip, map
    from collections     import Sized, Iterable, Iterator, Sequence
    import __builtin__ as builtins
else:
    from collections.abc import Sized, Iterable, Iterator, Sequence
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
    try:
        header = hdu.header
        data = hdu.data
    except AttributeError as e:
        raise TypeError('input must be the HDU object') from e
    else:
        data = xp.asarray(data,dtype=judge_dtype(dtype))

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
    This function is intended for use inside eclaire.
    It may be better to call astropy HDU class dilectly.
    '''
    hdu = hdu_class(data=cp.asnumpy(data))
    now = Time.now().isot

    hdu.header['ORIGIN'] = origin
    hdu.header['DATE']   = (now,'Date HDU was created')
    if header is not None:
        for key in ('ORIGIN','DATE'):
            header.remove(key,ignore_missing=True)
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

class FitsContainer(object):
    '''
    Class for storing multiple FITS data and performing SIMD processing.

    Attributes
    ----------
    list : list
        List of FITS name
    header : list
        List of FITS header
    data : 3D cupy.ndarray
        Array in which image data is stacked

    Notes
    -----
    Since the above attributes are implemented as properties,
    they do not support the inplace operator (e.g. +=).
    If you want to perform such operations, assign the attribute to a variable,
    and perform the operation on the variable.
    '''

    def __init__(self,object,dtype=None,method='files',**kwargs):
        '''
        Parameters
        ----------
        object : type required by the method
            An object which has image data in some way.
            If this is an instance of FitsContainer or its subclass,
            the resulting instance will be a copy of this.
            Otherwise, this will be given as an argument to the method
            specified by the keyword argument "method",
            and used to initialize the attributes "list", "header" and "data".
        method : str, default 'files'
            This specifies how to initialize the instance.
            * 'empty' - return an empty instance. "object" is ignored.
            * 'array' - use self.from_array. "object" is interpteted
                as an array in which images are stacked.
            * 'files' - use self.from_files. "object" is interpreted 
                as a sequence of FITS file paths.
            * 'hduls' - use self.from_hduls. "object" is interpreted
                as a sequence of astropy.io.fits.HDUList.
            * 'hdus' - use self.from_hdus. "object" is interpreted
                as a sequence of astropy HDU objects.
            * 'iterator' - use self.from_iterator. "object" is interpreted
                as an iterator which returns tuples of the header and array.
        dtype : str or dtype, default None
            dtype of ndarray.
            If None, use eclair.common.default_dtype.
        kwargs : keyword argments
            Additional keyword arguments passed to the method
            specified by "method".

        See Also
        --------
        from_array : Set array to the attribute
        from_files : Load headers and data from FITS files
        from_hduls : Load headers and data from a sequence of HDUList
        from_hdus : Load headers and data from a sequence of HDU
        from_iterator : Load headers and data from iterator
        '''
        self.dtype = judge_dtype(dtype)

        attributes = ('list','header','data')

        if isinstance(object,FitsContainer):
            for attr_name in attributes:
                setattr(self,getattr(object,attr_name))
        elif method == 'empty':
            for attr_name in attributes:
                setattr(self,[])
        else:
            methods = {
                'array': self.from_array,
                'hduls': self.from_hduls,
                'hdus': self.from_hdus,
                'files': self.from_files,
                'iterator': self.from_iterator,
            }
            try:
                callback = methods[method]
            except KeyError:
                raise ValueError('invalid method specification')
            else:
                callback(object,**kwargs)

    @property
    def list(self):
        '''
        List of FITS file name
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
        Array in which image data is stacked
        '''
        return self.__data

    @data.setter
    def data(self,array):
        array = cp.array(array,ndmin=3,copy=False,dtype=self.dtype)
        try:
            assert array.ndim == 3
        except AssertionError as e:
            raise ValueError('shape of given array is invalid') from e
        self.__data = array

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
            indices = [i for i in idx]
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
        n_l = len(self.__list)
        n_h = len(self.__header)
        n_d = len(self.__data)
        if (n_d != n_l) or (n_d != n_h):
            warnings.warn('length of list or header differs from length of data')

        return n_d

    def clip(self,indices,in_place=False,remove=False):
        '''
        Removes the items of list, header, and data
        that do not have index in the given indices.

        Parameters
        ----------
        indices : array-like
            indices of items to leave
        in_place : bool, default False
            If True, perform in-place array manipulations for ndarray.
            This saves memory, but the values of the original ndarray
            are not preserved.
        '''
        if remove:
            indices = sorted(
                set(range(len(self))) - set(indices)
            )
        else:
            indices = sorted(indices)

        list = self.list
        head = self.header
        data = self.data

        self.list   = (list[i] for i in indices)
        self.header = (head[i] for i in indices)

        tmpd = data[:len(indices)]
        if not in_place:
            tmpd = cp.empty_like(tmpd)
        for dst,j in zip(tmpd,indices):
            cp.copyto(dst,data[j])
        self.data = tmpd

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

        self.list += list(
            itertools.chain(
                *(arg.list for arg in args)
            )
        )
        self.header += list(
            itertools.chain(
                *(arg.header for arg in args)
            )
        )

        data_to_add = [cp.asarray(arg.data,dtype=self.dtype) for arg in args]
        try:
            self.data = cp.concatenate([self.data] + data_to_add, axis=0)
        except ValueError as e:
            raise ValueError('data shape mismatch') from e

        return self

    def from_array(self,data,list=None,header=None):
        '''
        Set array to the attribute

        Parameters
        ----------
        data : ndarray
            Array in which image data are stacked.
        list : array-like, default None
            Sequence of FITS file name.
            If None, use a list of None as many as the length of array.
        header : array-like, default None
            Sequence of FITS header.
            If None, use a list of None as many as the length of array.
        '''
        data = cp.asarray(data,dtype=self.dtype)
        length = len(data)

        if list is None:
            list = [None] * length
        elif len(list) != length:
            raise ValueError('length of list and shape of data do not match')
        if header is None:
            header = [None] * length
        elif len(header) != length:
            raise ValueError('length of header and shape of data do not match')
        
        self.list   = list
        self.header = header
        self.data   = data

    def from_iterator(self,iterable,mapping=None,wrapper=None,mempool=None):
        '''
        Load headers and data from iterator

        Parameters
        ----------
        iterable : iterable
            An iterable object that returns header and data
            This must return a tuple (header, data).
            'header' and 'data' must be instances of astropy.io.fits.Header
            and 2d ndarray, respectively.
        mapping : callable, default None
            Apply this to each items that 'iterable' returns.
            If None, use a function that returns arguments as is.
        wrapper : callable, default None
            A wrapper function of iterator.
            If None, use a function that returns arguments as is.
        mempool : cupy.cuda.MemoryPool, default None
            This is used for to release the extra memory allocated
            in the process of transferring images to VRAM.
            If you use the custom memory pool (e.g. unified memory pool),
            giving the MemoryPool instance is recommended for
            the efficient memory usage. If None, use the default
            memory pool of CuPy.

        Notes
        -----
        This method iterates the return values of the following expressions.
        ```
        wrapper(mapping(*args) for args in iterable)
        ```
        '''
        if mapping is None:
            mapping = null2
        if wrapper is None:
            wrapper = null1

        if mempool is None:
            mempool = cp.cuda.get_allocator().__self__
        elif not isinstance(mempool,cp.cuda.MemoryPool):
            raise TypeError('mempool must be cupy.cuda.MemoryPool')

        ini_total = mempool.total_bytes()

        tmp = tuple(
            (h, cp.asarray(d,dtype=self.dtype))
            for h, d in wrapper(mapping(*args) for args in iterable)
        )
        tmp_head, tmp_data = zip(*tmp)

        self.header = tmp_head
        self.data = cp.stack(tmp_data)

        del tmp, tmp_data

        if ini_total < mempool.total_bytes():
            mempool.free_all_blocks()
    
    def from_files(self,files,hdu_index=0,**kwargs):
        '''
        Load headers and data from FITS files

        Parameters
        ----------
        files : array-like
            A sequence of FITS file path or file-object.
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
        self.list = (str(f) for f in files)

        iterable = (
            fitsloader(x,dtype=self.dtype,xp=np,hdu_index=hdu_index)
                for x in files
        )

        self.from_iterator(iterable,**kwargs)

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

        self.list = (hdul.filename() for hdul in hduls)

        iterable = (
            hdu_splitter(x[hdu_index],dtype=self.dtype,xp=np) for x in hduls
        )

        self.from_iterator(iterable,**kwargs)

    def from_hdus(self,hdus,list=None,**kwargs):
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
        if list is None:
            self.list = [None] * nums
        elif len(list) == nums:
            self.list = list
        else:
            raise ValueError('length of list and hdus do not match')

        iterable = (
            hdu_splitter(hdu,dtype=self.dtype,xp=np) for hdu in hdus
        )

        self.from_iterator(iterable,**kwargs)

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