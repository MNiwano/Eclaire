# -*- coding: utf-8 -*-

import sys
from os.path   import isfile

if sys.version_info.major == 2:
    from future_builtins import zip, map
    import __builtin__ as builtins
else:
    import builtins

from astropy.io   import fits
from astropy.time import Time
import cupy  as cp
import numpy as np

from .common import __version__, __update__, null, judge_dtype

origin = (
    'Eclair v{version} {date}'.format(version=__version__,date=__update__),
    'FITS file originator'
)

class FitsContainer:
    '''
    Class for storing multiple FITS data and performing SIMD processing.

    Attributes
    ----------
    list : list
        List of FITS file paths
    header : list
        List of FITS header
        Before calling load method, this is a list of None.
    data : 3D cupy.ndarray
        array of image data stacked along 1st axis
        Before calling load method, this is a list of None.
    shape : tuple
        Shape of image data
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

    def __init__(self,list,dtype=None):
        '''
        Parameters
        ----------
        list : iterable
            List of FITS file paths
            load() method refer this list to load FITS file.
            Whether path like object is supported depends on 
            the version of Python and Astropy.
        dtype : str or dtype, default None
            dtype of cupy.ndarray
            If None, this value will be usually "float32", 
            but this can be changed with eclair.set_dtype.
        '''
        self.list  = builtins.list(list)
        self.dtype = judge_dtype(dtype)

        self.slices = dict(x_start=0,x_stop=None,y_start=0,y_stop=None)
        
        self.header = [None] * len(list)
        self.data   = [None] * len(list)

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

    def setslice(self,x_start=0,x_stop=None,y_start=0,y_stop=None):
        '''
        Set data slice range

        Parameters
        ----------
        x_start : int, default 0
        x_stop : int, default None
        y_start : int, default 0
        y_stop : int, default None
        '''
        self.slices = dict(
            x_start=x_start,x_stop=x_stop,y_start=y_start,y_stop=y_stop
        )

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

    def __stack(self,iterable,length,func=null,args=()):
        iterator = enumerate(iterable)
        
        try:
            i, (head, data) = next(iterator)

            y_len, x_len = data.shape
            array = cp.empty([length,y_len,x_len],dtype=self.dtype)

            self.header[i] = head
            array[i].set(data)
            func(i,*args)
            for i, (head, data) in iterator:
                self.header[i] = head
                array[i].set(data)
                func(i,*args)
            self.data = array
        except StopIteration:
            raise ValueError('iterable is empty')

    def load(self,hdu_index=0,check_exists=False,**kwargs):
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
        It cannot detect the file that exists but has invalid contents.
        '''
        if check_exists:
            self.list = filter(isfile,self.list)

        iterable = map(
            lambda x:fitsloader(x,dtype=self.dtype,xp=np,**self.slices),
            self.list
        )
        self.__stack(iterable,len(self.list),**kwargs)

    def concatenate(self,*args):
        '''
        Add containsts of the other instances to self.

        Parameters
        ----------
        *args : FitsContainer or its subclass
            instances which merges with self
        '''

        for arg in args:
            if not isinstance(arg,FitsContainer):
                raise TypeError(
                    'Only support FitsContainer or its subclasses'
                )

        try:
            self.data = cp.concatenate(
                [self.data] + [arg.data for arg in args],
                axis=0
            )
        except ValueError:
            raise ValueError('image shape mismatch')

        self.list = sum(
            (arg.list for arg in args),
            start = self.list
        )
        self.header = sum(
            (arg.header for arg in args),
            start = self.header
        )

        return self

    def write(self,outlist,func=null,args=(),**kwargs):
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
        if len(outlist) != len(self):
            msg = 'the length of outlist differs from the number of images'
            raise ValueError(msg)
        for i,(o,head,data) in enumerate(zip(outlist,self.header,self.data)):
            hdu = mkhdu(data,header=head)
            hdu.writeto(o,**kwargs)
            func(i,*args)

    @classmethod
    def from_array(cls,array,list=None,headers=None,dtype=None):
        '''
        Make instance from given array

        Parameters
        ----------
        array : array-like
            Array of image data stacked
        list : array-like, default None
            List of FITS file name
        headers : array-like, default None
            List of FITS file header
        dtype : str or dtype, default None
            dtype of cupy.ndarray
            See also __init__.

        Returns
        -------
        instance : FitsContainer
        '''
        nums = len(array)
        if list is None:
            list = [None] * nums
        elif len(list) != nums:
            raise ValueError('list does not match array')

        instance = cls(list,dtype=dtype)
        
        if headers is not None:
            if len(headers) == nums:
                instance.header = headers
            else:
                raise ValueError('headers does not match array')

        instance.data = cp.asarray(array,dtype=instance.dtype)

        return instance

    @classmethod
    def from_hduls(cls,hduls,hdu_index=0,dtype=None,**slices):
        '''
        Make instance from sequence of HDULists

        Parameters
        ----------
        hduls : sequence of HDUList
        hdu_index : int, default 0
            Index of HDU in HDUList
        dtype : str or dtype, default None
            dtype of cupy.ndarray
            See also __init__.
        slices : keyward arguments
            Arguments given to the setslice method
        
        Returns
        -------
        instance : FitsContainer
        '''
        instance = cls([hdul.filename() for hdul in hduls],dtype=dtype)
        instance.setslice(**slices)

        iterable = map(
            lambda hdul:split_hdu(hdul[hdu_index],dtype=dtype,xp=np,**slices),
            hduls
        )
        instance.__stack(iterable,len(hduls))

        return instance

    @classmethod
    def from_hdus(cls,hdus,list=None,dtype=None,**slices):
        '''
        Make instance from sequence of HDUs

        Parameters
        ----------
        hdus : sequence of astropy HDU objects
        list : array-like
            sequence of FITS file names
        dtype : str or dtype, default None
            dtype of cupy.ndarray
            See also __init__.
        slices : keyward arguments
            Arguments given to the setslice method

        Returns
        -------
        instance : FitsContainer
        '''
        nums = len(hdus)
        if list is None:
            list = [None] * nums
        elif len(list) != nums:
            raise ValueError('list does not match hdus')

        instance = cls(list,dtype=dtype)
        instance.setslice(**slices)

        iterable = map(
            lambda hdu:split_hdu(hdu,dtype=dtype,xp=np,**slices),
            hdus
        )
        instance.__stack(iterable,nums)

        return instance

def split_hdu(hdu,dtype=None,xp=cp,
        x_start=0,x_stop=None,y_start=0,y_stop=None):
    '''
    Split HDU object into header and data

    Parameters
    ----------
    hdu : astropy HDU object
    dtype : str or dtype, default None
        dtype of cupy.ndarray
        If None, this value will be usually "float32", 
        but this can be changed with eclair.set_dtype.
    xp : module object of numpy or cp, default cupy
        Whether the return value is numpy.ndarray or cupy.ndarray
    x_start : int, default 0
    x_stop : int, default None
    y_start : int, default 0
    x_stop : int, default None
        return the area of [y_start: y_stop, x_start: x_stop] in hdu.data
    
    Returns
    -------
    header : astropy.io.fits.Header
    data : ndarray
    '''

    header = hdu.header
    data = xp.asarray(
        hdu.data[y_start:y_stop,x_start:x_stop],
        dtype=judge_dtype(dtype)
    )

    try:
        header['CRPIX1'] -= x_start
        header['CRPIX2'] -= y_start
    except:
        pass

    return header, data

def mkhdu(data,header=None,hdu_class=fits.PrimaryHDU):
    '''
    Creates HDU from given data

    Parameters
    ----------
    data : ndarray
    header : astropy.io.fits.Header, default None
        data and header to HDU
    hdu_class : class, default astropy.io.fits.PrimaryHDU
        HDU class

    Returns
    -------
    hdu : HDU object

    Notes
    -----
    This function is intended for use inside eclair.
    It may be better to call astropy HDU class dilectly for an end-user.
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
        Additional keyword arguments to pass to the split_hdu function
        See also split_hdu.

    Returns
    -------
    header : astropy.io.fits.Header
    data : ndarray
    '''
    with fits.open(name) as hdul:
        result = split_hdu(hdul[hdu_index],**kwargs)
        
    return result
