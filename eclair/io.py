# -*- coding: utf-8 -*-

import sys
from os.path import isfile

if sys.version_info.major == 2:
    from itertools import izip as zip, imap as map

from astropy.io   import fits
from astropy.time import Time
import cupy  as cp
import numpy as np

from param import __version__, __update__, dtype, null

origin = (
    'Eclair v{version} {date}'.format(version=__version__,date=__update__),
    'FITS file originator'
)

_list = list

class FitsContainer:
    '''
    Class for storing multiple FITS data and performing SIMD processing.

    Attributes
    ----------
    list : list
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

        self.slices = dict(x_start=0,x_stop=None,y_start=0,y_stop=None)
        
        self.header = [None] * len(list)
        self.data   = self.header[:]

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

    def __stack(self,iterator,length,func=null,args=()):
        enumerated = enumerate(iterator)
        
        try:
            i, (head, data) = next(enumerated)

            y_len, x_len = data.shape
            array = cp.empty([length,y_len,x_len],dtype=self.dtype)

            self.header[i] = head
            array[i].set(data)
            func(i,*args)
            for i, (head, data) in enumerated:
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

        iterator = map(
            lambda x:fitsloader(x,dtype=self.dtype,xp=np,**self.slices),
            self.list
        )
        self.__stack(iterator,len(self.list),**kwargs)

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

    @classmethod
    def from_array(cls,array,list=None,headers=None,dtype=dtype):
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
        dtype : str or dtype, default 'float32'
            dtype of cupy.ndarray

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

        instance.data = cp.asarray(array,dtype=dtype)

        return instance

    @classmethod
    def from_hduls(cls,hduls,hdu_index=0,dtype=dtype,**slices):
        '''
        Make instance from sequence of HDULists

        Parameters
        ----------
        hduls : sequence of astropy.io.fits.HDUList
        hdu_index : int, default 0
            Index in HDUList of HDU
        dtype : str or dtype, default 'float32'
            dtype of cupy.ndarray
        slices : keyward arguments
            Arguments given to the setslice method
        
        Returns
        -------
        instance : FitsContainer
        '''
        instance = cls([hdul.filename() for hdu in hduls],dtype=dtype)
        instance.setslice(**slices)

        iterator = map(
            lambda hdul:split_hdu(hdul[hdu_index],dtype=dtype,xp=np,**slices),
            hduls
        )
        iterator.__stack(iterator,len(hduls))

        return instance

    @classmethod
    def from_hdus(cls,hdus,list=None,dtype=dtype,**slices):
        '''
        Make instance from sequence of HDUs

        Parameters
        ----------
        hdus : sequence of astropy HDU objects
        list : array-like
            sequence of FITS file names
        dtype : str or dtype, default 'float32'
            dtype of cupy.ndarray
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

        iterator = map(
            lambda hdu:split_hdu(hdu,dtype=dtype,xp=np,**slices),hdus
        )
        iterator.__stack(iterator,nums)

        return instance

def split_hdu(hdu,dtype=dtype,xp=cp,
        x_start=0,x_stop=None,y_start=0,y_stop=None):
    '''
    Split HDU object into header and data

    Parameters
    ----------
    hdu : astropy HDU object
    dtype : str or dtype, default 'float32'
        dtype of array returned
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
    data = xp.asarray(hdu.data[y_start:y_stop,x_start:x_stop],dtype=dtype)

    try:
        header['CRPIX1'] -= x_start
        header['CRPIX2'] -= y_start
    except:
        pass

    return header, data

def mkhdu(data,header=None,is_primary=True):
    '''
    Creates HDU from given data

    Parameters
    ----------
    data : ndarray
    header : astropy.io.fits.Header, default None
        data and header to HDU
    is_primary : bool, default True
        If True, create as PrimaryHDU.

    Returns
    -------
    hdu : HDU object

    Notes
    -----
    This function is intended for use inside eclair.
    It may be better to call astropy.io.fits.PrimaryHDU etc.
    directly for an end-user.
    '''
    if is_primary:
        HDUclass = fits.PrimaryHDU
    else:
        HDUclass = fits.ImageHDU

    hdu = HDUclass(data=cp.asnumpy(data))
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

def fitswriter(name,data,header=None,overwrite=False):
    '''
    Create FITS from given array

    Parameters
    ----------
    name : str
        path of FITS file
        Whether path-like objects are supported depends on
        the version of Python and Astropy.
    data : ndarray
    header : astropy.io.fits.Header, default None
        data and header to FITS
    overwrite : bool default False
        If True, overwrite the output file if it exists.
        Raises an IOError if False and the output file exists.

    Notes
    -----
    This function is intended for use inside eclair.
    It may be better to use astropy.io.fits.writeto etc. for an end-user.
    '''
    hdu = mkhdu(data,header=header,is_primary=True)

    hdu.writeto(name,overwrite=overwrite)