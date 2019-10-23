# -*- coding: utf-8 -*-

from copy       import copy
import time

from astropy.io import fits
import cupy     as cp

from param import dtype, origin

#############################
#   FitsContainer
#############################

class FitsContainer:
    '''
    Class for storing FITS name, header, data

    Attributes
    ----------
    list : array-like
        FITS name list
    header : dict, default None
        dict of FITS name and header
        Before calling load() medthod, it is None.
    data : 3-dimension cupy.ndarray, default None
        array of image data stacked along 1st axis
        Before calling load() medthod, it is None.
    slice : tupple of slice object, default (slice(0,None),slice(0,None))
        load() method refers to this value and slice the data.
        1st item is slice along Y, 2nd is along X.
        (i.e. If slice = (slice(ymin,ymax),slice(xmin,xmax)),
        load() method gets the range:[ymin:ymax,xmin:xmax] of FITS data)
        See also setslice() method.
    dtype : str or dtype (NumPy or CuPy), default 'float32'
        dtype of cupy.ndarray created

    Notes
    -----
    The order of items of list and data(along 1st axis) must be same.
    So, be careful when you edit these attributes.
    '''

    def __init__(self,list,dtype=dtype):
        '''
        Parameters
        ----------
        list : array-like
            FITS name list
            load() method refer this list to load FITS contents.
        '''
        self.list  = list
        self.dtype = dtype

        self.slice  = (slice(0,None),slice(0,None))
        self.header = None
        self.data   = None

    def __getitem__(self,key):
        '''
        self.__getitem__(key) <==> self[key]

        Parameters
        ----------
        key : str
            FITS name you want to access
        
        Returns
        -------
        header : astropy.io.fits.header
            FITS header
        data : 2-dimension cupy.ndarray
            FITS data 
        '''
        idx = self.list.index(key)
        return self.header[key], self.data[idx,:,:]

    def setslice(self,xmin=0,xmax=None,ymin=0,ymax=None):
        '''
        Set data slice range
        This method is equal to
        self.slice = (slice(ymin,ymax),slice(xmin,xmax)).

        Parameters
        ----------
        xmin : int, default 0
            minimum index along X
        xmax : int, default None
            maximum index along X plus 1
        ymin : int, default 0
            minimum index along Y
        ymax : int, default None
            maximum index along Y plus 1
        '''
        self.slice = (slice(ymin,ymax),slice(xmin,xmax))

    def load(self,progress=lambda *args:None,args=()):
        '''
        Load FITS header and data with refering list

        Parameters
        ----------
        progress : callable object, default lambda *args:None
            Function to be executed simultaneously with data reading
            and given arguments (i, *args), where i is index of FITS
            If you want to do something simultaneously with reading,
            input as a function. (e.g. logging, showing progress bar, etc)
        args : tupple, default ()
            argments given additionally to the progress function
        '''
        n = len(self.list)
        if n:
            head, data = self.__fitsopen(self.list[0])
            y_len, x_len = data.shape

            self.header = {self.list[0]: head}
            self.data   = cp.empty([n,y_len,x_len],dtype=self.dtype)
            self.data[0,:,:] = data
            progress(0,*args)
            for i,f in enumerate(self.list[1:],1):
                head, data = self.__fitsopen(f)
                self.header[f]   = head
                self.data[i,:,:] = data
                progress(i,*args)
        else:
            raise ValueError('self.list is empty')

    def __fitsopen(self,f):
        with fits.open(f) as img:
            head = img[0].header
            try:
                head['CRPIX1'] -= self.slice[1].start
                head['CRPIX2'] -= self.slice[0].start
            except KeyError:
                pass
            data = cp.asrray(img[0].data[self.slice],dtype=self.dtype)
        return head, data

    def write(self,outlist,overwrite=False):
        '''
        make FITS file with storing FITS header and data

        Parameters
        ----------
        outlist : array-like
            list of output FITS names
        overwrite : bool, default False
            If True, overwrite the output file if it exists.
            Raises an IOError if False and the output file exists.
        '''
        for f,o,data in zip(self.list,outlist,self.data):
            now_ut = time.strftime('%Y/%m/%dT%H:%M:%S',time.gmtime())

            hdu = fits.PrimaryHDU(
                data=data.get(),
                header=self.header[f].copy()
            )
            hdu.header.insert(6,('DATE',now_ut,'Date FITS file was generated'))
            hdu.header.append(origin)
            hdu.writeto(o,overwrite=overwrite)
