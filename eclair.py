# -*- coding: utf-8 -*-
'''
Eclair
======

Eclair: CUDA-based Library for Astronomical Image Reduction

This module provides some useful classes and functions
in astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.

This module requires
    1. NVIDIA GPU
    2. CUDA
    3. NumPy, Astropy and CuPy
'''

from itertools  import product
from os.path    import basename
import time

from astropy.io import fits
import numpy    as np
import cupy     as cp

__version__ = '0.5'
__update__  = '14 June 2019'

_origin = ('ORIGIN','Eclair v%s %s'%(__version__, __update__),
           'FITS file originator')

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

    Notes
    -----
    The order of items of list and data(along 1st axis) must be same.
    So, be careful when you edit these attributes.
    '''

    def __init__(self,list):
        '''
        Parameters
        ----------
        list : array-like
            FITS name list
            load() method refer this list to load FITS contents.
        '''
        self.list  = list
        self.slice = (slice(0,None),slice(0,None))

        self.header = None
        self.data   = None

    def __getitem__(self,key):
        '''
        You can access the items of instances like dict. (i.e. self[key])

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
            self.data   = cp.empty([n,y_len,x_len],dtype='f4')
            self.data[0,:,:] = data
            progress(0,*args)
            for i,f in enumerate(self.list[1:],1):
                head, data = self.__fitsopen(f)
                self.header[f]   = head
                self.data[i,:,:] = data
                progress(i,*args)
        else:
            raise ValueError('No FITS name in self.list')

    def __fitsopen(self,f):
        with fits.open(f) as img:
            head = img[0].header
            try:
                head['CRPIX1'] -= self.slice[1].start
                head['CRPIX2'] -= self.slice[0].start
            except KeyError:
                pass

            data = cp.array(img[0].data[self.slice].astype('f4'))
        return head, data

    def write(self,outlist,overwrite=False):
        '''
        make FITS file with storing FITS header and data

        Parameters
        ----------
        outlist : array-like
            list of output FITS names
        overwrite : bool, default Flase
            If True, overwrite the output file if it exists.
            Raises an IOError if False and the output file exists.
        '''
        for f,o,data in zip(self.list,outlist,self.data):
            now_ut = time.strftime('%Y/%m/%dT%H:%M:%S',time.gmtime())

            hdu = fits.PrimaryHDU(data.get())
            hdu.header = self.header[f].copy()
            hdu.header.insert(6,('DATE',now_ut,'Date FITS file was generated'))
            hdu.header.append(_origin)
            fits.HDUList(hdu).writeto(o,overwrite=overwrite)

#############################
#   reduction
#############################

def reduction(image,bias,dark,flat):
    '''
    This function is equal to the equation:
    result = (image - bias - dark) / flat, but needs less memory.
    Therefore, each inputs must be broadcastable shape.

    Parameters
    ----------
    image : cupy.ndarray
    bias  : cupy.ndarray
    dark  : cupy.ndarray
    flat  : cupy.ndarray
    
    Returns
    -------
    result : cupy.ndarray
    '''

    return _reduction_kernel(image,bias,dark,flat)

_reduction_kernel = cp.ElementwiseKernel('T x, T b, T d, T f', 'T z',
                                  'z = (x - b - d) / f', '_reduction_kernel')

#############################
#   imalign
#############################

class ImAlign:
    '''
    Generate imalign function
    Instance of this class can be used as function to align images.

    Attributes
    ----------
    x_len, y_len : int
        Shape of images to align
    interp : {'spline3', 'poly3', 'linear'}, default 'spline3'
        Subpixel interpolation algorithm in subpixel image shift
         spline3 - 3rd order spline interpolation
         poly3   - 3rd order polynomial interpolation
         linear  - linear interpolation
    '''
    def __init__(self,x_len,y_len,interp='spline3'):
        '''
        Parameters
        ----------
        x_len, y_len : int
            Shape of images to align
        interp : {'spline3', 'poly3', 'linear'}, default 'spline3'
            Subpixel interpolation algorithm in subpixel image shift
             spline3 - 3rd order spline interpolation
             poly3   - 3rd order polynomial interpolation
             linear  - linear interpolation
        '''
        self.x_len  = x_len
        self.y_len  = y_len
        self.interp = interp
        if   interp == 'spline3':
            self.shift = self.__spline
            self.mat = {'x':_Ms(x_len)}
            if y_len == x_len:
                self.mat['y'] = self.mat['x'].view()
            else:
                self.mat['y'] = _Ms(y_len)
        elif interp == 'poly3':
            self.shift = self.__poly
            self.mat = _Mp()
        elif interp == 'linear':
            self.shift = self.__linear
        else:
            raise ValueError('"%s" is not inpremented'%interp)
        
    def __call__(self,data,shifts,reject=False,baseidx=None,tolerance=None,
                 selected=None,progress=lambda *args:None,args=()):
        '''
        Stack the images with aligning their relative positions,
        and cut out the overstretched area

        Parameters
        ----------
        data : 3-dimension cupy.ndarray
            An array of images stacked along the 1st axis
            If the shape of image is differ from attributes x_len, y_len,
            ValueError is raised.
        shifts : 2-dimension numpy.ndarray
            An array of relative positions of images in units of pixel
            Along the 1st axis, values of each images must be the same order
            as the 1st axis of "data".
            Along the 2nd axis, the 1st item is interpreted as 
            the value of X, the 2nd item as the value of Y.
        reject : bool, default False
            If True, reject too distant image.
            Then, you must input baseidx, tolerance and selected.
        baseidx : int, default None
            Index of base image
            If you set reject True, set also this parameter.
        tolerance : int or float, default None
            Maximum distance from base image, in units of pixel
            If you set reject True, set also this parameter.
        selected : variable referencing empty list, default None
            List for storing indices of selected images
            If you set reject True, set also this parameter.
        progress : function, default lambda *args:None
            Function to be executed simultaneously with aligning
            and given arguments (i, *args), where i is index of image
            If you want to do something simultaneously with aligning,
            input as a function. (e.g. logging, showing progress bar, etc)
        args : tupple
            arguments given additionally to the progress function

        Returns
        -------
        align : 3-dimension cupy.ndarray (dtype float32)
            An array of images aligned and stacked along the 1st axis
        '''
        n_frames, y_len, x_len = data.shape
        if (y_len,x_len) != (self.y_len,self.x_len):
            message = 'shape of images is differ from (%d,%d)'
            raise ValueError(message%(self.y_len,self.x_len))

        x_u, y_u = np.ceil(shifts.max(axis=0)).astype('int')
        x_l, y_l = np.floor(shifts.min(axis=0)).astype('int')
    
        xy_i = np.floor(shifts).astype('int')
        xy_d = shifts - xy_i

        iterator = zip(xy_i,xy_d,data)
        if reject:
            if all(f!=None for f in (baseidx,tolerance,selected)):
                norm  = np.linalg.norm(shifts-shifts[baseidx,:],axis=1)
                flags = (norm <= tolerance)
                n_frames = flags.sum()
                selected += list(np.where(flags)[0])
                iterator = _compress(iterator,flags)
            else:
                raise ValueError('baseidx or tolerance or selected is invalid')

        align = cp.zeros([n_frames,y_u-y_l+y_len-1,x_u-x_l+x_len-1],dtype='f4')
        for i,((ix,iy),(dx,dy),layer) in enumerate(iterator):
            align[i, iy-y_l+1 : iy-y_l+y_len, ix-x_l+1 : ix-x_l+x_len]\
                = self.shift(layer,dx,dy)
            progress(i,*args)

        return align[:, y_u-y_l : y_len, x_u-x_l : x_len]

    def __linear(self,data,dx,dy):
        stack = cp.empty([4,self.y_len-1,self.x_len-1],dtype='f4')

        stack[0,:,:] = dx     * dy     * data[ :-1, :-1]
        stack[1,:,:] = dx     * (1-dy) * data[1:  , :-1]
        stack[2,:,:] = (1-dx) * dy     * data[ :-1,1:  ]
        stack[3,:,:] = (1-dx) * (1-dy) * data[1:  ,1:  ]

        return stack.sum(axis=0)
    
    def __poly(self,data,dx,dy):
        x_len = self.x_len-3
        y_len = self.y_len-3

        shifted = self.__linear(data,dx,dy)

        shift_vector = cp.empty([16],dtype='f4')
        stack = cp.empty([16,y_len,x_len],dtype='f4')
        for i, j in product(range(4),repeat=2):
            shift_vector[i*4+j] = (1-dx)**(3-i) * (1-dy)**(3-j)
            stack[i*4+j,:,:]  = data[i:i+y_len,j:j+x_len]

        tmpmat = cp.dot(shift_vector,self.mat)
        shifted[1:-1,1:-1] = cp.tensordot(tmpmat,stack,1)

        return shifted

    def __spline(self,data,dx,dy):
        tmpd = self.__spline1d(data.T,dx,'x')
        return self.__spline1d(tmpd.T,dy,'y')

    def __spline1d(self,data,d,axis):
        v = data[2:,:]+data[:-2,:]-2*data[1:-1,:]
        u = cp.zeros_like(data)
        u[1:-1,:] = cp.dot(self.mat[axis],v)
    
        return _spline(u[1:,:],u[:-1,:],data[1:,:],data[:-1,:],d)

def imalign(data,shifts,interp='spline3',reject=False,baseidx=None,
            tolerance=None,selected=None):
    '''
    Stack the images with aligning their relative positions,
    and cut out the overstretched area
    This function uses class eclair.ImAlign internally.

    Refer to eclair.ImAlign for documentation of parameters and return.

    See also
    --------
    eclair.ImAlign : Class to generate imalign function
    '''
    y_len, x_len = data.shape[1:]
    func = ImAlign(x_len=x_len,y_len=y_len,interp=interp)

    return func(data,shifts,baseidx=baseidx,reject=reject,
                tolerance=tolerance,selected=selected)

def _compress(data, selectors):
    for d, s in zip(data,selectors):
        if s:
            yield d

def _Mp():
    Mp = np.empty([16,16],dtype='f4')
    for y,x,k,l in product(range(4),repeat=4):
        Mp[y*4+x,k*4+l] = (x-1)**(3-k) * (y-1)**(3-l)
    Mp = cp.linalg.inv(cp.array(Mp))

    return Mp

def _Ms(ax_len):
    Ms = 4 * np.identity(ax_len-2,dtype='f4')
    Ms[1:  , :-1] += np.identity(ax_len-3,dtype='f4')
    Ms[ :-1,1:  ] += np.identity(ax_len-3,dtype='f4')
    Ms = cp.linalg.inv(cp.array(Ms))

    return Ms

_spline = cp.ElementwiseKernel('T u, T v, T x, T y, T d','T z',
    'z = (u-v)*(1-d)*(1-d)*(1-d) + 3*v*(1-d)*(1-d) + (x-y-u-2*v)*(1-d) + y',
    '_spline')

#############################
#   imcombine
#############################

def imcombine(data,name,list=None,combine='mean',header=None,iter=3,width=3.0,
              memsave=False,overwrite=False):
    '''
    Calculate sigma-clipped mean or median (no rejection) of images,
    and write to FITS file

    Parameters
    ----------
    data : 3-dimension cupy.ndarray
        An array of images stacked along the 1st axis
    name : str
        A name of output FITS file
    list : array-like, default None
        Names of images combined
        These are written to the header.
    combine : {'mean', 'median'}, default 'mean'
        An algorithm to combine images
        'mean' is sigma-clipped mean, 'median' is median (no rejection).
    header : astropy.io.fits.Header, default None
        A header for output FITS file
    iter : int, default 3
        A number of sigmaclipping iterations
    width : int or float, default 3.0
        A clipping width in sigma units
    memsave : bool, default False
        If True, divide data and calculate it serially.
        Then, VRAM is saved, but speed may be slower.
    overwrite : bool, default False
        If True, overwrite the output file if it exists.
        Raises an IOError if False and the output file exists.
    '''

    if   combine == 'mean':
        func = sigclipped_mean
    elif combine == 'median':
        func = _median
    else:
        raise ValueError('"%s" is not defined as algorithm'%combine)

    kwargs = dict(iter=iter,width=width,axis=0)
    if memsave:
        y_len, x_len = data.shape[1:]
        yhalf, xhalf = int(y_len/2), int(x_len/2)
        combined = cp.empty([y_len,x_len],dtype='f4')
        slices = tuple((slice(l),slice(l,None)) for l in(yhalf,xhalf))
        for yslice, xslice in product(*slices):
            combined[yslice,xslice] = func(data[:,yslice,xslice],**kwargs)
    else:
        combined = func(data,**kwargs)

    now_ut = time.strftime('%Y/%m/%dT%H:%M:%S',time.gmtime())
    hdu    = fits.PrimaryHDU(combined.get())

    if header:
        hdu.header = header.copy()
    hdu.header.insert(6,('DATE',now_ut,'Date FITS file was generated'))
    if list:
        for i,f in enumerate(list,1):
            hdu.header['IMCMB%03d'%i] = basename(f)
        hdu.header['NCOMBINE'] = len(list)
    hdu.header.append(_origin)

    fits.HDUList(hdu).writeto(name,overwrite=overwrite)

    print('Combine: %d frames, Output: %s'%(len(list),name))

def sigclipped_mean(data,iter=3,width=3.0,axis=None):
    '''
    Calculate sigmaclipped mean of given array

    Parameters
    ----------
    data : cupy.ndarray
    iter : int, default 3
        A number of sigmaclipping iterations
    width : int or float, default 3.0
        A clipping width in sigma units
    axis : int or tuple, default None
        Axis or axes along which the means are computed
        If this is a tuple of ints, a mean is performed over multiple axes.

    Returns
    -------
    mean : cupy.ndarray (dtype float32)
        sigmaclipped mean of "data"
    '''
    filt = cp.ones_like(data)
    for _ in range(iter):
        filt = _sigmaclip(data,filt,width,axis)
    mean = _filteredmean(data,filt,axis)

    return mean

def _sigmaclip(data,filt,width,axis):
    mean  = _filteredmean(data,filt,axis)
    sigma = cp.sqrt(_sqm(data,mean,filt,axis=axis)/cp.sum(filt,axis=axis))

    filt  = (width*sigma >= cp.abs(data - mean)).astype('f4')
    return filt

def _filteredmean(data,filt,axis):
    return _sum(data,filt,axis=axis) / cp.sum(filt,axis=axis)

_sum = cp.ReductionKernel('T x, T f','T y','x*f','a+b','y=a','0','_sum')
_sqm = cp.ReductionKernel('T x, T m, T f','T y','(x-m)*(x-m)*f','a+b','y=a',
                          '0','_sqm')

def _median(data,**kwargs):
    length = data.shape[0]
    idx    = int(length/2)
    sort   = cp.sort(data,axis=0)
    if length%2 == 0:
        median = (sort[idx-1,:,:]+sort[idx,:,:]) / 2
    else:
        median = sort[idx,:,:]

    return median

#############################
#   fixpix
#############################

def fixpix(data,mask):
    '''
    fill the bad pixel with mean of surrounding pixels

    Parameters
    ----------
    data : 3-dimension cupy.ndarray
        An array of images stacked along the 1st axis
    mask : 2-dimension cupy.ndarray
        An array indicates bad pixel positions
        The shape of mask must be same as image.
        The value of bad pixel is 1, and the others is 0.

    Returns
    -------
    fixed : 3-dimension cupy.ndarray (dtype float32)
        An array of images fixed bad pixel
    '''

    tmpm  = mask[cp.newaxis,:,:]
    ones  = cp.ones_like(tmpm)
    fixed = data.copy()
    while tmpm.sum():
        filt   = ones - tmpm
        fixed *= filt
        dconv  = _convolve(fixed)
        nconv  = _convolve(filt)
        zeros  = (nconv==0.0).astype('f4')
        fixed += _fix(tmpm, dconv, nconv, zeros)
        tmpm   = zeros

    return fixed

def _convolve(data):
    n_frames, y_len, x_len = data.shape
    conv = cp.zeros([n_frames,2+y_len,2+x_len],dtype='f4')
    for x,y in product(range(3),repeat=2):
        conv[:,y:y+y_len,x:x+x_len] += data

    return conv[:,1:-1,1:-1]

_fix = cp.ElementwiseKernel('T m, T d, T n, T f','T z','z=m*d/(n+f)','_fix')
