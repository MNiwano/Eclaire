'''
Module for handling FITS file with GPU
This module requires Astropy, NumPy, CuPy.
'''

from astropy.io import fits
from itertools  import product
import numpy    as np
import cupy     as cp

__version__ = 0.5
                
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

    bias : cupy.ndarray
    
    dark : cupy.ndarray

    flat : cupy.ndarray
    
    Returns
    ----------
    result : cupy.ndarray
    '''

    return _reduction(image,bias,dark,flat)

_reduction = cp.ElementwiseKernel('T x, T b, T d, T f', 'T z',
                                  'z = (x - b - d) / f', '_reduction_')

#############################
#   imalign
#############################

def imalign(data,shifts,interp='spline3',baseidx=0,
            tolerance=None,rejected=None):
    '''
    Stack the images with aligning their relative positions,
    and cut out the overstretched area
    This function can't rotate image.

    Parameters
    ----------
    data : 3-dimension cupy.ndarray
        An array of images stacked along the 1st axis

    shifts : 2-dimension numpy.ndarray
        An array of relative positions of images in units of pixel
        Along the 1st axis, values of each images must be the same order
        as the 1st axis of "data".
        Along the 2nd axis, the 1st item is interpreted as 
        the value of X, the 2nd item as the value of Y.

    interp : 'spline3' or 'poly3' or 'linear', default 'spline3'
        Subpixel interpolation algorithm in subpixel image shift
         spline3 - 3rd order spline interpolation
         poly3   - 3rd order polynomial interpolation
         linear  - linear interpolation

    baseidx : int, default 0
        Index of base image

    tolerance : int or float, default None
        Maximum distance from base image, in units of pixel
        If you set this argument, also imput rejected.
    
    rejected : list, default None
        List for storing indices of rejected images by tolerance
        If you set tolerance, also imput this argument.

    Returns
    ----------
    align : 3-dimension cupy.ndarray (dtype float32)
        An array of images aligned and stacked along the 1st axis
    '''
    global _Ms, _Mp

    n_frames, y_len, x_len = data.shape

    if   interp == 'spline3':
        _Ms = {'x': _gen_Ms(x_len), 'y': _gen_Ms(y_len)}
    elif interp == 'poly3':
        _Mp = _gen_Mp()
    elif interp != 'linear':
        raise TypeError('"%s" is not defined as algorithm'%interp)

    shift_subpix = _interp[interp]
    
    tmp_shifts = shifts - shifts[baseidx,:]

    x_u, y_u = np.ceil(tmp_shifts.max(axis=0)).astype('int')
    x_l, y_l = np.floor(tmp_shifts.min(axis=0)).astype('int')
    
    xy_i = np.floor(tmp_shifts).astype('int')
    xy_d = shifts - xy_i

    if   tolerance and (type(rejected)==list):
        flags = (np.linalg.norm(shifts,axis=1) <= tolerance).astype('int')
        n_frames = flags.sum()

        rejected += list(np.where(np.ones_like(flags)-flags)[0])

        iterator = enumerate(_compress(zip(xy_i,xy_d,data),flags))
    elif tolerance and (type(rejected)!=list):
        raise ValueError('input list object as rejected')
    else:
        iterator = enumerate(zip(xy_i,xy_d,data))

    align = cp.zeros([n_frames,y_u-y_l+y_len-1,x_u-x_l+x_len-1],dtype='f4')
    for i,((ix,iy),(dx,dy),layer) in iterator:
        align[i, iy-y_l+1 : iy-y_l+y_len, ix-x_l+1 : ix-x_l+x_len]\
            = shift_subpix(layer,dx,dy)

    align = align[:, y_u-y_l : y_len, x_u-x_l : x_len]

    return align

def _compress(data, selectors):
    iter_data = iter(data)
    iter_slct = iter(selectors)
    while True:
        d = next(iter_data)
        s = next(iter_slct)
        if s:
            yield d

def _gen_Mp():
    Mp = np.empty([16,16],dtype='f4')
    for y,x,k,l in product(range(4),repeat=4):
        Mp[y*4+x,k*4+l] = (x-1)**(3-k) * (y-1)**(3-l)
    Mp = cp.linalg.inv(cp.array(Mp))

    return Mp

def _gen_Ms(ax_len):
    Ms = 4 * np.identity(ax_len-2,dtype='f4')
    Ms[1:  , :-1] += np.identity(ax_len-3,dtype='f4')
    Ms[ :-1,1:  ] += np.identity(ax_len-3,dtype='f4')
    Ms = cp.linalg.inv(cp.array(Ms))

    return Ms

def _linear(data,dx,dy):
    y_len, x_len = data.shape

    stack = cp.empty([4,y_len-1,x_len-1],dtype='f4')

    stack[0,:,:] = dx     * dy     * data[ :-1, :-1]
    stack[1,:,:] = dx     * (1-dy) * data[1:  , :-1]
    stack[2,:,:] = (1-dx) * dy     * data[ :-1,1:  ]
    stack[3,:,:] = (1-dx) * (1-dy) * data[1:  ,1:  ]

    shifted = stack.sum(axis=0)

    return shifted

def _poly3(data,dx,dy):
    y_len, x_len = data.shape

    x_len -= 3
    y_len -= 3

    shifted = _linear(data,dx,dy)

    shift_vector = cp.empty([16],dtype='f4')
    stack = cp.empty([16,y_len,x_len],dtype='f4')
    for i, j in product(range(4),repeat=2):
        shift_vector[i*4+j] = (1-dx)**(3-i) * (1-dy)**(3-j)
        stack[i*4+j,:,:]  = data[i:i+y_len,j:j+x_len]

    coeff = cp.tensordot(_Mp,stack,1)
    shifted[1:-1,1:-1] = cp.tensordot(shift_vector,coeff,1)
    
    return shifted

def _spline3(data,dx,dy):
    tmpd    = _spline_1d(data.T,dx,'x')
    shifted = _spline_1d(tmpd.T,dy,'y')
    return shifted

def _spline_1d(data,d,axis):
    v = 6 * (data[2:,:]+data[:-2,:]-2*data[1:-1,:])
    u = cp.zeros_like(data)
    u[1:-1,:] = cp.dot(_Ms[axis],v)

    a = (u[1:,:]-u[:-1,:]) / 6
    b = u[:-1,:] / 2
    c = (data[1:,:]-data[:-1,:]) - (2*u[:-1,:]+u[1:,:])/6

    result = a*(1-d)**3 + b*(1-d)**2 + c*(1-d) + data[:-1,:]
    return result

_interp = {'spline3': _spline3, 'poly3': _poly3, 'linear': _linear}

#############################
#   imcombine
#############################

def imcombine(list,data,name,combine='mean',header=None,iter=3,width=3.0):
    '''
    Calculate sigma-clipped mean or median (no rejection) of images,
    and write to FITS file

    Parameters
    ----------
    list : array-like
        A list of names of images combined

    data : 3-dimension cupy.ndarray
        An array of images stacked along the 1st axis
    
    name : str
        A name of output FITS file

    combine : 'mean' or 'median', default 'mean'
        An algorithm of combine images
        'mean' is sigma-clipped mean, 'median' is median (no rejection).

    header : astropy.io.fits.header, default None
        A header for output FITS file

    iter : int, default 3
        A number of sigmaclipping iterations

    width : int or float, default 3.0
        A clipping width in sigma units
    '''

    if   combine == 'mean':
        combined = sigclipped_mean(data,iter=iter,width=width,axis=0)
    elif combine == 'median':
        combined = _median(data)
    else:
        raise TypeError('"%s" is not defined as algorithm'%combine)

    hdu = fits.PrimaryHDU(combined.get())

    if header:
        hdu.header = header
    for i,f in enumerate(list,1):
        hdu.header['IMCMB%03d'%i] = f
    hdu.header['NCOMBINE'] = len(list)

    fits.HDUList(hdu).writeto(name,overwrite=True)

    print('%d frames combined -> %s'%(len(list),name))

def sigclipped_mean(data,iter=3,width=3.0,axis='None'):
    '''
    Calculate sigmaclipped mean of given array data

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
    ----------
    mean : cupy.ndarray (dtype float32)
        sigmaclipped mean of "data"
    '''
    filt = cp.ones_like(data)
    for _ in range(iter):
        filt = _sigmaclip(data,filt,width,axis)
    mean = _filtered_mean(data,filt,axis)

    return mean

def _sigmaclip(data,filt,width,axis):
    mean  = _filtered_mean(data,filt,axis)
    sigma = cp.sqrt(_filtered_mean((data-mean)**2,filt,axis))

    filt  = (width*sigma >= cp.abs(data - mean)).astype('f4')

    return filt

def _filtered_mean(data,filt,axis):
    return cp.sum(data*filt,axis=axis) / cp.sum(filt,axis=axis)

def _median(data):
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

def fixpix(data,mask,range=2):
    '''
    fill the bad pixel with weighted mean of surrounding pixels

    Parameters
    ----------
    data : 3-dimension cupy.ndarray
        An array of images stacked along the 1st axis

    mask : 2-dimension cupy.ndarray
        An array indicates bad pixel positions
        The shape of mask must be same as image.
        The value of bad pixel is 1, and the others is 0.
    
    range : int, default 2
        A size of area for getting mean
        If range=2 and data[i,j] is badpixel, 
        this pixel is filled with weighted mean of pixels
        that aren't bad, and in the area: data[i-2:i+3,j-2:j+3].

    Returns
    ----------
    fixed : 3-dimension cupy.ndarray (dtype float32)
        An array of images fixed bad pixel

    Notes
    ----------
    If all pixels in reference area are bad pixel, the result is 0.
    Set range according to the density of bad pixels.
    '''

    filt   = (cp.ones_like(mask)-mask)[cp.newaxis,:,:]
    fixed  = filt * data.copy()
    dconv  = _convolve(fixed,range)
    nconv  = _convolve(filt,range)
    fixed += mask*dconv / (nconv+(nconv==0.0).astype('f4'))

    return fixed

def _convolve(data,r):
    n_frames, y_len, x_len = data.shape
    conv = cp.zeros([n_frames,r*2+y_len,r*2+x_len],dtype='f4')
    for x,y in product(range(r*2+1),repeat=2):
        r2 = ((x-r)**2 + (y-r)**2)/(r**2)
        conv[:,y:y+y_len,x:x+x_len] += np.exp(-r2/2) * data

    return conv[:,r:-r,r:-r]
