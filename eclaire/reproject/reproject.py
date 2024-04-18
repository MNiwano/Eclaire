
import os
import warnings
from collections.abc import Sequence
from concurrent import futures as cf

import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
from astropy.io import fits
from astropy.wcs import WCS

from .. import util
from . import wcslib

def reproject(
        data, input_projections, output_projection, shape=None, order=3,
        out=None, dtype=None, impliment='eclaire'):
    '''
    Reproject images to a given projection using the spline interpolation.

    Parameters
    ----------
    data : 3D array
        An array of images stacked along the 1st dimesion (axis=0).
    input_projections : sequence of astropy Header or WCS
        Projections of input images.
    output_projection : Header or WCS
        The output projection.
    shape : tuple, default None
        The shape of output images.
        If None, use the shape which output_projection contains.
    order : int, default 3
        The order of the spline interpolation.
    out : 3D cupy.ndarray, default None
        Alternate output array in which to place the result. The default
        is None; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    dtype : str or dtype, default None
        dtype of array used internally.
        If None, use eclair.common.default_dtype.
        If the input dtype is different, use a casted copy.
    implement: {'eclaire', 'astropy'}, default 'eclaire'
        Implementation of coordinate transformation.
        * 'eclaire' - it is performed by functions implemented with GPU.
            It is fast and low-load, but supports limited coordinate systems.
        * 'astropy' - it is performed by methods of astropy.wcs.WCS object.
            It uses CPU multithreads to compensate for the slowness,
            and does frequent host-device data transfers.
            This results a heavy load on the hardware.

    Returns
    -------
    reprojected : 3D cupy.ndarray
        An array in which the reprojected images are stacked.
        An area outside of the input image footprint are filled with NaN.
    '''

    dtype = util.judge_dtype(dtype)
    data = cp.asarray(data,dtype=dtype)
    if data.ndim != 3:
        raise ValueError('input array must be 3D array')
    
    wcslist = []
    if not isinstance(input_projections,Sequence):
        raise TypeError('input_projection must be a sequence of Header or WCS')
    for inp in input_projections:
        if isinstance(inp,fits.Header):
            wcs = WCS(inp)
        elif isinstance(inp,WCS):
            wcs = inp
        else:
            raise TypeError(
                'items of input_projections must be Header or WCS'
            )
        wcslist.append(wcs)
    else:
        if len(wcslist) != len(data):
            raise ValueError(
                'lengths of input_projections differs from input array'
            )

    if isinstance(output_projection,fits.Header):
        base_wcs = WCS(output_projection)
    elif isinstance(output_projection,WCS):
        base_wcs = output_projection
    else:
        raise TypeError('output_projection must be Header or WCS')

    if shape is None:
        if isinstance(base_wcs.array_shape,tuple):
            shape = base_wcs.array_shape
        else:
            raise ValueError('Need to specify output shape')
    else:
        shape = tuple(shape)

    out_shape = (len(data),*shape)
    if out is None:
        reprojected = cp.empty(out_shape,dtype=dtype)
    else:
        if not isinstance(out,cp.ndarray):
            raise TypeError('output array must be cupy.ndarray')
        elif out.shape != out_shape:
            raise ValueError('shape of output array is incorrect')
        else:
            reprojected = out
    
    params = dict(
        order=order,
        mode='constant',
        cval=cp.nan,
    )
    if impliment == 'eclaire':
        indice = __mk_indice_array(shape,cp)
        coords_ad = wcslib.pix2world(base_wcs,indice,dtype=dtype)

        for tdata, wcs, tout in zip(data,wcslist,reprojected):
            coords_xy = cp.flipud(
                wcslib.world2pix(wcs,coords_ad,dtype=dtype).T
            )
            ndimage.map_coordinates(
                tdata, coords_xy, output=tout.ravel(), **params
            )
    elif impliment == 'astropy':
        indice = __mk_indice_array(shape,np)
        coords_ad = base_wcs.wcs_pix2world(indice,0).astype(dtype,copy=False)

        with cf.ThreadPoolExecutor(max_workers=os.cpu_cout()) as executor:
            iterator = executor.map(
                lambda wcs: wcs.wcs_world2pix(coords_ad,0).T,
                wcslist
            )
            for tdata, coords_xy, tout in zip(data,iterator,reprojected):
                ndimage.map_coordinates(
                    tdata, cp.asarray(coords_xy,dtype=dtype),
                    output=tout.ravel(), **params
                )
    else:
        raise ValueError('Invalid argument for "implement"')

    return reprojected

def __mk_indice_array(shape,xp):
    indice = xp.flipud(
        xp.mgrid[tuple(slice(0,l) for l in shape)]
    ).reshape(2,-1).T
    return indice