#!/usr/bin/env pytho

import numpy as np
import cupy as cp
from astropy.wcs import WCS

from ..util import judge_dtype

preamble = '''
    template <typename C, typename T>
    __device__ void multiply_matrix (C mat, T *vec_in, T *vec_out) {
        typedef unsigned int uint;
        uint rows = mat.shape()[0];
        uint cols = mat.shape()[1];
        uint idx[2];
        uint *i = idx, *j = idx+1;
        T tmp;
        for (*i=0; *i<rows; (*i)++) {
            tmp = 0;
            for (*j=0; *j<cols; (*j)++) {
                tmp = fma(mat[idx], vec_in[*j], tmp);
            }
            vec_out[*i] = tmp;
        }
    }
    template <typename T>
    __device__ void sph2xyz(T cos_p, T sin_p, T cos_t, T sin_t, T *xyz) {
        xyz[0] = sin_t * cos_p;
        xyz[1] = sin_t * sin_p;
        xyz[2] = cos_t;
    }
'''

def __mkmat(wcs,dtype):

    crval = np.deg2rad(wcs.wcs.crval)

    cos_a0, cos_d0 = np.cos(crval)
    sin_a0, sin_d0 = np.sin(crval)

    mat = np.array(
        [
            [-sin_a0, -sin_d0*cos_a0, cos_d0*cos_a0],
            [cos_a0,  -sin_d0*sin_a0, cos_d0*sin_a0],
            [0,       cos_d0,         sin_d0       ]
        ],
    )
    cmat = cp.asarray(mat,dtype=dtype)

    return cmat

def pix2world(wcs,xy,dtype=None):
    dtype = judge_dtype(dtype)
    
    if not isinstance(wcs,WCS):
        raise TypeError('wcs must be astropy.wcs.WCS object')
    
    xy = cp.array(xy,dtype=dtype)
    if (xy.ndim != 2) or (xy.shape[-1] != 2):
        raise ValueError('shape of xy is incorrect')

    cd = cp.asarray(np.deg2rad(wcs.wcs.cd),dtype=dtype)
    crpix = cp.asarray(wcs.wcs.crpix-1,dtype=dtype)
    mat = __mkmat(wcs,dtype)
    
    ad = pix2world_kernel(
        crpix, cd, mat,
        xy,
        size=len(xy)
    )

    return ad

pix2world_kernel = cp.ElementwiseKernel(
    in_params='raw T crpix, raw T cd, raw T mat',
    out_params='raw T xy',
    operation='''
    // initialize
    const T rad2deg = 180 / M_PI;
    const T Tau = scalbn(M_PI, 1);
    unsigned int ind_0[] = {i,0}, ind_1[] = {i,1};
    T r_xy[2] = {
        xy[ind_0] - crpix[0],
        xy[ind_1] - crpix[1]
    };
    T r_ad[2];

    // apply the CD matrix
    multiply_matrix(cd, r_xy, r_ad);
    T *da = r_ad, *dd = r_ad+1;

    // reverse projection
    T nrm = hypot(*da,*dd);
    T cosgrd = *da / nrm;
    T singrd = *dd / nrm;
    T sep = atan(nrm);

    // convert to 3D Cartesian coordinates
    T sinsep, cossep;
    sincos(sep, &sinsep, &cossep);
    T xyz0[3], xyz1[3];
    sph2xyz(cosgrd, singrd, cossep, sinsep, xyz0);

    // rotation
    multiply_matrix(mat, xyz0, xyz1);
    T *x1 = xyz1, *y1 = xyz1+1, *z1 = xyz1+2;
    T sn = hypot(*x1, *y1);

    // convert to equatorial coordinates
    T a = atan2(*y1, *x1) + (signbit(*y1)? Tau : 0);
    T d = atan2(*z1, sn);
    xy[ind_0] = rad2deg * a;
    xy[ind_1] = rad2deg * d;
    ''',
    preamble=preamble,
    name='pix2world'
)

def world2pix(wcs,ad,dtype=None):
    dtype = judge_dtype(dtype)

    if not isinstance(wcs,WCS):
        raise TypeError('wcs must be astropy.wcs.WCS object')

    ad = cp.array(ad,dtype=dtype)
    if (ad.ndim != 2) or (ad.shape[-1] != 2):
        raise ValueError('shape of xy is incorrect')

    inv_cd = cp.asarray(np.linalg.inv(np.deg2rad(wcs.wcs.cd)),dtype=dtype)
    crpix = cp.asarray(wcs.wcs.crpix-1,dtype=dtype)
    mat = __mkmat(wcs,dtype).T

    xy = world2pix_kernel(
        crpix, inv_cd, mat,
        ad,
        size=len(ad)
    )

    return xy

world2pix_kernel = cp.ElementwiseKernel(
    in_params='raw T crpix, raw T inv_cd, raw T mat',
    out_params='raw T ad',
    operation='''
    // initialize
    const T NaN = nan("");
    const T deg2rad = M_PI / 180;
    unsigned int ind_0[] = {i,0}, ind_1[] = {i,1};
    T a = deg2rad * ad[ind_0];
    T d = deg2rad * ad[ind_1];

    // convert to 3D Cartesian coordinates
    T sin_a, cos_a, sin_d, cos_d;
    sincos(a, &sin_a, &cos_a);
    sincos(d, &sin_d, &cos_d);
    T xyz0[3], xyz1[3];
    sph2xyz(cos_a, sin_a, sin_d, cos_d, xyz0);

    // rotation
    multiply_matrix(mat, xyz0, xyz1);
    T *x1 = xyz1, *y1 = xyz1+1, *z1 = xyz1+2;

    // projection
    T sn = hypot(*x1, *y1);
    T da = *x1 / sn;
    T dd = *y1 / sn;
    T sep = atan2(sn, *z1);
    T nrm = tan(sep);
    nrm = (signbit(*z1)? NaN : nrm);
    T r_ad[] = {
        nrm * da,
        nrm * dd
    };
    T r_xy[2];
    
    // apply the inverse CD matrix
    multiply_matrix(inv_cd, r_ad, r_xy);
    ad[ind_0] = r_xy[0] + crpix[0];
    ad[ind_1] = r_xy[1] + crpix[1];
    ''',
    preamble=preamble,
    name='world2pix'
)