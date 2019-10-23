# -*- coding: utf-8 -*-
'''
This module contains cupy.ElementwiseKernel or cupy.ReductionKernel instances 
used internally in Eclair. 
These are not intended to be used directly.
'''

from cupy import ElementwiseKernel, ReductionKernel

reduction_kernel = ElementwiseKernel(
    in_params='T x, T b, T d, T f',
    out_params='T z',
    operation='z = (x - b - d) / f',
    name='reduction'
)

neighbor_kernel = ElementwiseKernel(
    in_params='raw T input, float32 dx, float32 dy, int32 width',
    out_params='T output',
    operation='''
    int ix = i%width - roundf(dx) + 0.5;
    int iy = i/width - roundf(dy) + 0.5;
    output = input[ix + iy*width];
    ''',
    name='neighbor'
)

linear_kernel = ElementwiseKernel(
    in_params='raw T x, float32 dx, float32 dy, int32 width',
    out_params='T z',
    operation='''
    float ex = 1 - dx;
    float ey = 1 - dy;
    z = (
        dx * dy * x[i] +
        ex * dy * x[i+1] +
        dx * ey * x[i+width] +
        ex * ey * x[i+1+width]
    )
    ''',
    name='linear'
)

spline_kernel = ElementwiseKernel(
    in_params='raw T u, raw T y, float32 d, int32 width',
    out_params='T z',
    operation='''
    T u1 = u[i];
    T u2 = u[i+width];
    T y1 = y[i];
    T y2 = y[i+width];
    z = (u2-u1)*pow(d,3) + 3*u1*pow(d,2) + (y2-y1-u2-2*u1)*d + y1
    ''',
    name='spline'
)

filterdsum_kernel = ReductionKernel(
    in_params='T x, T f',
    out_params='T y',
    map_expr='x*f',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    name='filterdsum'
)

filterdstd_kernel = ReductionKernel(
    in_params='T x, T m, T f',
    out_params='T y',
    map_expr='pow(x-m,2)*f',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    name='filterdstd'
)

median_kernel = ElementwiseKernel(
    in_params='T x, T f, T m',
    out_params='T z',
    operation='z = x*f + (1-f)*m',
    name='median'
)

clip_kernel = ElementwiseKernel(
    in_params='T d, T f, T c, T s, float32 w',
    out_params='T z',
    operation='''
    T tmp = (abs(d-c) <= w*s);
    z = tmp * f;
    ''',
    name='clip'
)

judge_kernel = ElementwiseKernel(
    in_params='T x',
    out_params='T z',
    operation='''
    z = (x==0)
    ''',
    name='judge'
)

replace_kernel = ElementwiseKernel(
    in_params='T x, T r',
    out_params='T z',
    operation='''
    T t = (x==0);
    z = (1-t)*x + t*r;
    ''',
    name='replace'
)

fix_kernel = ElementwiseKernel(
    in_params='T x ,T m, T d, T n, T f',
    out_params='T z',
    operation='z = (1-m)*x + m*d/(n+f)',
    name='fix'
)

conv_kernel = ElementwiseKernel(
    in_params='''
    T input, T filt, int32 lx0, int32 ly0, int32 lx1, int32 ly1, int32 lxy
    ''',
    out_params='raw T output',
    operation='''
    int ixy = i % lxy;
    int i_x = ixy % lx0;
    int i_y = ixy / lx0;
    int i_z = i / lxy;
    int s_y = i_x + (i_y + i_z*ly1)*lx1;
    int e_y = s_y + 2*lx1;
    T prod = input * filt;
    for (int y=s_y; y<=e_y; y+=lx1) {
        int e_x = y + 2;
        for (int idx=y; idx<=e_x; idx++) {
            output[idx] += prod;
        }
    }
    ''',
    name='convolution'
)