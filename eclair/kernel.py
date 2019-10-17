# -*- coding: utf-8 -*-

from cupy import ElementwiseKernel, ReductionKernel

reduction_kernel = ElementwiseKernel(
    in_params='T x, T b, T d, T f',
    out_params='T z',
    operation='z = (x - b - d) / f',
    name='reduction'
)

liner_kernel = ElementwiseKernel(
    in_params='T x1, T x2, T x3, T x4, T dx, T dy',
    out_params='T z',
    operation='''
    z = dx*dy*x1 + dx*(1-dy)*x2 + (1-dx)*dy*x3 + (1-dx)*(1-dy)*x4
    ''',
    name='linear'
)

spline_kernel = ElementwiseKernel(
    in_params='T u1, T u2, T y1, T y2, T d',
    out_params='T z',
    operation='''
    z = (u1-u2)*pow(1-d,3) + 3*u2*pow(1-d,2) + (y1-y2-u1-2*u2)*(1-d) + y2
    ''',
    name='spline'
)

sum_kernel = ReductionKernel(
    in_params='T x, T f',
    out_params='T y',
    map_expr='x*f',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    name='sum'
)

sqm_kernel = ReductionKernel(
    in_params='T x, T m, T f',
    out_params='T y',
    map_expr='pow(x-m,2)*f',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    name='sqared_sum'
)

median_kernel = ElementwiseKernel(
    in_params='T x, T f, T m',
    out_params='T z',
    operation='z = x*f + (1-f)*m',
    name='median'
)

clip_kernel = ElementwiseKernel(
    in_params='T d, T f, T c, T s, T w',
    out_params='T z',
    operation='''
    float t;
    if (fabsf(d-c) <= w*s) {
        t = 1.0;
    } else {
        t = 0.0;
    }
    z = t*f;
    ''',
    name='clip'
)

judge_kernel = ElementwiseKernel(
    in_params='T x',
    out_params='T z',
    operation='''
    if (x==0.0) {
        z=1.0;
    } else {
        z=0.0;
    }
    ''',
    name='judge'
)

fix_kernel = ElementwiseKernel(
    in_params='T m, T d, T n, T f',
    out_params='T z',
    operation='z = m*d / (n+f)',
    name='fix'
)

conv_kernel = ElementwiseKernel(
    in_params='''
    T input, int16 lx0, int16 ly0, int16 lx1, int16 ly1, int32 lxy
    ''',
    out_params='raw T output',
    operation='''
    int ixy = i % lxy;
    int i_x = ixy % lx0;
    int i_y = ixy / lx0;
    int i_z = i / lxy;
    for(int x=i_x; x<=i_x+2; x++){
        for(int y=i_y; y<=i_y+2; y++){
            int idx = x+(y+i_z*ly1)*lx1;
            output[idx] += input;
        }
    }
    ''',
    name='convolution'
)