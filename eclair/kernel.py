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
    operation='z= dx*dy*x1 + dx*(1-dy)*x2 + (1-dx)*dy*x3 + (1-dx)*(1-dy)*x4',
    name='linear'
)

spline_kernel = ElementwiseKernel(
    in_params='T u, T v, T x, T y, T d',
    out_params='T z',
    operation='z = (u-v)*(1-d)*(1-d)*(1-d) + 3*v*(1-d)*(1-d) + (x-y-u-2*v)*(1-d) + y',
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
    map_expr='(x-m)*(x-m)*f',
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
    in_params='T d, T c, T s, T w',
    out_params='T z',
    operation='if (fabsf(d-c) <= w*s) {z = 1.0;} else {z = 0.0;}',
    name='clip'
)

judge_kernel = ElementwiseKernel(
    in_params='T x',
    out_params='T z',
    operation='if (x==0.0) {z=1.0;} else {z=0.0;}',
    name='judge'
)

fix_kernel = ElementwiseKernel(
    in_params='T m, T d, T n, T f',
    out_params='T z',
    operation='z=m*d/(n+f)',
    name='fix'
)
