# -*- coding: utf-8 -*-

from cupy import ElementwiseKernel, ReductionKernel

reduction_kernel = ElementwiseKernel(
    'T x, T b, T d, T f',
    'T z',
    'z = (x - b - d) / f',
    'reduction'
)

liner_kernel = ElementwiseKernel(
    'T x1, T x2, T x3, T x4, T dx, T dy',
    'T z',
    'z= dx*dy*x1 + dx*(1-dy)*x2 + (1-dx)*dy*x3 + (1-dx)*(1-dy)*x4',
    'linear'
)

spline_kernel = ElementwiseKernel(
    'T u, T v, T x, T y, T d','T z',
    'z = (u-v)*(1-d)*(1-d)*(1-d) + 3*v*(1-d)*(1-d) + (x-y-u-2*v)*(1-d) + y',
    'spline'
)

sum_kernel = ReductionKernel(
    'T x, T f',
    'T y',
    'x*f',
    'a+b',
    'y=a',
    '0',
    'sum'
)

sqm_kernel = ReductionKernel(
    'T x, T m, T f',
    'T y',
    '(x-m)*(x-m)*f',
    'a+b',
    'y=a',
    '0',
    'sqared_sum'
)

median_kernel = ElementwiseKernel(
    'T x, T f, T m',
    'T z',
    'z = x*f + (1-f)*m',
    'median'
)

clip_kernel = ElementwiseKernel(
    'T d, T c, T s, T w',
    'T z',
   'if (fabsf(d-c) <= w*s) {z = 1.0;} else {z = 0.0;}',
   'clip'
)

judge_kernel = ElementwiseKernel(
    'T x',
    'T z',
    'if (x==0.0) {z=1.0;} else {z=0.0;}',
    'judge'
)

fix_kernel = ElementwiseKernel(
    'T m, T d, T n, T f',
    'T z',
    'z=m*d/(n+f)',
    'fix'
)
