# -*- coding: utf-8 -*-
'''
This module contains cupy.ElementwiseKernel or cupy.ReductionKernel instances 
used internally in Eclair. 
These are not intended to be used directly.
'''

from cupy import ElementwiseKernel, ReductionKernel

reduction_kernel = ElementwiseKernel(
    in_params='T x, T b, T d, T f',
    out_params='F z',
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

poly_kernel = ElementwiseKernel(
    in_params='raw T input, raw T vec, int32 w0, int32 w1',
    out_params='T output',
    operation='''
    int i_x = i % w0;
    int i_y = i / w0;
    output = 0;
    for (int dy=0; dy<=3; dy++) {
        int y = (i_y + dy) * w1;
        int dy4 = dy * 4;
        for (int dx=0; dx<=3; dx++) {
            int x = i_x + dx;
            output += vec[dx + dy4] * input[x + y];
        }
    }
    ''',
    name='polynomial'
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

nonzerosum_kernel = ReductionKernel(
    in_params='T x',
    out_params='T y',
    map_expr='x',
    reduce_expr='a+b',
    post_map_expr='y = a + (a==0)',
    identity='0',
    name='nonzerosum'
)

replace_kernel = ElementwiseKernel(
    in_params='T x, T f, T m',
    out_params='T z',
    operation='z = x*f + (1-f)*m',
    name='replace'
)

median_kernel = ElementwiseKernel(
    in_params='raw T input, T nums, int32 nop',
    out_params='T output',
    operation='''
    int n = nums;
    T even = 1 - n%2;
    int idx = i + (n/2 - even)*nop;
    T s = input[idx] + even*input[idx + nop];
    output = (n!=0)*s / (1+even);
    ''',
    name='median'
)

updatefilt_kernel = ElementwiseKernel(
    in_params='T d, T f, T c, T s, float32 w',
    out_params='T z',
    operation='''
    T tmp = (abs(d-c) <= w*s);
    z = tmp * f;
    ''',
    name='updatefilt'
)

zeroreplace_kernel = ElementwiseKernel(
    in_params='T x, T r',
    out_params='T z',
    operation='''
    T t = (x==0);
    z = (1-t)*x + t*r;
    ''',
    name='zeroreplace'
)

fix_kernel = ElementwiseKernel(
    in_params='T x ,T f, T d, T n',
    out_params='T z',
    operation='''
    T t = (n==0);
    z = x + (1-f)*d/(n+t);
    ''',
    name='fix'
)

conv_kernel = ElementwiseKernel(
    in_params='''
    raw T input, int32 lx, int32 ly, int32 lxy
    ''',
    out_params='T output',
    operation='''
    int ixy = i % lxy;
    int i_x = ixy % lx;
    int i_y = ixy / lx;
    int s_x = i_x-1, e_x = i_x+1;
    int s_y = i_y-1, e_y = i_y+1;
    int z = (i / lxy) * ly;
    output = 0;
    for (int y=s_y; y<=e_y; y++) {
        int my = (y + ly) % ly;
        int yz = (z + my) * lx;
        int fy = (my==y);
        for (int x=s_x; x<=e_x; x++) {
            int mx = (x + lx) % lx;
            int f = (mx==x) * fy;
            output += f * input[mx + yz];
        }
    }
    ''',
    name='convolution'
)