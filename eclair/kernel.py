
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
    in_params='raw T input, T dx, T dy',
    out_params='T output',
    operation='''
        int width = input.shape()[1];
        T h = 0.5;
        int idx[] = {
            h + i/width - (dy>=h),
            h + i%width - (dx>=h)
        };
        output = input[idx];
    ''',
    name='neighbor'
)

linear_kernel = ElementwiseKernel(
    in_params='raw T x, T dx, T dy',
    out_params='T z',
    operation='''
        T ex = 1-dx, ey = 1-dy;
        int i2 = i + x.shape()[1];
        z = dy*(dx*x[i] + ex*x[i+1]) + ey*(dx*x[i2] + ex*x[i2+1]);
    ''',
    name='linear'
)

poly_kernel = ElementwiseKernel(
    in_params='raw T input, raw T mat',
    out_params='T output',
    operation='''
        int width = input.shape()[1] - 3;
        int i_x = i % width, i_y = i / width;
        int idx1[2], idx2[2];
        int *y1 = &(idx1[0]), *x1 = &(idx1[1]);
        int *y2 = &(idx2[0]), *x2 = &(idx2[1]);
        output = 0;
        for (*y1=0, *y2=i_y; *y1<=3; (*y1)++, (*y2)++) {
            T tmp = 0;
            for (*x1=0, *x2=i_x; *x1<=3; (*x1)++, (*x2)++) {
                tmp += mat[idx1] * input[idx2];
            }
            output += tmp;
        }
    ''',
    name='polynomial'
)

spline_kernel = ElementwiseKernel(
    in_params='raw T u, raw T y, T d',
    out_params='T z',
    operation='''
        int i2 = i + u.shape()[1];
        T u1 = u[i], u2 = u[i2];
        T y1 = y[i], y2 = y[i2];
        T a3 = u2 - u1;
        T a2 = 3 * u1;
        T a1 = (y2-y1) - (2*u1+u2);
        z = y1 + d*(a1 + d*(a2 + d*a3))
    ''',
    name='spline'
)

checkfinite = ElementwiseKernel(
    in_params='T x, T f',
    out_params='T z',
    operation='z = isfinite(x) * f',
    name='checkfinite'
)

replace_kernel = ElementwiseKernel(
    in_params='T input, T before, T after',
    out_params='T output',
    operation='''
        output = (
            (input==before) ? after:input
        )
    ''',
    name='replace'
)

filterdsum = ReductionKernel(
    in_params='T x, T f',
    out_params='T y',
    map_expr='(f ? x : f)',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    name='filterdsum'
)

filterdvar = ReductionKernel(
    in_params='T x, T m, T f',
    out_params='T y',
    map_expr='square(x,m,f)',
    reduce_expr='a+b',
    post_map_expr='y=a',
    identity='0',
    preamble='''
        template <typename T>
        __device__ T square(T x, T m, T f) {
            T dev = x-m;
            T var = dev*dev;
            return (f ? var : f);
        }
    ''',
    name='filterdvar'
)

default_mean = ElementwiseKernel(
    in_params='T x, T n, T d',
    out_params='T z',
    operation='''
        int f = (n==0);
        T t = x /(n+f);
        z = (f ? d : t);
    ''',
    name='default_mean'
)

median_kernel = ElementwiseKernel(
    in_params='raw T input, T nums, T d',
    out_params='T output',
    operation='''
        int n = nums;
        int f = n!=0;
        n -= f;
        int c = n/2;
        int m = _ind.size();
        int i_1 = i + c*m;
        int i_2 = i + (n-c)*m;
        T s = (input[i_1] + input[i_2])/2;
        output = (f ? s : d);
    ''',
    name='median'
)

updatefilt_kernel = ElementwiseKernel(
    in_params='T d, T f, T c, T s, T w',
    out_params='T z',
    operation='''
        T dev = d-c, lim = w*s;
        z = f * (dev*dev < lim*lim);
    ''',
    name='updatefilt'
)

mask2filter = ElementwiseKernel(
    in_params='T m',
    out_params='T f',
    operation='f = (m==0)',
    name='mask2filter'
)

fix_kernel = ElementwiseKernel(
    in_params='T x ,F f, T d, F n',
    out_params='T z',
    operation='''
        T val = d / (n+(n==0));
        z = ((int)f ? x : val);
    ''',
    name='fix'
)

conv_kernel = ElementwiseKernel(
    in_params='raw T input',
    out_params='T output',
    operation='''
        int ly = input.shape()[1];
        int lx = input.shape()[2];
        int iyz = i / lx;
        int i_x = i % lx;
        int i_y = iyz % ly;
        int i_z = iyz / ly;
        int idx[3] = {i_z};
        int *t_y = &(idx[1]), *t_x = &(idx[2]);
        int s_x = max(i_x-1,0), e_x = min(i_x+2,lx);
        int s_y = max(i_y-1,0), e_y = min(i_y+2,ly);
        T c1 = 0, c2 = 0;
        output = 0;
        for (*t_y=s_y; *t_y<e_y; (*t_y)++) {
            T tmp = 0;
            for (*t_x=s_x; *t_x<e_x; (*t_x)++) {
                T val = input[idx];
                int fn = isfinite(val);
                tmp += (fn ? val : fn);
                c1 += 1, c2 += fn;
            }
            output += tmp;
        }
        output *= c1/(c2+(c2==0));
    ''',
    name='convolution'
)
