# -*- coding: utf-8 -*-

__version__ = '0.9'
__update__  = '25 October 2019'

dtype = 'float32'

def null(*args):
    pass

def judge_dtype(arg):
    return dtype if (arg is None) else arg