# -*- coding: utf-8 -*-

__version__ = '0.9'
__update__  = '25 October 2019'

dtype = 'float32'

null = lambda *args : None

judge_dtype = lambda x : (
    dtype if (x is None) else x
)