#! /usr/bin/env python
#-*- encoding:utf-8 -*-

import sys
import re
import shlex
import subprocess
import pkg_resources

from setuptools import setup
from setuptools import find_packages

desc  = 'Eclaire: CUDA-based Library for Astronomical Image REduction'
ldesc = '''
This package provides some useful classes and functions
in astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.
'''

def getoutput(cmd):
    if sys.version_info.major == 2:
        output = subprocess.check_output(cmd)
    else:
        cp = subprocess.run(cmd,stdout=subprocess.PIPE)
        cp.check_returncode()
        output = cp.stdout.decode()

    return output

def get_cuda_version():
    try:
        output = getoutput(shlex.split('nvcc -V'))
    except Exception:
        return None

    match = re.search(r'release\s([\d\.]+)',output)
    if match is not None:
        result, = match.groups()
        return result
    else:
        return None

def get_correspond_cupy():
    cuda_version = get_cuda_version()
    pkg_name = 'cupy'
    if cuda_version is not None:
        try:
            current = pkg_resources.get_distribution(
                'cupy-cuda{}'.format(cuda_version.replace('.',''))
            )
        except Exception:
            pass
        else:
            pkg_name = current.project_name
    
    return pkg_name

if __name__ == '__main__':

    with open('eclaire/common.py') as f:
        exec(f.read())
        assert __version__

    cupyname = get_correspond_cupy()

    setup(
        name='eclaire',
        packages=['eclaire'],
        version=__version__,
        description=desc,
        long_description=ldesc,

        author='Masafumi Niwano',
        author_email='niwano@hp.phys.titech.ac.jp',
        url='https://github.com/MNiwano/Eclaire',

        install_requires=[cupyname,'astropy','numpy'],
        keywords = ['astronomy', 'science', 'fits', 'GPU', 'CUDA'],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: GPU'
            'Environment :: GPU :: NVIDIA CUDA',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering'
        ]
    )