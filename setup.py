#! /usr/bin/env python
#-*- encoding:utf-8 -*-

import sys
import re
import shlex
import subprocess
import pkg_resources

from setuptools import setup
from setuptools import find_packages

desc  = 'Eclair: CUDA-based Library for Astronomical Image Reduction'
ldesc = '''
This package provides some useful classes and functions
in astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.
'''

def getoutput(cmd):
    if sys.version_info.major == 2:
        output = subprocess.check_output(cmd)
    else:
        cp = subprocess.run(cmd)
        cp.check_returncode()
        output = cp.stdout

    return output

def get_cuda_version():
    try:
        output = getoutput(shlex.split('nvcc -V'))
    except subprocess.CalledProcessError:
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

    cuda_version = get_cuda_version()
    if cuda_version is None:
        requires.append('cupy')

    with open('eclair/common.py') as f:
        exec(f.read())
        assert __version__

    setup(
        name='eclair',
        packages=['eclair'],
        version=__version__,
        description=desc,
        long_description=ldesc,

        author='Masafumi Niwano',
        author_email='niwano@hp.phys.titech.ac.jp',
        url='https://github.com/MNiwano/Eclair',

        install_requires=[get_correspond_cupy(),'astropy','numpy'],
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