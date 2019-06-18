Eclair
======

Eclair : CUDA-based Library for Astronomical Image Reduction

## Description
This module provides some useful classes and functions
for astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.

If you want to try it, read readme.pdf and use reduction.ipynb and fitsget.ipynb

## Requirements
* NVIDIA GPU
* CUDA
* Below Python modules
  * NumPy
  * Astropy
  * CuPy

## Components
* **eclair.py**  
    source file of module
* **readme.pdf**  
    manual of reduction.ipynb and fitsget.ipynb
* **reduction.ipynb**  
    sample program running on Google Colaboratory for trying Eclair
* **fitsget.ipynb**  
    program to get FITS files as test data for running reduction.ipynb

