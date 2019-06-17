Eclair
======

Eclair : CUDA-based Library for Astronomical Image Reduction

## Description
This module provides some useful classes and functions
for astronomical image reduction, 
and their processing speed is acceralated by using GPU via CUDA.

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
* **reduction.ipynb**  
    sample program running on Google Colaboratory for trying Eclair
* **fitsget.ipynb**  
    program to get FITS files as test data for running reduction.ipynb
* **readme.pdf**  
    manual of reduction.ipynb and fitsget.ipynb
