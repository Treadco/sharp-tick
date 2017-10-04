#!/usr/bin/python
# this defaults to python 2 on my machine
# (c) 2017 Treadco software.
#
# Regularized image sharpening library
# 
# Synthetic kernel methods
#
# Cryo-em specific kernels
# added to the original kernel.py synthetic kernel library
# so import either kernel or cryo but not both.
# (unless you want undefined behavoir)
#
license =''' 
Copyright (c) 2017  Treadco LLC, Amelia Treader, Robert W Harrison

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import sys,os
from math import exp as exp
from math import sqrt as sqrt
from PIL import Image
from PIL import ImageChops
#from pylab import *

#import tick 
# cannot import tick methods because tick imports kernel

#Gaussian blur
def gaussian(an_image, radius):
   sys.stdout.flush()
   rout = int(radius*2)
   radius = radius*radius
   b = np.zeros_like(np.float32(an_image))
   if an_image.ndim == 2:
     for i in range(-rout,rout): 
        ir = float(i)
        iu = i
        if iu < 0:
          iu += an_image.shape[0]
        for j in range(-rout,rout): 
          jr = float(j)
          x = exp( -(ir*ir+jr*jr)/radius) 
          ju = j
          if ju < 0:
            ju += an_image.shape[1]
          b[(iu,ju)] += x
   elif an_image.ndim == 3:
     for i in range(-rout,rout): 
        ir = float(i)
        iu = i
        if iu < 0:
          iu += an_image.shape[0]
        for j in range(-rout,rout): 
          jr = float(j)
          ju = j
          if ju < 0:
            ju += an_image.shape[1]
          for k in range(-rout,rout):
            kr = float(k)
            ku = k
            if ku < 0:
              ku += an_image.shape[2]
            x = exp( -(ir*ir+jr*jr+kr*kr)/radius) 
            b[(iu,ju,ku)] += x
#
#  the FFT will scale by sqrt(size)
#  the ACF uses size, but that's because it uses
#  two FFTs so it's already sqrt(size) bigger
#
   x = float(an_image.size)
   x = sqrt(x)
   b = b.__div__(x)
   return b

def gaussian_3d(an_image, rx,ry,rz):
   routx = int(rx*2)
   routy = int(ry*2)
   routz = int(rz*2)
   rx = rx*rx
   ry = ry*ry
   rz = rz*rz
   b = np.zeros_like(np.float32(an_image))
   for i in range(-routx,routx): 
        ir = float(i)
        iu = i
        if iu < 0:
          iu += an_image.shape[0]
        fx = (-ir*ir/rx)
        for j in range(-routy,routy): 
          jr = float(j)
          ju = j
          if ju < 0:
            ju += an_image.shape[1]
          fy = (-jr*jr/ry)
          for k in range(-routz,routz):
            kr = float(k)
            ku = k
            if ku < 0:
              ku += an_image.shape[2]
            fz = (-kr*kr/rz)
            b[(iu,ju,ku)] += exp(fx+fy+fz)
#
#  the FFT will scale by sqrt(size)
#  the ACF uses size, but that's because it uses
#  two FFTs so it's already sqrt(size) bigger
#
   x = float(an_image.size)
   x = sqrt(x)
   b = b.__div__(x)
   return b

def defocus(an_image, radius,alpha):
   rout = int(radius*2)
   alpha = alpha/radius/radius
   b = np.zeros_like(np.float32(an_image))
   if an_image.ndim == 2:
     for i in range(-rout,rout): 
        ir = float(i)
        iu = i
        if iu < 0:
          iu += an_image.shape[0]
        for j in range(-rout,rout): 
          jr = float(j)
          x = ir*ir+jr*jr
          if( x > radius):
              x = 0.
          else:
              x = x*alpha + 1.
          ju = j
          if ju < 0:
            ju += an_image.shape[1]
          b[(iu,ju)] += x
   elif an_image.ndim == 3:
     for i in range(-rout,rout): 
        ir = float(i)
        iu = i
        if iu < 0:
          iu += an_image.shape[0]
        for j in range(-rout,rout): 
          jr = float(j)
          ju = j
          if ju < 0:
            ju += an_image.shape[1]
          for k in range(-rout,rout):
            kr = float(k)
            ku = k
            if ku < 0:
              ku += an_image.shape[2]
            x = ir*ir+jr*jr+kr*kr
            if( x > radius):
                x = 0.
            else:
                x = x*alpha + 1.

            b[(iu,ju,ku)] += x
#
#  the FFT will scale by sqrt(size)
#  the ACF uses size, but that's because it uses
#  two FFTs so it's already sqrt(size) bigger
#
   x = float(an_image.size)
   x = sqrt(x)
   b = b.__div__(x)
   return b

#
# threshold at the mean
# using Zscore
#
def threshold(an_image):
   a = np.float32(an_image)
#   b = np.zeros_like(a)
   m = np.mean(a)
   s = np.std(a)
   return (((np.float32( (np.uint((a.__sub__(m)).__div__(s)) ).clip(0,1) )).__mul__(s))).__add__(m)

# define a jacobi psf from a kernel
def prep_kernel_for_jacobi( a):
#   b = tick.normalize(a)
   b = a.__add__(0)  
   lindx = []
   for i in range(0,b.ndim):
       lindx.append(0)
   b[tuple(lindx)] = 0.0
   sys.stdout.flush()
   return b 


def jacobi_step_with_kernel( a, kern, n):
# peform n steps of jacobi sharpening on a
#   for i in range(0,10):
#      print(i,a[i,0],kern[i,0]);
#      sys.stdout.flush()
   aft = np.fft.fftn(a)
   psf = np.fft.fftn(kern)
#   b = a.__mul__(1.0) # make a copy
   b = threshold(a)
   for i in range(0,n):
      delta = np.real(np.fft.ifftn( np.multiply(aft,psf)))
#      b = np.add( b, np.subtract(a,delta)).__mul__(0.5) 
      b = np.subtract(a,delta)
      aft = np.fft.fftn(b)
   return np.real(b)

