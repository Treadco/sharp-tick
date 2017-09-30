#!/usr/bin/python
# (c) 2017 Treadco software.
# this defaults to python 2 on my machine

import numpy as np
import sys,os
from PIL import Image
from PIL import ImageChops
from pylab import *

import kernel

def normalize( a ):
# written generally
# a tuple resolves to an index
# just need to find the right tuple
    lindx = []
    for i in range(0, a.ndim):
      lindx.append(0)
    div = 1.0/a[tuple(lindx)]
    return a.__mul__(div)

def generate_psf( a):
# first get the autocorrelation function
   acf = normalize(np.fft.ifftn( np.absolute( np.fft.fftn(rescale(a,1.0))).__ipow__(2)))
   lindx = []
   for i in range(0, a.ndim):
      lindx.append(0)
   volume = a.size
   acf = rescale(np.real(acf),1.0/volume)
   acf[tuple(lindx)] = 0.0
   return np.real(acf)

def jacobi_step( a, n):
# peform n steps of jacobi sharpening on a
   aft = np.fft.fftn(a)
   psf = np.fft.fftn(generate_psf(a))
   b = a.__mul__(1.0) # make a copy
   for i in range(0,n):
      delta = np.real(np.fft.ifftn( np.multiply(aft,psf)))
#      b = np.add( b, np.subtract(a,delta)) 
      b = np.subtract(a,delta)
      aft = np.fft.fftn(b)
   return np.real(b)

def jacobi_step_with_kernel( a, kern, n):
# peform n steps of jacobi sharpening on a
#   for i in range(0,10):
#      print(i,a[i,0],kern[i,0]);
#      sys.stdout.flush()
   aft = np.fft.fftn(a)
   sys.stdout.flush()
   psf = np.fft.fftn(kern)
   b = a.__mul__(1.0) # make a copy
   for i in range(0,n):
      delta = np.real(np.fft.ifftn( np.multiply(aft,psf)))
#      b = np.add( b, np.subtract(a,delta)) 
      b = np.subtract(a,delta)
      aft = np.fft.fftn(b)
   return np.real(b)

def rescale(a, upper):
   amax = a.max()
   amin = a.min()
   amax -= amin
   return (a.__sub__(amin)).__mul__(upper/amax)
#   b = a.__sub__(amin)
#   c = b.__mul__(upper/amax)
#   return c

def main():
    try:
      image = Image.open(sys.argv[1])
    except IOError:
      print("Could not open the input \nUsage tick_jpg inputfile.")
      sys.exit()

    r,g,b = image.split()
    rr = np.real(np.array(r))
    gr = np.real(np.array(g))
    br = np.real(np.array(b))
# too big    kern = kernel.gaussian(rr, 30.0)
#    kern = kernel.gaussian(rr, 20.0)
    kern = kernel.gaussian(rr, 10.0)
    kern[0,0] = 0.0
    rp = jacobi_step_with_kernel(rr,kern,5) 
    gp = jacobi_step_with_kernel(gr,kern,5) 
    bp = jacobi_step_with_kernel(br,kern,5) 
    rn = Image.fromarray(np.uint8(rescale(rp,255.0)))
    gn = Image.fromarray(np.uint8(rescale(gp,255.0)))
    bn = Image.fromarray(np.uint8(rescale(bp,255.0)))
    inew = Image.merge("RGB",(rn,gn,bn))
    inew.save('after.jpg')
    ix = ImageChops.subtract(inew, image,0.1)
    ix.save('difference.jpg')
main()
