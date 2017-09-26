#!/usr/bin/python
# (c) 2017 Treadco software.
# this defaults to python 2 on my machine

import numpy as np
import sys,os
from PIL import Image
from PIL import ImageChops
from pylab import *

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
   b = a.__mul__(1.0)
   for i in range(0,n):
      delta = np.real(np.fft.ifftn( np.multiply(aft,psf)))
#      a = np.subtract(a, delta.__mul__(0.1))
   #   b = np.add( a, np.subtract(b, delta).__mul__(0.5))
#      b = np.add( a, np.subtract(b, delta))
      b = np.add( b, np.subtract(a, delta))
      aft = np.fft.fftn(b)
   return np.real(b)

def rescale(a, upper):
   amax = a.max()
   amin = a.min()
   amax -= amin
   b = a.__sub__(amin)
   c = b.__mul__(upper/amax)
   return c

def main():
    try:
      image = Image.open(sys.argv[1])
    except IOError:
      print("Could not open the input \nUsage tick_jpg inputfile.")
      sys.exit()

# have an open image right now.
    im = np.real(np.array(image.convert('L')))
    b = jacobi_step(im, 10)
    inew = Image.fromarray(np.uint8(rescale(im,255.0)))
    inew.save('before.jpg')
    iafter = Image.fromarray(np.uint8(rescale(b,255.0)))
    iafter.save('after.jpg')
    ix = ImageChops.subtract(iafter, inew,0.1)
    ix.save('difference.jpg')
    inew = Image.fromarray(np.uint8(rescale(generate_psf(im),255.0)))
    inew.save('psf.jpg')
 
#    imshow(im)

main()
