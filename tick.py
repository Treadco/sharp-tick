#!/usr/bin/python
# this defaults to python 2 on my machine
# (c) 2017 Treadco software.
#
# Regularized image sharpening library
# 
license = "
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
"

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
   b = a.__mul__(1.0) # make a copy
   for i in range(0,n):
      delta = np.real(np.fft.ifftn( np.multiply(aft,psf)))
      b = np.add( b, np.subtract(a,delta)) 
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

def jacobi_RGB( an_image, ncycles=5):
   r,g,b = an_image.split()
   ar = np.real(array(r))
   ag = np.real(array(r))
   ab = np.real(array(r))
   rp = jacobi_step(rr,ncycles) 
   gp = jacobi_step(gr,ncycles) 
   bp = jacobi_step(br,ncycles) 
   rn = Image.fromarray(np.uint8(rescale(rp,255.0)))
   gn = Image.fromarray(np.uint8(rescale(gp,255.0)))
   bn = Image.fromarray(np.uint8(rescale(bp,255.0)))
   return Image.merge("RGB",(rn,gn,bn))



def main():
    try:
      image = Image.open(sys.argv[1])
    except IOError:
      print("Could not open the input \nUsage tick_jpg inputfile.")
      sys.exit()

# have an open image right now.
# old b&w code
#    im = np.real(np.array(image.convert('L')))
#    b = jacobi_step(im, 10)
#    inew = Image.fromarray(np.uint8(rescale(im,255.0)))
#    inew.save('before.jpg')
#    iafter = Image.fromarray(np.uint8(rescale(b,255.0)))
#    iafter.save('after.jpg')
#    ix = ImageChops.subtract(iafter, inew,0.1)
#    ix.save('difference.jpg')
#    inew = Image.fromarray(np.uint8(rescale(generate_psf(im),255.0)))
#    inew.save('psf.jpg')
# 
#    imshow(im)
    r,g,b = image.split()
    rr = np.real(np.array(r))
    gr = np.real(np.array(g))
    br = np.real(np.array(b))
    rp = jacobi_step(rr,20) 
    gp = jacobi_step(gr,20) 
    bp = jacobi_step(br,20) 
    rn = Image.fromarray(np.uint8(rescale(rp,255.0)))
    gn = Image.fromarray(np.uint8(rescale(gp,255.0)))
    bn = Image.fromarray(np.uint8(rescale(bp,255.0)))
    inew = Image.merge("RGB",(rn,gn,bn))
    inew.save('after.jpg')
    ix = ImageChops.subtract(inew, image,0.1)
    ix.save('difference.jpg')
main()
