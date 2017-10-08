#!/usr/bin/python
# this defaults to python 2 on my machine
# (c) 2017 Treadco software.
#
# Regularized image sharpening library
#
# MRC file version
# 
license = ''' Copyright (c) 2017  Treadco LLC, Amelia Treader, Robert W Harrison
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
from PIL import Image
from PIL import ImageChops
#from pylab import *
import mrcfile

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
      b =  np.subtract(a,delta)
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
      original = mrcfile.open(sys.argv[1],mode='r')
      original2 = mrcfile.open(sys.argv[2],mode='r')
    except IOError:
      print("Could not open the input \nUsage tick_mrc inputfile outputfile.")
      sys.exit()

# create list of layers.
    layers = []
    for layer in original.data:
       layers.append(np.float32(layer))
    layers2 = []
    for layer in original2.data:
       layers2.append(np.float32(layer))
#layers are the arrays containing the data.
    the_image = np.zeros_like(layers[0])
    the_diff_image = np.zeros_like(layers[0])
#    for i in range(int(len(layers)*0.35),int(len(layers)*0.65)):
    for i in range(int(len(layers)*0.45),int(len(layers)*0.55)):
        the_image = np.add(the_image,layers[i])
    for i in range(int(len(layers)*0.35),int(len(layers)*0.65)):
        the_diff_image = np.add(the_diff_image, np.subtract(layers[i],layers2[i]))

    s = the_image.std()
    m = the_image.mean()
    
    the_sum = rescale(np.clip(the_image,m-3*s,m+3*s),255.)
#    the_sum = rescale(np.clip(the_image,m-2*s,m+2*s),255.)
        
    the_image = Image.fromarray(np.uint8( the_sum))
    the_image.save('sum.jpg')
    the_diff_image = Image.fromarray(np.uint8( rescale(the_diff_image,255.)))
    the_diff_image.save('diff.jpg')





main()
