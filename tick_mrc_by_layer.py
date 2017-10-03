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
      output = mrcfile.open(sys.argv[2],mode='w+')
    except IOError:
      print("Could not open the input \nUsage tick_mrc inputfile outputfile.")
      sys.exit()

# create list of layers.
    layers = []
    for layer in original.data:
       layers.append(np.float32(jacobi_step(layer,2)))
    output.set_data(np.array(layers))

#    output.set_data(original.data)
#    output.set_data(np.float32( jacobi_step(original.data, 2)))
# I cannot believe there isn't a method for this
# but there isn't. FTW!
    output.header.nx = original.header.nx
    output.header.ny = original.header.ny
    output.header.nz = original.header.nz
#    output.header.mode = original.header.mode
    output.header.mode = 2   #it is now float.
    output.header.nxstart = original.header.nxstart
    output.header.nystart = original.header.nystart
    output.header.nzstart = original.header.nzstart
    output.header.mx = original.header.mx
    output.header.my = original.header.my
    output.header.mz = original.header.mz
    output.header.cella = original.header.cella
    output.header.cellb = original.header.cellb
    output.header.mapc = original.header.mapc
    output.header.mapr = original.header.mapr
    output.header.maps = original.header.maps
    output.header.ispg = original.header.ispg
    output.header.nsymbt = original.header.nsymbt
    output.set_extended_header(original.extended_header)
    output.header.exttyp = original.header.exttyp
    output.header.nversion = original.header.nversion
    output.header.origin = original.header.origin
    output.header.map = original.header.map
    output.header.machst = original.header.machst
    output.header.rms = original.header.rms
    output.header.nlabl = original.header.nlabl
    output.header.label = original.header.label
    output.close()





main()
