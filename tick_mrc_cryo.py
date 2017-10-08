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
#import kernel
import cryo
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
      b = np.subtract(a,delta)
      aft = np.fft.fftn(b)
   return np.real(b)

def jacobi_step_with_kernel( a, kern, n):
# peform n steps of jacobi sharpening on a
#   for i in range(0,10):
#      print(i,a[i,0],kern[i,0]);
#      sys.stdout.flush()
   aft = np.fft.fftn(a)
   psf = np.fft.fftn(kern)
   b = a.__mul__(1.0)
   for i in range(0,n):
      delta = np.real(np.fft.ifftn( np.multiply(aft,psf)))
#      b = np.add( b, np.subtract(a,delta)).__imul__(0.5) 
      b = np.subtract(a,delta)
      aft = np.fft.fftn(b)
   return np.float32(b)

def rescale(a, upper):
   amax = a.max()
   amin = a.min()
   amax -= amin
   return (a.__sub__(amin)).__mul__(upper/amax)


def main():
    try:
      original = mrcfile.open(sys.argv[1],mode='r')
      output = mrcfile.open(sys.argv[2],mode='w+')
    except IOError:
      print("Could not open the input \nUsage tick_mrc inputfile outputfile.")
      sys.exit()

#    output.set_data(original.data)
#    output.set_data(np.float32( jacobi_step(original.data, 2)))
#    kern = kernel.defocus( original.data, 30. , .1)
#    kern = kernel.defocus( original.data, 50. , .2)
    acell = float(original.header.cella['x'])
    bcell = float(original.header.cella['y'])
    ccell = float(original.header.cella['z'])
    cryo.the_metric = cryo.metric( original.header.nx,original.header.ny,original.header.nz, acell,bcell,ccell)
   # kern = cryo.conic( original.data, 80., 30., 5)
   # kern = cryo.conic( original.data, 80., 2.0, 0.2)
#too big
#    kern = cryo.conic( original.data, 80., 60.*cryo.the_metric.fa, 5*cryo.the_metric.fb)
#both 60 and 45 smooth well, but lose contrast
#    kern = cryo.conic( original.data, 80., 45.*cryo.the_metric.fa, 5*cryo.the_metric.fb)
#    kern = np.add(cryo.conic( original.data, 80., 45.*cryo.the_metric.fa, 5*cryo.the_metric.fb),cryo.defocus( original.data, 30.*cryo.the_metric.fa, 0.1))
    kern = cryo.conic( original.data, 80., 30.*cryo.the_metric.fa, 5*cryo.the_metric.fb)

    kern[(0,0,0)] = 0.
    output.set_data(np.float32( jacobi_step_with_kernel(original.data,kern, 2)))
#    output.set_data(np.float32( jacobi_step_with_kernel(original.data,kern, 10)))
#    output.set_data(np.float32( jacobi_step_with_kernel(original.data,kern, 20)))
# I cannot believe there isn't a method for this
# but there isn't. FTW!
    output.header.nx = original.header.nx
    output.header.ny = original.header.ny
    output.header.nz = original.header.nz
#    output.header.mode = original.header.mode
# it's now float.
    output.header.mode = 2
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
