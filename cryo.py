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
from math import sin as sin
from math import cos as cos 
from math import acos as acos 
from PIL import Image
from PIL import ImageChops
#from pylab import *

#import tick 
# cannot import tick methods because tick imports kernel

#
# conic
# generate a conic section kernel to compensate for
# the tomographic errors
#
#  on Y
#  blur/defocus x
#  z will be f(x,y)
#
class metric:
   def __init__(self, nx,ny,nz, ca,cb,cc):
#      print(ca,cb,cc)
#      sys.stdout.flush()
      self.fa = float(ca)/nx
      self.fb = float(cb)/ny
      self.fc = float(cc)/nz
   def distance( me, dx,dy,dz):
      dx *= me.fa
      dy *= me.fb
      dz *= me.fc
      return sqrt(dx*dx + dy*dy + dz*dz)
   def distance2( me, dx,dy,dz):
      dx *= me.fa
      dy *= me.fb
      dz *= me.fc
      return dx*dx + dy*dy + dz*dz
   def angle_from_zaxis_rot_on_y(me,x,y,z):
      dx = x*me.fa
      dz = z*me.fc
      r = sqrt( dx*dx + dz*dz)
      if r > 0:
         return acos(dx/r)
      else: 
        return 0.
 
#
# create a default metric
# this is pixel-wise
#
the_metric = metric(1,1,1,1.0,1.0,1.0)
  
def conic( an_image, half_angle, radius, blur_radius):
#   print( the_metric.fa, the_metric.fb,the_metric.fc)
#   sys.stdout.flush()
   ha = half_angle*3.14159265/180.  # not exact but close enuf 
   b = np.zeros_like(np.float32(an_image))
# the section is y = 0, x,z
   rout = int(radius/max(the_metric.fa,the_metric.fc))
   bout = int(blur_radius/the_metric.fb)
   for ix in range(-rout,rout):
      ixu = ix
      if ixu < 0:
         ixu += an_image.shape[0]
      if ixu >= an_image.shape[0]:
         ixu -= an_image.shape[0]
      xr = float(ix)
      for iz in range(-rout,rout):
         izu = iz
         if izu < 0:
            izu += an_image.shape[2]
         if izu >= an_image.shape[2]:
            izu -= an_image.shape[2]
         zr = float(iz)*the_metric.fa/the_metric.fc
         r = sqrt( xr*xr + zr*zr)
         print(ix,iz, xr,zr,r) 
         sys.stdout.flush()
         if r > radius:
           continue
         if r > 0 :
            theta = acos(xr/r)
         else:
            theta = 0.
         if theta < ha:
               b[(ixu,0,izu)] = 1.
               for iy in range(-bout,bout):
                 iyu = iy
                 if iyu < 0:
                    iyu += an_image.shape[1]
                 b[(ixu,iyu,izu)] = 1.
#
#  the FFT will scale by sqrt(size)
#  the ACF uses size, but that's because it uses
#  two FFTs so it's already sqrt(size) bigger
#
   x = float(an_image.size)
   x = sqrt(x)
   b = b.__idiv__(x)
   return b

def conic_compliment( an_image,sign, half_angle, radius, blur_radius):
#   print( the_metric.fa, the_metric.fb,the_metric.fc)
#   sys.stdout.flush()
   ha = half_angle*3.14159265/180.  # not exact but close enuf 
   b = np.zeros_like(np.float32(an_image))
# the section is y = 0, x,z
   rout = int(radius/max(the_metric.fa,the_metric.fc))
   bout = int(blur_radius/the_metric.fb)
   for ix in range(-rout,rout):
      ixu = ix
      if ixu < 0:
         ixu += an_image.shape[0]
      if ixu >= an_image.shape[0]:
         ixu -= an_image.shape[0]
      xr = float(ix)
      for iz in range(-rout,rout):
         izu = iz
         if izu < 0:
            izu += an_image.shape[2]
         if izu >= an_image.shape[2]:
            izu -= an_image.shape[2]
         zr = float(iz)*the_metric.fa/the_metric.fc
         r = sqrt( xr*xr + zr*zr)
         print(ix,iz, xr,zr,r) 
         sys.stdout.flush()
         if r > radius:
           continue
         if r > 0 :
            theta = acos(xr/r)
         else:
            theta = 0.
         if theta > ha:
               b[(ixu,0,izu)] = sign 
               for iy in range(-bout,bout):
                 iyu = iy
                 if iyu < 0:
                    iyu += an_image.shape[1]
                 b[(ixu,iyu,izu)] = sign
#
#  the FFT will scale by sqrt(size)
#  the ACF uses size, but that's because it uses
#  two FFTs so it's already sqrt(size) bigger
#
   x = float(an_image.size)
   x = sqrt(x)
   b = b.__idiv__(x)
   return b

              
def conic_defocus( an_image, half_angle, radius, blur_radius,alpha,beta):
#   print( the_metric.fa, the_metric.fb,the_metric.fc)
#   sys.stdout.flush()
   ha = half_angle*3.14159265/180.  # not exact but close enuf 
   b = np.zeros_like(np.float32(an_image))
# the section is y = 0, x,z
   rout = int(radius/max(the_metric.fa,the_metric.fc))
   bout = int(blur_radius/the_metric.fb)
   rmax = radius*radius
   bmax = blur_radius/the_metric.fb
   bmax = bmax*bmax
   for ix in range(-rout,rout):
      ixu = ix
      if ixu < 0:
         ixu += an_image.shape[0]
      if ixu >= an_image.shape[0]:
         ixu -= an_image.shape[0]
      xr = float(ix)
      for iz in range(-rout,rout):
         izu = iz
         if izu < 0:
            izu += an_image.shape[2]
         if izu >= an_image.shape[2]:
            izu -= an_image.shape[2]
         zr = float(iz)*the_metric.fa/the_metric.fc
         r = sqrt( xr*xr + zr*zr)
         print(ix,iz, xr,zr,r) 
         sys.stdout.flush()
         if r > radius:
           continue
         if r > 0 :
            theta = acos(xr/r)
         else:
            theta = 0.
         if theta < ha:
               b[(ixu,0,izu)] = 1. +alpha*(r*r/rmax) 
               for iy in range(-bout,bout):
                 iyu = iy
                 if iyu < 0:
                    iyu += an_image.shape[1]
                 fy = float(iy)
                 b[(ixu,iyu,izu)] = 1. + alpha*(r*r/rmax) + beta*(fy*fy/bmax)
#
#  the FFT will scale by sqrt(size)
#  the ACF uses size, but that's because it uses
#  two FFTs so it's already sqrt(size) bigger
#
   x = float(an_image.size)
   x = sqrt(x)
   b = b.__idiv__(x)
   return b

              
def conic_compliment_defocus( an_image,sign, half_angle, radius, blur_radius,alpha,beta):
#   print( the_metric.fa, the_metric.fb,the_metric.fc)
#   sys.stdout.flush()
   ha = half_angle*3.14159265/180.  # not exact but close enuf 
   b = np.zeros_like(np.float32(an_image))
# the section is y = 0, x,z
   rout = int(radius/max(the_metric.fa,the_metric.fc))
   bout = int(blur_radius/the_metric.fb)
   rmax = radius*radius
   bmax = blur_radius/the_metric.fb
   bmax = bmax*bmax
   for ix in range(-rout,rout):
      ixu = ix
      if ixu < 0:
         ixu += an_image.shape[0]
      if ixu >= an_image.shape[0]:
         ixu -= an_image.shape[0]
      xr = float(ix)
      for iz in range(-rout,rout):
         izu = iz
         if izu < 0:
            izu += an_image.shape[2]
         if izu >= an_image.shape[2]:
            izu -= an_image.shape[2]
         zr = float(iz)*the_metric.fa/the_metric.fc
         r = sqrt( xr*xr + zr*zr)
         print(ix,iz, xr,zr,r) 
         sys.stdout.flush()
         if r > radius:
           continue
         if r > 0 :
            theta = acos(xr/r)
         else:
            theta = 0.
         if theta > ha:
               b[(ixu,0,izu)] =sign*( 1. +alpha*(r*r/rmax) )
               for iy in range(-bout,bout):
                 iyu = iy
                 if iyu < 0:
                    iyu += an_image.shape[1]
                 fy = float(iy)
                 b[(ixu,iyu,izu)] = sign*(1. + alpha*(r*r/rmax) + beta*(fy*fy/bmax))
#
#  the FFT will scale by sqrt(size)
#  the ACF uses size, but that's because it uses
#  two FFTs so it's already sqrt(size) bigger
#
   x = float(an_image.size)
   x = sqrt(x)
   b = b.__idiv__(x)
   return b

              
#Gaussian blur
def gaussian(an_image, radius):
   sys.stdout.flush()
   rout = int(radius*2*max( the_metric.fa,the_metric.fb,the_metric.fc))
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
   routx = int(rx*2*the_metric.fa)
   routy = int(ry*2*the_metric.fb)
   routz = int(rz*2*the_metric.fc)
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
   rout = int(radius*2/max(the_metric.fa,the_metric.fb,the_metric.fc))
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
          jr = float(j)*the_metric.fb/the_metric.fa
          ju = j
          if ju < 0:
            ju += an_image.shape[1]
          for k in range(-rout,rout):
            kr = float(k)*the_metric.fc/the_metric.fa
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

