# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:40:57 2015

@author: Kimberly
"""

import rpy2.robjects as ro

ro.r('x=c()')
ro.r('x[1]=22')
ro.r('x[2]=44')
pyx = (ro.r('x'))

print "len(pyx)=", len(pyx)
print "values=", pyx


