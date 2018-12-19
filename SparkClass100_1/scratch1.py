# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:19:05 2015

@author: Kimberly
"""

words = [('cat', [1, 4]), ('elephant', [1]), ('rat', [1, 1])]
print words
counts = map(lambda (k,v): (k,sum(v)), words)
print counts

