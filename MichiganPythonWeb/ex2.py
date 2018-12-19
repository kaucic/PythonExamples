# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 20:46:40 2016

@author: Kimberly
"""

import re

def sum_numbers(txt):
    nums = re.findall('[0-9]+',txt)   
    return sum([int(num) for num in nums])
    
# Start of main program
with open("regex_sum_42.txt") as fd1:
    txt = fd1.read()
    print "sum1 is", sum_numbers(txt)

fd2 = open("regex_sum_251828.txt")
txt = fd2.read()
print "sum2 is", sum_numbers(txt)
fd2.close()
