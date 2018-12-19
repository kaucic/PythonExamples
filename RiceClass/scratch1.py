# -*- coding: utf-8 -*-
"""
Created on Wed Sep 09 20:15:02 2015

@author: Kimberly
"""

import math

def f(x):
        return (-5.0 * x**5 + 69 * x**2 - 47)
        
print f(0), f(1), f(2), f(3)

def future_value(present_value, annual_rate, periods_per_year, years):
    rate_per_period = annual_rate / periods_per_year
    periods = periods_per_year * years
    fv = present_value * (1+rate_per_period)**periods
    return (fv)
    
print future_value(500, .04, 10, 10)
print future_value(1000, .02, 365, 3)

def compute_polygon_area(nsides,length):
    return (0.25 * nsides * length*length / math.tan(math.pi/nsides) )

print compute_polygon_area(5,7)
print compute_polygon_area(7,3)

def project_to_distance(point_x, point_y, distance):
    dist_to_origin = math.sqrt(point_x ** 2 + point_y ** 2)    
    scale = distance / dist_to_origin
    print point_x * scale, point_y * scale
    
project_to_distance(2, 7, 4)