# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 13:50:26 2016

@author: Kimberly
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# Start main prograrm
mat = sp.io.loadmat( "./Data/ex7data2.mat" )
X = mat['X']
K = 3

initial_centroids = array([[3, 3], [6, 2], [8, 5]])
