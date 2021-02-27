# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 13:50:26 2016

@author: Kimberly
"""

import logging

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Kmeans:

    def __init__(self):
        mat = sp.io.loadmat( "./Data/ex7data2.mat" )
    
    def run(self):
        X = mat['X']
        K = 3

        initial_centroids = array([[3, 3], [6, 2], [8, 5]])

# Start main program
if __name__ == "__main__":
    inst = Kmeans("./Data/ex7data2.mat")
    inst.run()
    