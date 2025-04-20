# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 18:12:24 2021

@author: home
"""

import logging
import doLogging

import numpy as np
import scipy.misc, scipy.io, scipy.optimize
import imageio
from PIL import Image

from matplotlib import pyplot, cm, colors, lines

class Utils:
    def __init__(self):
        return 

    @staticmethod    
    def loadMatlabData(fname) -> np.array:
        mat = scipy.io.loadmat(fname)
        X = mat['X']
        return X

    @staticmethod
    def loadImageData(fname) -> np.array:
        A = imageio.imread(fname)
        logging.info(f"Size of data {A.shape}")
        A = A / 255.0
        img_size = np.shape( A )
        X = A.reshape(img_size[0] * img_size[1], 3)
        return X

    # Normalize vectore by subtracting the mean and dividing by the standard deviation
    # return the normalized data, mean, and std    
    @staticmethod
    def featureNormalize(data):
        print (f"shape of data is {data.shape}")
        u = np.mean(data,axis=0) # take mean of each column
        print (f"mean vec is {u}")
        X_zero_mean = data - u
        X_std = np.std(X_zero_mean,axis=0,ddof=1) # compute variance of each column
        print (f"standard deviation is {X_std}")
        X_norm = X_zero_mean / X_std

        return X_norm, u, X_std

    def test(self):
           logging.info(f"Utils test Completed Successfully")

if __name__ == "__main__":
    doLogging.set_up_logger()
    A = Utils()
    A.test()   
    doLogging.clean_up_logger()
    