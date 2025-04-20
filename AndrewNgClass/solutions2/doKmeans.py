# -*- coding: utf-8 -*-
"""
Created on Sun Jun 06 13:50:26 2021

@author: home
"""

import logging
import doLogging

import numpy as np
import scipy.io as spio
import imageio

import matplotlib.pyplot as plt

class Kmeans:
      
    def __init__(self, niters ):
        self._niters = niters

    def loadMatlabData(self, fname, k ) -> np.array:
        mat = spio.loadmat(fname)
        self._X = mat['X']
        self._K = k
        logging.info(f"Size of data {self._X.shape}")
        return self._X
        
    def loadImageData(self, fname, k) -> None:
        A = imageio.imread(fname)
        self.plotImage(A)
        A = A / 255.0
        self._img_size = np.shape( A )
        self._X = A.reshape( self._img_size[0] * self._img_size[1], 3 )
        self._K = k
        logging.info(f"Size of data {self._X.shape}")
        
    def convertToImage(self, X, idx, centroids) -> np.array:
        # mapping the centroids back to compressed image,
        # e.g. all pixels in that cluster shares the same color as the centroid
        m = np.shape( X )[0]
        X_recovered = np.zeros( np.shape(X) )
        for i in range( 0, m ):
            k = int(idx[i])
            X_recovered[i] = centroids[k]
        
        A_recovered = X_recovered.reshape( self._img_size[0], self._img_size[1], 3 )
        
        return A_recovered

    def initCentroids(self, X, K ) -> np.array:
        return np.random.permutation( X )[:K]

    def findClosestCentroids(self,  X, centroids ) -> np.array:
        #func_name = inspect.stack()[0][3]
        #logging.info(f"In method {func_name}")
        
        K 	= np.shape( centroids )[0]
        m   = np.shape( X )[0]
        idx = np.zeros( (m,) )
    
        for i in range(0, m):
            lowest 		 = 1e10
            lowest_index = 0
            
            for k in range( 0, K ):
                cost = X[i] - centroids[k]
                cost = cost.T.dot( cost )
                if cost < lowest:
                    lowest_index = k
                    lowest 		 = cost
                    
            idx[i] = lowest_index
            
        return idx

    def computeCentroids(self, X, idx, K ) -> np.array:
        #func_name = inspect.stack()[0][3]
        #logging.info(f"In method {func_name}")
        
        m, n = np.shape( X )	
        centroids = np.zeros((K, n))
        
        for k in range(0,K):
            indices = np.where(idx == k) # quickly extract X that falls into the cluster
            samps = X[indices]
            count = np.shape(samps)[0] # count number of entries for that cluster
            centroids[k] = sum(samps) / count
            
        return centroids

    def run(self) -> None:
        #centroids = np.array([[3, 3], [6, 2], [8, 5]])
        centroids = self.initCentroids( self._X, self._K)
        logging.info(f"Initial centroids {centroids}")
        for i in range(0,self._niters):
            idx = self.findClosestCentroids( self._X, centroids )
            centroids = self.computeCentroids( self._X, idx, self._K )
        logging.info(f"Final centroids {centroids}")

    def run_image(self) -> None:
        centroids = self.initCentroids( self._X, self._K)
        logging.info(f"Initial centroids {centroids}")
        for i in range(0,self._niters):
            idx = self.findClosestCentroids( self._X, centroids )
            centroids = self.computeCentroids( self._X, idx, self._K )
        logging.info(f"Final centroids {centroids}")
        compressed_image = self.convertToImage(self._X, idx, centroids)
        self.plotImage(compressed_image)
          
    def test(self) -> None:
        centroids = np.array([[3, 3], [6, 2], [8, 5]])
        idx = self.findClosestCentroids( self._X, centroids )
        logging.info(f"Cluster numbers {idx[0:3]} should be [0,2,1]") # should be [0,2,1]
        centroids = self.computeCentroids( self._X, idx, self._K )
        logging.info(f"Final centroids {centroids}")
        #should be
        # [[ 2.428301  3.157924]
        #  [ 5.813503  2.633656]
        #  [ 7.119387  3.616684]]
        
    def plotImage(self, A) -> None:
        # shows the image
        axes = plt.gca()
        figure = plt.gcf()
        axes.imshow( A )
        plt.show(  )
        
# Start main program
if __name__ == "__main__":
    doLogging.set_up_logger()
    inst = Kmeans(10)
    X = inst.loadMatlabData("./Data/ex7data2.mat",3)
    #inst.test()
    inst.run()
    inst.loadImageData("./Data/bird_small.png", 16)
    inst.run_image()
    doLogging.clean_up_logger()

    