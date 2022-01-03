# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 18:12:24 2021

@author: home
"""

import logging
import doLogging

import numpy as np
import scipy.misc, scipy.io, scipy.optimize
#import scipy.io as spio
import imageio
from PIL import Image

from matplotlib import pyplot, cm, colors, lines
#import matplotlib.pyplot as plt

from AndrewNgUtils import Utils

class PCA:
    def __init__(self):
        return   

    # Compute the principal components of a Matrix
    def pca(self,data):
        # Remove the mean and scale to N(0,1) before doing PCA
        X,mu,X_std = Utils.featureNormalize(data)
        X_cov = X.T.dot(X) / X.shape[0]
        U, S, V = np.linalg.svd(X_cov)
        print (f"PCA vecs are {U}")
        print (f"Singular values are {S}")

        return X,mu,X_std,U,S

    def projectData(self,X,project,dims) -> np.array:
        X_low_dim = X.dot(project)[:,:dims]
        return X_low_dim

    def recoverData(self,X_low_dim,project,dims) -> np.array:
        X_recovered = X_low_dim.dot(project[:,:dims].T)
        return X_recovered

    def run(self) -> None:
        X = Utils.loadMatlabData("./Data/ex7data1.mat")
        print (f"first sample is {X[0]}")
        #self.plot2DData(X)
        X_norm,mu,X_std,U,S = self.pca(X)
        print (f"first normalized sample is {X_norm[0]}")
        self.plotPCAAxes(X,mu,U,S)
        X_low_dim = self.projectData(X_norm,U,1)
        print (f"projected data is {X_low_dim[0]}") # Should be 1.481
        X_recovered = self.recoverData(X_low_dim,U,1)
        print (f"recovered data is {X_recovered[0]}") # Should be [-1.047 -1.047]
        self.plotProjectedData(X_norm,X_recovered)
        img = Utils.loadMatlabData("./Data/ex7faces.mat")
        X_norm,mu,X_std,U,S = self.pca(img)
        # Scale from norm 1 PCA vector to [0,1]
        #self.displayImageData(U[:,:36].T * U.shape[0])
        K = 100
        Z = self.projectData(X_norm,U,K)
        X_rec = self.recoverData(Z,U,K)
        # Scale from normalized data to [0,1] for image display
        self.displayOrigAndReduced((X_norm + 2)*0.25,(X_rec + 2)*0.25,K)
        img2 = Utils.loadImageData("./Data/bird_small.png")
        X_norm,mu,X_std,U,S = self.pca(img2)
        Z2 = self.projectData(X_norm,U,K)
        self.plotScatter(Z2,K)
        logging.info(f"doPCA run Completed Successfully")

    def plot2DData(self,X) -> None:
        pyplot.plot( X[:, 0], X[:, 1], 'bo' )
        pyplot.axis( [0.5, 6.5, 2, 8] )
        pyplot.axis( 'equal' )
        pyplot.show(  )

    def plotPCAAxes(self,X,mu,U,S) -> None:
        mu = mu.reshape( 1, 2)[0]
        mu_1 = mu + 1.5 * S[0] * U[:, 0]
        mu_2 = mu + 1.5 * S[1] * U[:, 1]
        
        pyplot.plot( X[:, 0], X[:, 1], 'bo' )
        pyplot.gca().add_line( lines.Line2D( xdata=[mu[0], mu_1[0]], ydata=[mu[1], mu_1[1]], c='r', lw=2 ) )	
        pyplot.gca().add_line( lines.Line2D( xdata=[mu[0], mu_2[0]], ydata=[mu[1], mu_2[1]], c='r', lw=2 ) )	
        pyplot.axis( [0.5, 6.5, 2, 8] )
        pyplot.axis( 'equal' )
        pyplot.show(  )

    def plotProjectedData(self,X_norm,X_rec) -> None:
        for i in range( 0, np.shape( X_rec)[0] ):
            pyplot.gca().add_line( lines.Line2D( xdata=[X_norm[i,0], X_rec[i,0]], ydata=[X_norm[i,1], X_rec[i,1]], c='g', lw=1, ls='--' ) )	
            
        pyplot.plot( X_norm[:, 0], X_norm[:, 1], 'bo' )
        pyplot.plot( X_rec[:, 0], X_rec[:, 1], 'ro' )
        pyplot.axis( 'equal' )
        pyplot.axis( [-4, 3, -4, 3] )
        pyplot.show(  )

    def plotScatter(self,X_2dim,K) -> None:
        pyplot.scatter( X_2dim[:K, 0], X_2dim[:K, 1], c='r', marker='o' )
        pyplot.show()

    def displayImageData(self,X) -> None:
        width = 32
        rows = cols = int(np.sqrt( np.shape(X)[0] ))
        out = np.zeros(( width * rows, width * cols ))
        
        counter = 0
        for y in range(0, rows):
            for x in range(0, cols):
                start_x = x * width
                start_y = y * width
                out[start_x:start_x+width, start_y:start_y+width] = X[counter].reshape( width, width ).T * 255
                counter += 1
                
        #img = scipy.misc.toimage( out )
        img = Image.fromarray( out )
        axes 	= pyplot.gca()
        figure 	= pyplot.gcf()
        axes.imshow( img ).set_cmap( 'gray' )
        return

    def displayOrigAndReduced(self,X_norm,X_rec,K) -> None:
        pyplot.subplot( 1, 2, 1 )
        self.displayImageData( X_norm[:K, :] )
        pyplot.subplot( 1, 2, 2 )
        self.displayImageData( X_rec[:K, :] )
        pyplot.show(  )

if __name__ == "__main__":
    doLogging.set_up_logger()
    A = PCA()
    A.run()   
    doLogging.clean_up_logger()
    