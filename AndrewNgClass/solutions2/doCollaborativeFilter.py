# -*- coding: utf-8 -*-
"""
Created on Sun Jan 2 18:12:24 2022

@author: home
"""

from typing import Tuple

import logging
import doLogging

import numpy as np
import scipy.misc, scipy.io, scipy.optimize

class CollaborativeFilter:
    def __init__(self):
        return

    def loadMatlabData(self,fname) -> Tuple[np.array,np.array]:
        mat = scipy.io.loadmat(fname)
        Y = mat['Y']
        R = mat['R']
        return Y,R

    def unrollParams( self, params, num_users, num_movies, num_features ):
        X 		= params[:num_movies * num_features]
        X 		= X.reshape( (num_features, num_movies) ).transpose()
        theta 	= params[num_movies * num_features:]
        theta 	= theta.reshape( num_features, num_users ).transpose()

        return X, theta
    
    def cofiCostFunc( self, params, Y, R, num_users, num_movies, num_features, lamda ) -> float:
        X, theta 	   = unrollParams( params, num_users, num_movies, num_features  )
        J 			   = 0.5 * sum( (X.dot( theta.T ) * R - Y) ** 2 )
        regularization = 0.5 * lamda * (sum( theta**2 ) + sum(X**2))
        
        return J + regularization

    def run(self)-> None:
        Y,R = self.loadMatlabData('Data/ex8_movies.mat')
        print (f"Average movie rating {np.mean( np.extract ( Y[0,:] * R[0,:] > 0, Y[0, :] ) ) }" )
        print (f"Completed Successfully")

# Start main program
if __name__ == "__main__":
    doLogging.set_up_logger()
    inst = CollaborativeFilter()
    inst.run()
    doLogging.clean_up_logger()