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

    # Returns R which is an indicator function of whether the movie rating is valid
    # Y which are the movie ratings for each movie
    # Both R and Y are [number of movies, number of users] [1682,943]
    def loadMatlabData(self,fname) -> Tuple[np.array,np.array]:
        mat = scipy.io.loadmat(fname)
        Y = mat['Y']
        R = mat['R']
        logging.info(f"Ratings are {Y.shape[0]} movies and {Y.shape[1]} users")
        return Y,R

    # Returns X which is the feature vector for each movie
    # theta which is the user parameter vector
    # X is [number of movies, embedding dimension] [1682,10]
    # theta is [number of users, embedding dimension] [843,10]
    # Y = X * theta^T where Y is the user-movie rating matrix
    def loadMatlabParameterData(self,fname):
        mat = scipy.io.loadmat(fname)
        num_features = mat['num_features']
        num_users 	 = mat['num_users']
        num_movies 	 = mat['num_movies']
        logging.info(f"Data has {num_movies} movies {num_users} users with {num_features} dimensional embedding space")
        X 			 = mat['X']
        theta 		 = mat['Theta']
        return X,theta

    def loadMovieList(self):
        movies = {}
        counter = 0
        with open('Data/movie_ids.txt', 'r') as f:
            contents = f.readlines()
            for content in contents:
                movies[counter] = content.strip().split(' ', 1)[1]
                counter += 1

        print ("found %d movies in movie_ids.txt" % counter)
        return movies

    def normalizeRatings(self, Y, R ):
        m = np.shape( Y )[0]
        Y_mean = np.zeros((m, 1))
        Y_norm = np.zeros( np.shape( Y ) )

        for i in range( 0, m ):
            idx 			= np.where( R[i] == 1 )
            Y_mean[i] 		= np.mean( Y[i, idx] )
            Y_norm[i, idx] 	= Y[i, idx] - Y_mean[i]

        return Y_norm, Y_mean

    # Converts params, the concatenated vectors [theta X], back to matrices
    # theta are the user parameter vectors [num users, embedding dimension] and
    # X are the movie parameter vectors [num movies. embedding dimension]
    def unrollParams( self, params, num_users, num_movies, num_features ) -> Tuple[np.array,np.array]:
        X 		= params[:num_movies * num_features]
        X 		= X.reshape( (num_features, num_movies) ).transpose()
        theta 	= params[num_movies * num_features:]
        theta 	= theta.reshape( num_features, num_users ).transpose()

        return X, theta
    
    # Given a one-dimensional paramater vector, compute the reguralized cost function
    # Returns the scalar cost function
    def cofiCostFunc( self, params, Y, R, num_users, num_movies, num_features, lamda ) -> float:
        X, theta 	   = self.unrollParams( params, num_users, num_movies, num_features  )
        J 			   = 0.5 * np.sum( (X.dot( theta.T ) * R - Y) ** 2 )
        regularization = 0.5 * lamda * (np.sum( theta**2 ) + np.sum(X**2))
        
        return J + regularization

    # Given a one-dimensional parameter vector, compute the regularized movie and user gradients
    # Return the partials with respect to movie and user parameters
    def cofiGradFunc( self, params, Y, R, num_users, num_movies, num_features, lamda ) -> Tuple[np.array,np.array]:
        X, theta 	= self.unrollParams( params, num_users, num_movies, num_features )
        inner 		= X.dot( theta.T ) * R - Y
        X_grad 		= inner.dot( theta ) + lamda * X
        theta_grad 	= inner.T.dot( X ) + lamda * theta
        
        return np.r_[X_grad.T.flatten(), theta_grad.T.flatten()]

    def run(self)-> None:
        Y,R = self.loadMatlabData('Data/ex8_movies.mat')
        print (f"Average movie rating {np.mean( np.extract ( Y[0,:] * R[0,:] > 0, Y[0, :] ) ) }" )
        X,theta = self.loadMatlabParameterData('Data/ex8_movieParams.mat')
        # Use a subset of the data for testing parameter unrolling and cost and gradient computations
        num_users    = 4
        num_features = 3
        num_movies 	 = 5
        
        X 		= X[:num_movies, :num_features]
        theta 	= theta[:num_users, :num_features]
        Y 		= Y[:num_movies, :num_users]
        R 		= R[:num_movies, :num_users]

        # Test out flatening and unrolling parameters
        params = np.r_[X.T.flatten(), theta.T.flatten()]
        print (f"Unregularized cost {self.cofiCostFunc( params, Y, R, num_users, num_movies, num_features, 0)}")
        print (f"Unregularized gradient {self.cofiGradFunc( params, Y, R, num_users, num_movies, num_features, 0)}")
        print (f"Regularized cost {self.cofiCostFunc( params, Y, R, num_users, num_movies, num_features, 1.5)}")
        print (f"Regularized gradient {self.cofiGradFunc( params, Y, R, num_users, num_movies, num_features, 1.5)}")

    def part2_3(self):
        movies = self.loadMovieList()

        # Give ratings to some movies
        my_ratings = np.zeros((1682, 1))
        my_ratings[0] = 4
        my_ratings[97] = 2
        my_ratings[6]  = 3
        my_ratings[11] = 5
        my_ratings[53] = 4
        my_ratings[63] = 5
        my_ratings[65] = 3
        my_ratings[68] = 5
        my_ratings[182] = 4
        my_ratings[225] = 5
        my_ratings[354] = 5
        
        # Read user-movie data
        mat = scipy.io.loadmat('Data/ex8_movies.mat')
        Y, R = mat['Y'], mat['R']

        # Prepend my_ratings by column stacking
        Y = np.c_[my_ratings, Y]
        R = np.c_[my_ratings > 0, R]

        Y_norm, Y_mean = self.normalizeRatings( Y, R )

        num_movies, num_users = np.shape( Y )
        num_features = 10

        # initialize user and movie embeddings
        X 		= np.random.randn( num_movies, num_features )
        theta 	= np.random.randn( num_users, num_features )
        # create parameter vector by stacking rows
        initial_params = np.r_[X.T.flatten(), theta.T.flatten()]
        lamda = 10.0

        result = scipy.optimize.fmin_cg( self.cofiCostFunc, fprime=self.cofiGradFunc, x0=initial_params, \
                                        args=( Y, R, num_users, num_movies, num_features, lamda ), \
                                        maxiter=200, disp=True, full_output=True )
        J, params = result[1], result[0]

        X, theta = self.unrollParams( params, num_users, num_movies, num_features )
        prediction = X.dot( theta.T )

        # Add mean movie rating to normalized predictions
        my_prediction = prediction[:, 0:1] + Y_mean
        
        idx = my_prediction.argsort(axis=0)[::-1]
        my_prediction = my_prediction[idx]

        for i in range(0, 20):
            j = idx[i, 0]
            print ("Predicting rating %.1f for movie %d named %s" % (my_prediction[j], j, movies[j]))

# Start main program
if __name__ == "__main__":
    doLogging.set_up_logger()
    inst = CollaborativeFilter()
    #inst.run()
    inst.part2_3()
    doLogging.clean_up_logger()