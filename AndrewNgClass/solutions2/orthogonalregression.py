# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:43:54 2015

@author: Robert Kaucic
"""

import numpy as np
import scipy as sp
import random
from scipy import optimize,special
import matplotlib.pyplot as plt

LAMB = 0.01
 
def plotCost(cost):
    i = np.arange(len(cost))
    plt.figure()
    plt.plot(i,cost,'-b' )
    plt.title("Cost Function")
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()    
    
def plotData(x,y):
    plt.figure()
    plt.plot(x,y,'rx',markersize=5 )
    plt.ylabel('Actual wind speed')
    plt.xlabel('Forecast wind speed')

def plotFit(theta,X,y):
    plotData(X[:,1],y)
    xs = np.array( [ np.min(X[:,1]), np.max(X[:,1]) ] )
    ys = theta[1] * xs + theta[0]
    plt.plot(xs,ys)
    plt.show()
    
def normalizeVals(x):
    """
    Scale values by subtracting the mean and dividing by half the range 
    to get data in the range [-1,1]    
    Save mean and scale factors for reuse in testing data
    With simple scaling, RNN weights should be able to undo scaling
    """
    maxvals = x.max(axis=0).astype(float)
    minvals = np.absolute(x.min(axis=0)).astype(float)
    scale_factors = np.maximum(maxvals,minvals)
    # replace 0 scale factors with 1 to prevent dividing by 0
    scale_factors[scale_factors==0] = 1.0 
    nx = np.zeros(x.shape)
    for i in range(len(scale_factors)):
        nx[:,i] = x[:,i] / scale_factors[i]
    return nx, scale_factors

def hypothesis(theta,x):
    """
    x is (m x n) matrix of samples
    y is (n x 1) vector of targets
    theta is (n x 1) vector of parameters
    h is (m x 1) vector of predictions
    """
    h = x.dot(theta)  
    return h
    
def computeCost(theta,x,y,lamb):
    """
    Cost = 0.5/m * (h(theta*x) - y)**2 + 0.5*lambda/m * theta^2
    """
    m = float(len(y))
    err = hypothesis(theta,x) - y
    # add regularization term;  skip bias term, i.e. theta when j=0
    regularization = lamb * np.sum(theta[1:]**2)
    cost = 0.5 / m * (np.sum(err**2) + regularization)
    return cost

def computeGradient(theta,x,y,lamb):
    """
    grad = 1/m * (h(theta*x) - y) * x + lambda/m * theta 
    """
    m = float(len(y))
    err = hypothesis(theta,x) - y
    grad = x.T.dot(err) / m
    # add regularization term;  skip bias term, i.e. theta when j=0
    grad[1:] += (lamb/m * theta[1:])    
    return grad

def costAndGradientFunction(theta,x,y,lamb):
    cost = computeCost(theta,x,y,lamb)
    grad = computeGradient(theta,x,y,lamb)    
    return cost,grad

def gradientDescent(orig_theta,x,y,lamb,alpha,num_iter):
    """
    Input
        orig_theta is (n x 1) vector of linear regression coefficients
        x is (m x n) matrix of samples
        y is (n x 1) vector of targets        
        alpha is the learning rate
    Output
        theta is (n x 1) final solution
        cost is array of resultant costs as a function of iteration
    """
    theta = np.copy(orig_theta)
    cost = np.zeros((num_iter,))
    for i in range(num_iter):
        grad = computeGradient(theta,x,y,lamb)
        theta -= (alpha * grad)
        cost[i] = computeCost(theta,x,y,lamb)
        #print "Iter = %d cost = %f" % (i,cost[i])
        #print "theta=", theta
    return theta,cost

def findMinTheta(orig_theta,x,y,lamb,num_iter):
    """
    Input
        orig_theta is (n x 1) vector of linear regression coefficients
        x is (m x n) matrix of samples
        y is (n x 1) vector of targets
    Output
        theta is (n x 1) final solution
        cost is the resultant cost
    """
    theta = np.copy(orig_theta)
    result = sp.optimize.minimize(costAndGradientFunction,x0=theta,args=(x,y,lamb),method='L-BFGS-B',jac=True,options={'maxiter': num_iter})  
    print "success=", result.success    
    #print result
    return result.x, result.fun

class OrthogRegress:
    def solve(self,a,b):
        xt = np.arange(0,10,1)
        rx = np.random.normal(loc=0.0,scale=0.01,size=np.shape(xt))
        ry = np.random.normal(loc=0.0,scale=0.01,size=np.shape(xt))
        x = xt + rx   
        y = a * xt + b + ry
        #plotData(x,y)

        z = np.c_[x, y]
        U,S,V = np.linalg.svd(z)
        #print "U=", U
        print "S=", S
        print "V=", V
        VXY = V[0,1]
        VYY = V[1,1]
        B = -VXY/VYY
        print "B=", B
        
        # append bias = 1 to the sample vectors
        m = len(x)
        X = np.c_[np.ones((m,1)), x]
        theta = self.normal_equation(X,y)
        plotFit(theta,X,y)
        return theta

    def normal_equation(self,X,y):
        m,n = np.shape(X)
        # compute closed form solution
        reg_matrix = LAMB * np.identity(n)
        reg_matrix[0,0] = 0.0
        exact = np.linalg.inv( X.T.dot(X) + reg_matrix ).dot(X.T).dot(y)
        print "algebretic fit=", exact
        return exact
 
if __name__ == '__main__':
    regress = OrthogRegress()
    ans = regress.solve(0.6,0)
    print 'ans=', ans
