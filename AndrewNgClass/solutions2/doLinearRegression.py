# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:43:54 2015

@author: Robert Kaucic
"""

import numpy as np
import scipy as sp
from scipy import optimize,special
import matplotlib.pyplot as plt

alpha = 0.03
iters = 30000
LAMB = 0.01

# Determine which DS to use
DS = 2
if (DS == 1):
    FILENAME = "./Data/ex1data1.txt"
else:
    FILENAME = "./Data/ex1data2.txt"

def readFile(fname):
    data = np.genfromtxt(fname,delimiter=",")
    return (data)
    
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
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')

def plotFit(theta,x,y):
    plotData(x[:,1],y)
    xs = np.array( [ np.min(x[:,1]), np.max(x[:,1]) ] )
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

    
# Start main program    
vals = readFile(FILENAME)
m,n = np.shape(vals)
print "number of samples = %d number of parameters = %d" % (m,n)

if (0):
    scaledX = vals[:,:-1]
    scales = np.ones((n-1,))
else:
    scaledX, scales = normalizeVals(vals[:,:-1])
    
print "scales=", scales
#print "scaledX=", scaledX

# append bias = 1 to the sample vectors
X = np.c_[np.ones((m,1)), scaledX]
Y = vals[:,-1]

#plotData(X[:,1],Y)

params = np.zeros((n,))

# test out functions
h = hypothesis(params,X)
initial_cost = computeCost(params,X,Y,0.1)
print "Initial cost=", initial_cost

# Look at using various optimization routines including Gradient Descent 
if (1):
    # run Gradient Descent algorithm
    Theta, cost = gradientDescent(params,X,Y,LAMB,alpha,iters)
    print "starting cost = %f final cost = %f" % (cost[0],cost[-1])
    print "final theta=", Theta
    plotCost(cost)
else:
    # run fminunc or L-BFGS-B
    Theta, cost = findMinTheta(params,X,Y,LAMB,iters)
    print "final cost = %f" % cost
    print "final theta=", Theta

# compute closed form solution
reg_matrix = LAMB * np.identity(n)
reg_matrix[0,0] = 0.0
exact = np.linalg.inv( X.T.dot(X) + reg_matrix ).dot(X.T).dot(Y)
print "exact theta=", exact

# predict new values for test data
if (n == 2):
    plotFit(Theta,X,Y)
    x1 = np.array([1.0, 3.5])
    x1[1:] = x1[1:] / scales # exclude intercept unit
    predict1 = hypothesis(Theta,x1) 
    print "predict1=", predict1
    
    x2 = np.array([1.0, 7.0])
    x2[1:] = x2[1:] / scales 
    predict2 = hypothesis(Theta,x2) 
    print "predict2=", predict2
else:
    # 1650 sq feet 3 bedroom house
    x3 = np.array([1.0, 1650.0, 3.0])   
    x3[1:] = x3[1:] / scales # exclude intercept unit
    predict3 = hypothesis(Theta,x3) 
    print "predict3=", predict3
 