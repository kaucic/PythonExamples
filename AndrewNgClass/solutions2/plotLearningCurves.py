# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:43:54 2015

@author: Robert Kaucic
"""

import logging

import numpy as np
import scipy as sp
from scipy import optimize,special
import matplotlib.pyplot as plt

poly = True
DEGREE = 8

alpha = 0.001
iters = 30000
LAMB = 0.0

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
    plt.xlabel('Change in water level(x)')
    plt.ylabel('Water flowing out of the dam(y)')

def plotFit(theta,x,y):
    plotData(x[:,1]*scales[0],y)
    #xs = np.arange(np.min(x[:,1]), np.max(x[:,1]), 0.05)
    xs = np.arange(-60,50,1.0)
    m = len(xs)
    if poly:
        pr = polyFeatures(xs,DEGREE)
        ps = pr / scales
        xb = np.c_[np.ones((m,1)), ps]
    else:
        xb = np.c_[np.ones((m,1)), xs/scales[0]]
    ys = hypothesis(theta,xb)
    plt.plot(xs,ys,linestyle='--',linewidth=1)
    plt.show()

def plotErrors(err1,err2):
    i = np.arange(len(err1))
    plt.figure()
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.ylim([-2, 100])
    plt.xlim([0, 13])
    plt.plot(i,err1,color='b',linewidth=2,label='Train')
    plt.plot(i,err2,color='g',linewidth=2,label='Cross Validation')
    plt.legend()
    plt.show()

def normalEquation(x,y,lamb):
    """
    Closed form solution for regularized linear least squares
    Input
        x is (num_samples x num_features) NDarray of features
        y is (num_samples x 1) vector of targets
    Output
        theta is (num_features x 1) least squares solution
    """
    n = len(x[0]) # number of parameters 
    reg_matrix = lamb * np.identity(n)
    reg_matrix[0,0] = 0.0
    theta = np.linalg.inv( x.T.dot(x) + reg_matrix ).dot(x.T).dot(y)
    return theta
    
def learningCurve(x,y,xval,yval,lamb,num_iters):
    m = len(x)
    n = len(x[0]) # number of parameters
    train_err = []
    val_err = []

    for i in range(n,m+1): # use at least n parameters for training
        orig_theta = np.zeros((n,))
        theta, cost = findMinTheta(orig_theta,x[0:i,:],y[0:i],lamb,num_iters)
        train_err.append(computeCost(theta,x[0:i,:],y[0:i],0.0))
        val_err.append(computeCost(theta,xval,yval,0.0))

    return np.array(train_err), np.array(val_err)

def validationCurve(x,y,xval,yval,num_iters):
    n = len(x[0]) # number of parameters
    lambdas	= np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    train_err = []
    val_err = []

    for lamb in lambdas: # use at least n parameters for training
        orig_theta = np.zeros((n,))
        theta, cost = findMinTheta(orig_theta,x,y,lamb,num_iters)
        train_err.append(computeCost(theta,x,y,0.0))
        val_err.append(computeCost(theta,xval,yval,0.0))

    return np.array(train_err), np.array(val_err)

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

def polyFeatures(x,degree):
    out = np.copy(x)
    for i in range(1,degree):
        out = np.c_[out, x**(i+1)]
    return out

def hypothesis(theta,x):
    """
    Linear prediction of samples given weights, theta
    Input
        x is (num_samples x num_features) NDarray of samples
        y is (num_samples x 1) vector of targets
        theta is (num_features x 1) vector of parameters
    Output
    h is (num_samples x 1) vector of predictions
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
    print ("success=", result.success)    
    #print result
    return result.x, result.fun

    
# Start main program    
vals = sp.io.loadmat("./Data/ex5data1.mat")
Xraw, Yraw = vals['X'], vals['y']
Xv, Yv = vals['Xval'], vals['yval']
Xt, Yt = vals['Xtest'], vals['ytest']

m = len(Xraw)
n = len(Xraw[0]) + 1 # add 1 for bias term
mv = len(Xv)
mt = len(Xt)
print ("number of samples = %d number of parameters = %d" % (m,n))

Praw = polyFeatures(Xraw,DEGREE)
Pv = polyFeatures(Xv,DEGREE)
Pt = polyFeatures(Xt,DEGREE)

# Look at scaling the data versus not scaling and polynomial vs. linear features
if (not poly):
    scaledX = Xraw
    scales = np.ones((n-1,))
    scaledXval = Xv
    scaledXtest = Xt
else:
    scaledX, scales = normalizeVals(Praw)
    scaledXval = Pv / scales
    scaledXtest = Pt / scales
    
print ("scales=", scales)

# append bias = 1 to the sample vectors
X = np.c_[np.ones((m,1)), scaledX]
Xval = np.c_[np.ones((mv,1)), scaledXval]
Xtest = np.c_[np.ones((mt,1)), scaledXtest]

Y = Yraw.flatten()
Yval = Yv.flatten()
Ytest = Yt.flatten()

plotData(X[:,1],Y)
plotData(Xtest[:,1],Ytest)

if (not poly):
    # test out functions
    test_theta = np.array([1.0, 1.0])
    initial_cost = computeCost(test_theta,X,Y,1.0)
    print ("Initial cost=", initial_cost)

    initial_grad = computeGradient(test_theta,X,Y,1.0)
    print ("Initial grad=", initial_grad)
else:
    n = DEGREE + 1
    test_theta = np.zeros((n,))
    
# Look at using various optimization routines including Gradient Descent 
if (0):
    # run Gradient Descent algorithm
    Theta, cost = gradientDescent(test_theta,X,Y,LAMB,alpha,iters)
    print ("starting cost = %f final cost = %f" % (cost[0],cost[-1]))
    print ("final theta=", Theta)
    plotCost(cost)
else:
    # run fminunc or L-BFGS-B
    Theta, cost = findMinTheta(test_theta,X,Y,LAMB,iters)
    print ("final cost = %f" % cost)
    print ("final theta=", Theta)
    # Compute error for test set
    test_err = computeCost(Theta,Xtest,Ytest,0.0)
    print ("test error = %f" % test_err)

# Compute regularized least squares solution
exact = normalEquation(X,Y,LAMB)
print ("exact theta=", exact)

plotFit(Theta,X,Y)

# Plot Learning curve using different number of training samples
train_err, val_err = learningCurve(X,Y,Xval,Yval,LAMB,iters)
plotErrors(train_err,val_err)

# Plot Validation curve using different regularization paramters
train_err, val_err = validationCurve(X,Y,Xval,Yval,iters)
plotErrors(train_err,val_err)

