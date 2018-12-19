# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:43:54 2015

@author: Kimberly
"""

import numpy as np
import scipy as sp
from scipy import optimize,special
import matplotlib.pyplot as plt

# Determine which DS to use
DS = 2
if (DS == 1):
    FILENAME = "./Data/ex2data1.txt"
    lamb = 0.0
    alpha = 3.0
    iters = 5000
else:
    # POLYNOMIIAL LINEAR REGRESSION NOT IMPLEMENTED
    FILENAME = "./Data/ex2data2.txt"
    lamb = 0.01

def readFile(fname):
    data = np.genfromtxt(fname,delimiter=",")
    return data

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
    positives  = x[y == 1]
    negatives  = x[y == 0]

    plt.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
    plt.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" )

    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    #plt.xlim([25, 115])
    #plt.ylim([25, 115])
    plt.legend()

def plotBoundary(theta,x,y):
    plotData(x,y)
    xs = np.array( [ np.min(x[:,0]), np.max(x[:,0]) ] )
    ys = (-1./ theta[2]) * (theta[1] * xs + theta[0])
    plt.plot(xs,ys)
    plt.show()

def plotContour(theta):
    
    u = np.linspace( -1, 1.5, 50 )
    v = np.linspace( -1, 1.5, 50 )
    z = np.zeros( (len(u), len(v)) )
    
    for i in range(0, len(u)): 
        for j in range(0, len(v)):
            mapped = mapFeature( np.array([u[i]]), np.array([v[j]]) )
            z[i,j] = mapped.dot( theta )
            z = z.transpose()
            
            u, v = np.meshgrid( u, v )
            plt.contour( u, v, z, [0.0, 0.0], label='Decision Boundary' )
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
    return (nx, scale_factors)

def sigmoid(z):
    # return 1.0 / (1.0 + exp( -z ))
    return sp.special.expit(z)
	
def hypothesis(theta,x):
    """
    x is (m x n) matrix of samples
    y is (n x 1) vector of targets
    theta is (n x 1) vector of parameters
    h is (m x 1) vector of predictions
    """
    # return 1.0 / (1.0 + exp( -z ))
    return sigmoid(x.dot(theta)) 

def computeGradient(theta,x,y):
    """
    grad = 1/m * (h(theta*x) - y) * x
    """
    m = float(len(y))
    err = hypothesis(theta,x) - y
    grad = x.T.dot(err) / m
    grad[1:] += (lamb/m * theta[1:])    
    return grad

def computeCost(theta,x,y):
    """
    Cost = 1/m * ( -y * log(h(x)) - (1 - y) * log (1 - h(x)) ) + lamb/2m * theta^2
    """
    m = float(len(y))
    h = hypothesis(theta,x)
    term1 = -y.T.dot(np.log(h))
    term2 = (1.0 - y).T.dot(np.log(1.0 - h))
    regularization = lamb / 2.0 * theta[1:].dot(theta[1:])
    cost = 1.0 / m * ( term1 - term2 + regularization )
    return cost

def costAndGradientFunction(theta,x,y):
    cost = computeCost(theta,x,y)
    grad = computeGradient(theta,x,y)    
    return cost,grad    

def gradientDescent(orig_theta,x,y,alpha,num_iter):
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
    cost = np.zeros(num_iter)
    for i in range(num_iter):
        grad = computeGradient(theta,x,y)
        theta -= (alpha * grad)
        cost[i] = computeCost(theta,x,y)
        #print "Iter = %d cost = %f" % (i,cost[i])
        #print "theta=", theta
    return theta,cost

def findMinTheta(orig_theta,x,y,num_iter):
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
    result = sp.optimize.minimize(costAndGradientFunction,x0=theta,args=(x,y),method='L-BFGS-B',jac=True)  
    print result
    return result.x, result.fun

def mapFeature( X1, X2 ):
    """
    Compute 28 sixth order polynomials for 2D data
    """
    degrees = 6
    out = np.ones( (np.shape(X1)[0], 1) )
    
    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = X1 ** (i-j)
            term2 = X2 ** (j)
            term  = (term1 * term2).reshape( np.shape(term1)[0], 1 ) 
            out   = np.hstack(( out, term ))
    
    return out

# Start main program    
raw_vals = readFile(FILENAME)
Y = raw_vals[:,-1]

if (DS == 1):
    vals = raw_vals
    m,n = np.shape(vals)
    
    # Look at scaling the data versus not scaling
    if (0):
        scaledX = vals[:,:-1]
        scales = np.ones((n-1,))
    else:
        scaledX, scales = normalizeVals(vals[:,:-1])
        print "scales=", scales

    # append bias = 1 to the sample vectors
    X = np.c_[np.ones((m,1)), scaledX]
    
    plotData(scaledX,Y)

else:
    X = mapFeature(raw_vals[:,0],raw_vals[:,1])
    m,n = np.shape(X)
    scales = np.ones((n-1,))
    
    plotData(raw_vals[:,:-1],Y)
    
print "number of samples = %d number of parameters = %d" % (m,n)

params = np.zeros((n,))

# test out functions
h = hypothesis(params,X)
cost = computeCost(params,X,Y)
#print "h=", h
print "Initial cost=", cost

# Look at using various optimization routines including Gradient Descent 
if (0):
    # run Gradient Descent algorithm
    theta, cost = gradientDescent(params,X,Y,alpha,iters)
    print "starting cost = %f final cost = %f" % (cost[0],cost[-1])
    plotCost(cost)
else:
    # run fminunc or L-BFGS-B
    theta, cost = findMinTheta(params,X,Y,iters)
    print "final cost = %f" % cost

print "final theta=", theta

# predict new values for test data
if (DS == 1):
    plotBoundary(theta,scaledX,Y)
    x1 = np.array([1.0, 45.0, 85.0])
    x1[1:] = x1[1:] / scales # exclude intercept unit
    predict1 = hypothesis(theta,x1) 
    print "predict1=", predict1
else:
    plotData(raw_vals[:,0:2],Y)
    plotContour(theta)
