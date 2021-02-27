# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:43:54 2015

Exercise 3

@author: Kimberly
"""

import logging
 
import numpy as np
import scipy as sp
#from scipy import optimize,special
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

lamb = 0.1
alpha = 3.0
iters = 300

def plotCost(cost):
    i = np.arange(len(cost))
    plt.figure()
    plt.plot(i,cost,'-b' )
    plt.title("Cost Function")
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

def displayData(x,thetas=None):
    m = len(x)
    # remove bias term from images/samples
    x_raw = x[:,1:]
    
    width = 20
    rows, cols = 10, 10

    im_out = np.zeros((width*rows,width*cols))
    rand_indices = np.random.permutation(m)[0:rows*cols]

    counter = 0
    for r in range(0, rows):
        for c in range(0, cols):
            start_x = c * width
            start_y = r * width
            im_out[start_x:start_x+width,start_y:start_y+width] = x_raw[rand_indices[counter]].reshape(width, width).T
            counter += 1
    #img = sp.misc.toimage(im_out)
    img = Image.fromarray(im_out*255)

    # display image of digits
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.imshow(img)

    # display classification results
    if thetas is not None:
        result_matrix 	= []
        for idx in rand_indices:
            result = classifySample(thetas,x[idx])
            result_matrix.append(result)
        result_matrix = np.array(result_matrix).reshape(rows,cols).T
        print (result_matrix)

    plt.show( )

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

def computeGradient(theta,x,y):
    """
    grad = 1/m * (h(theta*x) - y) * x
    """
    m = float(len(y))
    err = hypothesis(theta,x) - y
    grad = x.T.dot(err) / m
    grad[1:] += (lamb/m * theta[1:])    
    return grad

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
        #print ("Iter = ", i, " cost = ", cost[i])
        #print ("theta= ", theta)
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
    result = sp.optimize.minimize(costAndGradientFunction,x0=theta,args=(x,y),method='L-BFGS-B',jac=True,options={'maxiter': num_iter}) 
    print ("success= ", result.success)
    #print (result)
    return result.x, result.fun

def predictOneVsAll(thetas,x,y):
    """
    Function to determine the identity of a digit given 10 separate One vs. All classifiers
    
    Input
        thetas is (K x n) matrix of logistic regression coefficients
        x is (m x n) matrix of samples
        y is (n x 1) vector of class type (1-K)
    Output
        accuracy is the classification accuracy in percent
    """
    m = len(y)
    K = len(thetas)    
    correct = 0
    for i in range(0,m):
        prediction = classifySample(thetas,X[i])
        actual = y[i] % K # Digit 0 is coded as k = 10
        # print ("prediction = ", prediction, " actual = ", actual)
        if actual == prediction:
            correct += 1
    accuracy = correct / float(m) * 100.0
    return accuracy

def classifySample(thetas,samp):
    """
    Since the outputs of the logistic classifiers are sigmoids, we can take the max
    of the matrix multiplication of the logistic weights with each sample to 
    determine the most likely class
    Input
        thetas is (K x n) matrix of logistic regression coefficients
        x is (n x 1) matrix of samples
    Output
        which_class is scalar (0 to K-1)
    """
    which_class = np.argmax(thetas.dot(samp))
    return which_class
    
# Start main program    
np.set_printoptions(precision=6, linewidth=200)

vals = sio.loadmat("./Data/ex3data1.mat")
Xvals, Y = vals['X'], vals['y']
m = len(Xvals)
n = len(Xvals[0]) + 1
print ("number of samples = ", m, " number of parameters = ", n)

# Look at scaling the data versus not scaling
if (1):
    scaledX = Xvals
    scales = np.ones((n-1,))
else:
    scaledX, scales = normalizeVals(Xvals)
#print ("scales= ", scales)

# append bias = 1 to the sample vectors
X = np.c_[np.ones((m,1)), scaledX]

displayData(X)

# Train K = 10 separate One versus All classifiers
K = np.max(Y)
thetas = np.zeros((K,n))
costs = np.zeros((K,))
for k in range(1,K+1):
    k_mod = k % K # Digit 0 is coded as k = 10
    newY =  ((Y == k) + 0).reshape(-1)    
    params = np.zeros((n,))
    theta, cost = findMinTheta(params,X,newY,iters)
    print (k, " Cost: ", cost)
    #print ("final theta= ", theta[0:5])
    thetas[k_mod,:] = theta
    costs[k_mod] = cost
    
# predict new values for test data
accuracy = predictOneVsAll(thetas,X,Y)
print ("Accuracy: ", accuracy)

displayData(X,thetas)
