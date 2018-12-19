# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:43:54 2015

@author: Robert Kaucic
"""

import numpy as np
import scipy as sp
from scipy import optimize,special
import matplotlib.pyplot as plt

lamb = 1.0
iters = 50

SAMP = 2000

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

    out = np.zeros((width*rows,width*cols))
    rand_indices = np.random.permutation(m)[0:rows*cols]

    counter = 0
    for r in range(0, rows):
        for c in range(0, cols):
            start_x = c * width
            start_y = r * width
            out[start_x:start_x+width,start_y:start_y+width] = x_raw[rand_indices[counter]].reshape(width, width).T
            counter += 1
    img = sp.misc.toimage(out)

    # display image of digits
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.imshow(img)

    # display classification results
    if thetas is not None:
        theta1, theta2 = unrollParams(thetas,input_nodes,hidden_nodes,output_nodes)        
        result_matrix 	= []
        for idx in rand_indices:
            result = classifySample(theta1,theta2,x[idx])
            result_matrix.append(result)
        result_matrix = np.array(result_matrix).reshape(rows,cols).T
        print result_matrix

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

def recodeLabels(yvec):
    m = len(yvec)
    K = max(yvec)
    Ymatrix = np.zeros((m,K))
    #indices = np.nonzero(yvec==k)
    for i in range(m):
        Ymatrix[i, yvec[i]-1] = 1.0 # index 0 = class 1, index 9 = class 10 (0)
    return Ymatrix
        
def flattenParams(theta1,theta2):
    nn_params = np.r_[theta1.flatten(), theta2.flatten()]
    return nn_params

def unrollParams(nn_params,input_layer_size,hidden_layer_size,num_classes):
    theta1_elems = (input_layer_size+1) * hidden_layer_size
    theta1_size  = (hidden_layer_size,input_layer_size+1)
    theta2_size  = (num_classes,hidden_layer_size+1)

    theta1 = nn_params[:theta1_elems].reshape(theta1_size)	
    theta2 = nn_params[theta1_elems:].reshape(theta2_size)
    return theta1, theta2

def sigmoid(z):
    # 1.0 / (1.0 + exp( -z ))
    return sp.special.expit(z)

def sigmoidGradient(z):
    # sig'(z) = sig(z) * (1.0 - sig(z))
    return sigmoid(z) * (1.0 - sigmoid(z))
    
def feedforward(theta1,theta2,samp):
    """
    Create hypothesis vector for a single sample feature vector
    Input
        theta1 is (hidden x n_features) matrix of parameters
        theta2 is (n_classes x hidden+1) matrix of parameters
        samp is (n_features x 1) vector of samples with bias 1
    Output
        y is (n_classes x 1) vector of predictions
    """
    a1 = samp
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2)        
    # add bias node  to a2
    a2 = np.r_[1.0, a2]
    z3 = theta2.dot(a2)
    a3 = sigmoid(z3)
    h = a3
    return h, a2, a1, z2

def computeCost2(theta1,theta2,samp,one_y):
    """
    Compute cost for one (sample,target) pair
    Cost = ( -y * log(h(x)) - (1 - y) * log (1 - h(x)) )
    """
    #K = len(theta2)
    #ones = np.ones((K,))
    h, a2, a1, z2 = feedforward(theta1,theta2,samp)
    term1 = sum(-one_y * np.log(h))
    term2 = sum( (1 - one_y) * np.log(1 - h) )    
    cost = term1 - term2
    return cost
    
def nnComputeCost(nn_params,x,y):    
    """
    Compute cost for all (sample,target) pairs
    Cost = 1/m * ( -y * log(h(x)) - (1 - y) * log (1 - h(x)) ) + lamb/2m * theta^2
    """
    theta1, theta2 = unrollParams(nn_params,input_nodes,hidden_nodes,output_nodes)

    m = len(y)
    total_cost = 0
    for i in range(m):
        total_cost += computeCost2(theta1,theta2,x[i],y[i])
    # add regularization term;  skip bias terms, i.e. theta when k=0
    regularization = lamb / 2.0 * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))
    total_cost = (total_cost + regularization) / float(m)  
    return total_cost
    
def computeGradient2(theta1,theta2,samp,one_y):
    """
    Compute the gradient of a neural network using backpropagation for a single sample
    1) Feedforward -- cash hypothesis, activations, and weights times features
    2) Backpropagate errors -- remove bias terms from error function
    3) Accumulate gradients using backpropagation errors and activations
    """
    # h = (10 x 1), a2 = (26 x 1), a1 = (401 x 1), z2 = (25 x 1)
    h, a2, a1, z2 = feedforward(theta1,theta2,samp)
    delta3 = h - one_y # (10 x 1)
    
    # back propagate
    delta2 = theta2.T.dot(delta3) * sigmoidGradient(np.r_[1, z2]) # (26 x 1) .* (26 x 1)
    delta2 = delta2[1:] # remove bias term error, (25 x 1)       
        
    theta2p = np.outer(delta3,a2.T) # (10 x 1) * (1 x 26) = (10 x 26)
    theta1p = np.outer(delta2,a1.T) # (25 x 1) * (1 x 401) = (25 x 401)    

    return theta1p, theta2p

def nnComputeGradient(nn_params,x,y):    
    """
    Compute the gradient of a neural network using backpropagation
    1) Feedforward -- cash hypothesis, activations, and weights times features
    2) Backpropagate errors -- remove bias terms from error function
    3) Accumulate gradients using backpropagation errors and activations
    4) Regularize over entire sequence
    """
    theta1, theta2 = unrollParams(nn_params,input_nodes,hidden_nodes,output_nodes)     

    # Accumulators for gradients
    Delta1 = np.zeros(theta1.shape) # (25 x 401)
    Delta2 = np.zeros(theta2.shape) # (10 x 26)

    m = len(y)
    for i in range(m):
        theta1p, theta2p = computeGradient2(theta1,theta2,x[i],y[i])
        Delta1 += theta1p    
        Delta2 += theta2p

    # add regularization term;  skip bias terms, i.e. theta when k=0
    Delta1[:,1:] += (lamb * theta1[:,1:])
    Delta2[:,1:] += (lamb * theta2[:,1:])
    
    # scale by sequence length
    Delta1 = Delta1 / float(m)
    Delta2 = Delta2 / float(m)
    
    grad = flattenParams(Delta1,Delta2)
    
    return grad

def nnCostAndGradientFunction(nn_params,x,y):
    cost = nnComputeCost(nn_params,x,y)
    grad = nnComputeGradient(nn_params,x,y)
    return cost,grad    

def randInitializeWeights(input_layer,output_layer):
    eps = 0.12
    theta = 2.0 * eps * np.random.rand(output_layer,input_layer+1) - eps
    return theta

def findMinTheta(nn_params,x,y,num_iter):
    """
    Input
        nn_params is a flattened vector of neural network weights
        x is (m x n) matrix of samples
        y is (m x p) vector of one-hot class labels
    Output
        thetas is a flattened vector final solution
        cost is the resultant cost
    """
    thetas = np.copy(nn_params)
    result = sp.optimize.minimize(nnCostAndGradientFunction,x0=thetas,args=(x,y),method='L-BFGS-B',jac=True,options={'maxiter': num_iter}) 
    print "success=", result.success
    #print result
    return result.x, result.fun

def classifySample(theta1,theta2,samp):
    h, a2, a1, z2 = feedforward(theta1,theta2,samp)
    prediction =  np.argmax(h) + 1 # Digit 1 is index 0
    return prediction

def computeAccuracy(nn_params,x,y):
    """
    Function to determine the identity of a digit    
    Input
        nn_params is a flattened vector of neural network weights
        x is (m x n) matrix of samples
        y is (m x p) vector of one-hot class labels
    Output
        accuracy is the classification accuracy in percent
    """
    theta1, theta2 = unrollParams(nn_params,input_nodes,hidden_nodes,output_nodes)
    m = len(y)  
    correct = 0
    for i in range(0,m):
        prediction = classifySample(theta1,theta2,x[i])
        actual =  np.argmax(y[i]) + 1 # Digit 1 is index 0
        # print "prediction = %d actual = %d" % (prediction, actual)
        if actual == prediction:
            correct += 1
    accuracy = correct / float(m) * 100.0
    return accuracy
    
# Used to verify numerical gradients
def computeCost(nn_params,samp,one_y):
    theta1, theta2 = unrollParams(nn_params,input_nodes,hidden_nodes,output_nodes)
    cost = computeCost2(theta1,theta2,samp,one_y)
    return cost

# Used to verify numerical gradients
def computeGradient(nn_params,samp,one_y):
    theta1, theta2 = unrollParams(nn_params,input_nodes,hidden_nodes,output_nodes)    
    theta1p, theta2p = computeGradient2(theta1,theta2,samp,one_y)
    grad = flattenParams(theta1p,theta2p)    
    return grad
    
# Used to verify backpropagation calculation
def computeNumericalGradient(theta,samp,target):
    """
    Compute the gradient numerically by
    grad(x) = (J(theta + eps) - J(theta - eps)) / (2 * eps)
    """
    eps = 1e-4
    n = len(theta)
    grad = np.zeros((n,))
    for i in range(n):
        plus = theta.copy()
        plus[i] = theta[i] + eps
        minus = theta.copy()
        minus[i] = theta[i] - eps
        grad[i] = ( computeCost(plus,samp,target) - computeCost(minus,samp,target)) / (2.0 * eps)
    return grad
        
# Start main program    
np.set_printoptions(precision=6, linewidth=200)

print "test sigmoid gradient = ", sigmoidGradient(np.array([-5.0,-1.0,0,1.0,5.0]))

vals = sp.io.loadmat("./Data/ex4data1.mat")
Xvals, Yvals = vals['X'], vals['y']
m = len(Xvals)
n = len(Xvals[0]) + 1
print "number of samples = %d number of parameters = %d" % (m,n)

# Look at scaling the data versus not scaling
if (1):
    scaledX = Xvals
    scales = np.ones((n-1,))
else:
    scaledX, scales = normalizeVals(Xvals)
#print "scales=", scales

# append bias = 1 to the sample vectors
X = np.c_[np.ones((m,1)), scaledX]
Yvals = Yvals.flatten()
Y = recodeLabels(Yvals)

# read in learnt thetas
theta_vals = sp.io.loadmat("./Data/ex4weights.mat")
GT_Theta1, GT_Theta2 = theta_vals['Theta1'], theta_vals['Theta2']

# Global variables
hidden_nodes = len(GT_Theta1)
input_nodes = len(GT_Theta1[0]) - 1 # theta matrix includes bias node
output_nodes = len(GT_Theta2) 

NN_Params = flattenParams(GT_Theta1,GT_Theta2)
print "NN_Params dimensions = ", NN_Params.shape

displayData(X,NN_Params)

Theta1, Theta2 = unrollParams(NN_Params,input_nodes,hidden_nodes,output_nodes)
print "Theta1 dimensions = ", Theta1.shape
print "Theta2 dimensions = ", Theta2.shape

predict, a2, a1, z2  = feedforward(Theta1,Theta2,X[SAMP])
print "predict = ", predict
print "Y=", Y[SAMP]

cost = computeCost2(Theta1,Theta2,X[SAMP],Y[SAMP])
print "cost = ", cost

total_cost = nnComputeCost(NN_Params,X,Y)
print "total_cost = ", total_cost

GT_accuracy = computeAccuracy(NN_Params,X,Y)
print "GT_accuracy: %.2f%%" % GT_accuracy

# learn neural network
orig_theta1 = randInitializeWeights(input_nodes,hidden_nodes)
orig_theta2 = randInitializeWeights(hidden_nodes,output_nodes)
orig_params = flattenParams(orig_theta1,orig_theta2)

# confirm that backpropagation math is correct
if (0):
    print "verifying backpropagation calculation ..."
    grad = computeGradient(orig_params,X[SAMP],Y[SAMP])
    ngrad = computeNumericalGradient(orig_params,X[SAMP],Y[SAMP])
    grad_err = np.sum((grad - ngrad)**2).mean()
    print "grad_err = ", grad_err

#cost, grad = nnCostAndGradientFunction(NN_Params,X,Y)

print "training neural network ..."
Thetas, cost = findMinTheta(orig_params,X,Y,iters)
print "training complete"
print "cost = ", cost

accuracy = computeAccuracy(Thetas,X,Y)
print "accuracy: %.2f%%" % accuracy

displayData(X,Thetas)