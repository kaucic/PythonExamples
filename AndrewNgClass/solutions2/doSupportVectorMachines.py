# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:43:54 2015

@author: Robert Kaucic
"""

import logging

import numpy as np
import scipy as sp
from scipy import optimize,special
#from sklearn import svm, grid_search
from sklearn import svm
import matplotlib.pyplot as plt

def readFile(fname):
    data = np.genfromtxt(fname,delimiter=",")
    return (data)
    
def plotData(x,y):
    plt.figure()
    positives  = x[y == 1]
    negatives  = x[y == 0]
    plt.scatter( negatives[:, 0], negatives[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
    plt.scatter( positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" )
    plt.legend()
    
def plotFit(lsvm,x,y):
    # this only works for two dimensional x data i.e. (x1,x2)
    plotData(x,y)
    kernel = lsvm.get_params()['kernel']
    if kernel == 'linear':
        x1s = np.linspace(min(x[:,0]),max(x[:,0]),100)
        w = lsvm.coef_.flatten()
        b = lsvm.intercept_
        x2s = -w[0]/w[1] * x1s - b/w[1]
        plt.plot(x1s,x2s,'r-')
    elif kernel == 'rbf':
        # create a mesh to plot in
        x1_min, x1_max = min(x[:,0]), max(x[:,0])
        x2_min, x2_max = min(x[:,1]), max(x[:,1])
        xx1, xx2 = np.meshgrid(np.linspace(x1_min,x1_max,100),np.linspace(x2_min,x2_max,100))
        vals = np.zeros_like(xx1)
        for i in range(0,len(xx1[1])):
            this_X = np.c_[ xx1[:, i], xx2[:, i] ]
            vals[:, i] = lsvm.predict(this_X)
        plt.contour(xx1,xx2,vals,colors='red')

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

def findParameters(x,y,xval,yval):
    C_values 	 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
    m = len(C_values)
    n = len(sigma_values)
    
    # dictionary for saving best result
    scores = np.zeros((m,n))
    best = {'score': -999, 'C': 0.0, 'sigma': 0.0 }

    # train svm	
    lsvm = svm.SVC(kernel='rbf')

    for i in range(m):
        for j in range(n):
            # train the SVM first
            lsvm.set_params(C=C_values[i], gamma=1.0/sigma_values[j])
            lsvm.fit(x,y)

            # compute score on validation data
            score = lsvm.score(xval,yval)
            scores[i,j] = score
			
            # get the lowest error
            if score > best['score']:
                best['score'] = score
                best['C'] = C_values[i]
                best['sigma'] = sigma_values[i]

    best['gamma'] = 1.0 / best['sigma']
    
    return best, scores

def gaussianKernel(x1,x2,sigma):
    val = np.exp(-0.5/(sigma**2) * np.sum((x1-x2)**2))
    return val
    
# Start main program    
vals = sp.io.loadmat("./Data/ex6data1.mat")
Xraw, Yraw = vals['X'], vals['y']
#Xv, Yv = vals['Xval'], vals['yval']
#Xt, Yt = vals['Xtest'], vals['ytest']

m = len(Xraw)
n = len(Xraw[0])
print ("number of samples = %d number of parameters = %d" % (m,n))

# Look at scaling the data versus not scaling
if (1):
    scaledX = Xraw
    scales = np.ones((n,))
else:
    scaledX, scales = normalizeVals(Xraw)
    
print ("scales=", scales)

# no need to append bias = 1 for SVM
#X = np.c_[np.ones((m,1)), scaledX]
X = scaledX
Y = Yraw.ravel()

# linear SVM with C = 1
linear_svm = svm.SVC(C=1,kernel='linear',verbose=False)
linear_svm.fit(X,Y)

print ("num support vecs =", linear_svm.n_support_)
print ("coefs =", linear_svm.coef_)
print ("intercept =", linear_svm.intercept_)

plotFit(linear_svm,X,Y)

# linear SVM with C = 100
linear_svm = svm.SVC(C=100,kernel='linear',verbose=False)
linear_svm.fit(X,Y)
plotFit(linear_svm,X,Y)

x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2.0

print ("Gaussian kernel: %f" % gaussianKernel(x1,x2,sigma))

# load data set #2
vals = sp.io.loadmat("./Data/ex6data2.mat")
Xraw, Yraw = vals['X'], vals['y']

scaledX, scales = normalizeVals(Xraw)
print ("scales=", scales)

X = scaledX
Y = Yraw.ravel()

# kernel SVM with C = 1
sigma = 0.01 # gamma is actually inverse of sigma
rbf_svm = svm.SVC(C=1,kernel='rbf',gamma=1.0/sigma,verbose=False) 
rbf_svm.fit(X,Y)

#plotData(X,Y)
plotFit(rbf_svm,X,Y)

# load data set #3
vals = sp.io.loadmat("./Data/ex6data3.mat")
Xraw, Yraw = vals['X'], vals['y']
Xv, Yv = vals['Xval'], vals['yval']

scaledX, scales = normalizeVals(Xraw)
print ("scales=", scales)

X = scaledX
Xval = Xv / scales
Y = Yraw.ravel()
Yval = Yv.ravel()

best, scores = findParameters(X,Y,Xval,Yval)
print ("scores=", scores)
print ("best=", best)
rbf2_svm = svm.SVC(C=best['C'],kernel='rbf',gamma=best['gamma']) 
rbf2_svm.fit(X,Y)

plotFit(rbf2_svm,X,Y)
