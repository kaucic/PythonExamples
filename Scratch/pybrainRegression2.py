# -*- coding: utf-8 -*-
"""
Example of a Recurrent Neural Network used for regression

Attempt to see if a simple (2,2,1) RNN with 1 feedback layer can
learn y(n) = 2*x1(n) - 0.1*x1(n-1) + 0.5*x2(n)

@author: Robert Kaucic
"""

import random
import numpy as np

import matplotlib.pyplot as plt

from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def fir1(x1,x2,noise=0.0):
    """
    Compute y(n) = 2*x1(n) - 0.1*x1(n-1) + 0.5*x2(n) + noise
    """
    y = np.zeros(len(x1))
    for i in range(len(x1)):
        if (i == 0):
            n_minus_1 = 0
        else:
            n_minus_1 = -0.1*x1[i-1]
            n_minus_1 = 0
        y[i] = 2.0*x1[i]  + 0.5*x2[i] + n_minus_1 + noise*random.gauss(0,1)
    return (y)
 
def iir1(x1,x2,noise=0.0):
    """
    Compute y(n) = 2*x1(n) + 0.5*x2(n) - 0.5*y(n-1) + noise
    """
    y = np.zeros(len(x1))
    for i in range(len(x1)):
        if (i == 0):
            n_minus_1 = 0
        else:
            n_minus_1 = -0.5*y[i-1]
        y[i] = 2.0*x1[i]  + 0.5*x2[i] + n_minus_1 + noise*random.gauss(0,1)
    return (y)
    
def create_sin(N):
    """
    Simple function for creating sample input training data
    Use period of 24 to mimic 24 hours in a day
    Add noise to period to increase frequency range of signal
    Provide unique training values by perturbing ~1% of magnitude
    """
    y = np.zeros(N)
    i = np.arange(float(N))
    for i in range(N):
        y[i] = 4.0*np.sin(2.0*np.pi*i/(24.0+random.gauss(0,1)) + random.gauss(0,0.04))
    return (y)

def create_ramp(N):
    """
    Simple function for creating sample input training data
    Ramp up first half of day and ramp down second half of day
    Use period of 23 to mimic 24 hours in a day
    Add noise to period to increase frequency range of signal    
    Provide unique training values by perturbing ~1% 
    """
    y = np.zeros(N)
    for i in range(N):
        rem = i % 23
        if (rem < 12):
            y[i] = 10.0*(rem/(12.0+random.gauss(0,1)) + 0.1*random.gauss(0,1))
        else:
            rem -= 12
            y[i] = 10.0*(1-(rem/(11.0+random.gauss(0,1))) + 0.1*random.gauss(0,1))
    return (y)
    
def create_rnd(N):
    y = np.random.normal(0,1,N)
    return (y)
    
def normalize_vals(x):
    """
    Do simple scaling values by max positive or negative value to make 
    data in the range [-1,1]
    With simple scaling, RNN weights should be able to undo scaling
    """
    maxvals = x.max(axis=0).astype(float)
    minvals = np.absolute(x.min(axis=0)).astype(float)
    scale_factors = np.maximum(maxvals,minvals)
    # assume x is 1D    
    nx = x / scale_factors    
    return (nx, scale_factors)
    
def list_diff(y,x):
    """
    Compute difference of lists to determine prediction error
    """
    diff = np.zeros(len(y))
    for i in range(len(y)):
        diff[i] = y[i] - x[i]
    return (diff)

def plotLists(a,b,c):
    i = np.arange(len(a))
    plt.figure()
    l1, = plt.plot(i,a,'r',label='predicted')
    l2, = plt.plot(i,b,'g',label='reference')
    l3, = plt.plot(i,c,'b',label='pred-ref')
    plt.legend(handles=[l1,l2,l3])
    plt.show()

def plotCost(cost):
    i = np.arange(len(cost))
    plt.plot(i,cost,'-b' )
    plt.title("Cost Function")
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()
    
def print_network(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]

    for conn in n.recurrentConns:
        print conn
        for cc in range(len(conn.params)):
            print conn.whichBuffers(cc), conn.params[cc]
                
# Start main program   
                
# Create training data sets           
#x1train = create_sin(240)
#x2train = create_ramp(240)
x1train = create_rnd(240)
x2train = create_rnd(240)

# scale all inputs to be around [-1,1]
x1train, x1_scale = normalize_vals(x1train)
x2train, x2_scale = normalize_vals(x2train)
ytrain = iir1(x1train,x2train,0.01)

#print "x1=", x1train
#print "x2=", x2train
#print "y=", ytrain
#plotLists(x1train,x2train,ytrain)

# Create regression training data for RNN
ds = SupervisedDataSet(2, 1)
for i in range(len(x1train)):
    ds.addSample((x1train[i], x2train[i]), (ytrain[i]))

# Create Recurrent Network Structure 
net = RecurrentNetwork()
net.addInputModule(LinearLayer(2, name='in'))
net.addModule(LinearLayer(1, name='hidden'))
net.addOutputModule(LinearLayer(1, name='out'))
net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))
#net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='c3'))
net.addRecurrentConnection(FullConnection(net['out'], net['out'], name='c3'))
net.sortModules()

# Train network
trainer = BackpropTrainer(net,ds,learningrate=0.01,batchlearning=True,verbose=True)
#trainUntilConvergence does not work for RNNs, because validationProportion
#rearranges the order of the input samples
#trainErr = trainer.trainUntilConvergence(maxEpochs=50,validationProportion=0.2)
#plotCost(trainErr[1])
trainErr = np.zeros(10)
print "Network before training"
print_network(net)
print ""
for i in range(10):
    trainErr[i] = trainer.train()
    print_network(net)
    print ""
    net.reset()
plotCost(trainErr)

# Create testing data set
x1test = np.array([1.0,   3.0,  3.5,  4.0,  3.5,  3.0,  2.0, 1.0, 0.0, -1.0, -2.0, -1.5, -0.5,  0.0, 1.0,   2.0,  3.0,  4.0, 3.0, 2.0])
x2test = np.array([10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,  7.5,  8.0,  9.0, 10.0, 12.0, 12.5, 11.0,  10.0, 9.0, 8.0, 7.5])
ycorrect = iir1(x1test,x2test)
#ycorrect = func(x1train,x2train)

testds = SupervisedDataSet(2, 1)
for i in range(len(x1test)):
    testds.addSample((x1test[i], x2test[i]), (0.0))
#for i in range(len(x1train)):
#    testds.addSample((x1train[i], x2train[i]), (0.0))

ytest = net.activateOnDataset(testds)
diff = list_diff(ytest,ycorrect)
mse = np.multiply(diff,diff).mean()

print "Ycorrect=", ycorrect
print "Ypredicted=", ytest
#print "diff=", diff
print "mse=", mse

plotLists(ytest,ycorrect,diff)

# Reset the network, so that we can retrain
#net.reset()

# estimate the parameters of an FIR filter using linear regression
def estimate_fir_params(x1n,x2n,yn):
    y = np.mat(yn).T
    A = np.mat(np.zeros([len(x1n),3]))
    for i in range(len(x1n)):
        A[i,0] = x1n[i]
        if (i == 0):
            A[i,1] = 0
        else:
            A[i,1] = x1n[i-1]
        A[i,2] = x2n[i]
    theta = np.linalg.inv(A.T*A) * A.T * y
    return (theta)
  
#fir1_params = estimate_fir_params(x1train,x2train,ytrain)
#print "fir1_params=", fir1_params


    





