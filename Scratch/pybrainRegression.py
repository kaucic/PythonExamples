# -*- coding: utf-8 -*-
"""
Example of a Recurrent Neural Network used for regression

Attempt to see if a simple (2,2,1) RNN with 1 feedback layer can
learn y(n) = 2*x1(n) - x1(n-1) + 0.5*x2(n)

@author: Robert Kaucic
"""

import math
import random
import numpy as np

import matplotlib.pyplot as plt

from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def func(x1,x2,noise=0.0):
    """
    Compute y(n) = 2*x1(n) - 1*x1(n-1) + 0.5*x2(n) + noise
    """
    y = [0.0] * len(x1)
    for n in range(len(x1)):
        if (n == 0):
            previous = 0
        else:
            previous = -0.1*x1[n-1]
        y[n] = 2.0*x1[n]  + 0.5*x2[n] + previous + noise*random.gauss(0,1)
    return (y)
 
def create_sin(N):
    """
    Simple function for creating sample input training data
    Use period of 24 to mimic 24 hours in a day
    Provide unique training values by perturbing ~1% 
    """
    out = [0.0] * N
    for n in range(N):
        out[n] = 4.0*math.sin(2.0*math.pi*n/24.0) + 0.04*random.gauss(0,1)
    return (out)

def create_ramp(N):
    """
    Simple function for creating sample input training data
    Use period of 24 to mimic 24 hours in a day
    Ramp up first half of day and ramp down second half of day
    Provide unique training values by perturbing ~1% 
    """
    out = [0.0] * N
    for n in range(N):
        rem = n % 24
        if (rem < 12):
            out[n] = 10.0*(rem/12.0) + 0.1*random.gauss(0,1)
        else:
            rem -= 12
            out[n] = 10.0*(1-(rem/12.0)) + 0.1*random.gauss(0,1)
    return (out)
    
def normalize_vals(x):
    """
    Do simple scaling values by max positive or negative value to make 
    data in the range [-1,1]
    With simple scaling, RNN weights should be able to undo scaling
    """
    a = max(x)
    b = abs(min(x))
    scale = max(a,b)
    out = [0.0] * len(x)
    for i in range(len(x)):
        out[i] = x[i] / scale
    return (out,scale)

def list_diff(y,x):
    """
    Compute difference of lists to determine prediction error
    """
    diff = [0.0] * len(y)
    for i in range(len(y)):
        diff[i] = y[i] - x[i]
    return (diff)

def plot_lists(a,b,c):
    i = list(range(len(a)))
    plt.figure()
    l1, = plt.plot(i,a,'r',label='predicted')
    l2, = plt.plot(i,b,'g',label='reference')
    l3, = plt.plot(i,c,'b',label='pred-ref')
    plt.legend(handles=[l1,l2,l3])
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
                

# Create training data sets           
x1 = create_sin(240)
x2 = create_ramp(240)

# scale all inputs to be around [-1,1]
x1, x1_scale = normalize_vals(x1)
x2, x2_scale = normalize_vals(x2)
y = func(x1,x2,0.0)

#print "x1=", x1
#print "x2=", x2
#print "y=", y


# Create regression training data for RNN
ds = SupervisedDataSet(2, 1)
for n in range(len(x1)):
    ds.addSample((x1[n], x2[n]), (y[n]))

# Create Recurrent Network Structure 
net = RecurrentNetwork()
net.addInputModule(LinearLayer(2, name='in'))
net.addModule(LinearLayer(2, name='hidden'))
net.addOutputModule(LinearLayer(1, name='out'))
net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))
#net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='c3'))
net.sortModules()

# Train network
trainer = BackpropTrainer(net,ds,learningrate=0.01,verbose=True)
trainErr = trainer.trainUntilConvergence(maxEpochs=20,validationProportion=0.2)
#print "trainErr=", trainErr

print_network(net)

# Create testing data set
n1 = [1.0,   3.0,  3.5,  4.0,  3.5,  3.0,  2.0, 1.0, 0.0, -1.0, -2.0, -1.5, -0.5,  0.0, 1.0,   2.0,  3.0,  4.0, 3.0, 2.0]
n2 = [10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,  7.5,  8.0,  9.0, 10.0, 12.0, 12.5, 11.0,  10.0, 9.0, 8.0, 7.5]
ycorrect = func(n1,n2)

testds = SupervisedDataSet(2, 1)
for n in range(len(n1)):
    testds.addSample((n1[n], n2[n]), (0.0))

yout = net.activateOnDataset(testds)
diff = list_diff(yout,ycorrect)
mse = np.multiply(diff,diff).mean()

plot_lists(yout,ycorrect,diff)
print "Ycorrect=", ycorrect
print "Ypredicted=", yout
print "diff=", diff
print "mse=", mse

# Reset the network, so that we can retrain
#net.reset()
#del ds,net,trainer,testds
