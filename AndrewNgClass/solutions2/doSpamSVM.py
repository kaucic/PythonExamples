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
import re
import csv
import nltk
import pickle
import matplotlib.pyplot as plt

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


def getVocabSet(fname):
    vocab_set = {}
    with open(fname, 'r') as file:
        reader = csv.reader( file, delimiter='\t' )
        for row in reader:
            vocab_set[row[1]] = int(row[0])
    
    return vocab_set

def emailFeatures(word_list,vocab_set):
    feature_vec = np.zeros((len(vocab_set),))
    
    for word in word_list:
        if word in vocab_set:
            feature_vec[vocab_set[word]] = 1
            
    return feature_vec

def processEmail(text):
    word_list = []    
    
    text = text.lower()
    text = re.sub( '<[^<>]+>', ' ', text )
    text = re.sub( '[0-9]+', 'number', text )
    text = re.sub( '(http|https)://[^\s]*', 'httpaddr', text )
    text = re.sub( '[^\s]+@[^\s]+', 'emailaddr', text )
    text = re.sub( '[$]+', 'dollar', text )
    
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']',text)
    for token in tokens:
        token = re.sub( '[^a-zA-Z0-9]', '', token )
        token = stemmer.stem( token.strip() )
        if len(token) > 0:
            word_list.append(token)
        
    return word_list

def computeAccuracy(lsvm,x,y):
    predictions = lsvm.predict(x)
    num_correct = sum (predictions == y)
    accuracy = 100.0 * num_correct / len(y)
    print ("accuracy =", accuracy)

    return accuracy
    
# Start main program  
with open('./Data/emailSample1.txt', 'r' ) as f:
    emails = f.read()   

processed = processEmail(emails)

#print emails 
#print processed

Vocab_set = getVocabSet('./Data/vocab.txt')
#print Vocab_set

features = emailFeatures(processed,Vocab_set)
print ("len of features is", len(features))
print ("num features is", sum(features))
  
vals = sp.io.loadmat('./Data/spamTrain.mat')
X, y = vals['X'], vals['y']
Y = y.ravel()

vals = sp.io.loadmat('./Data/spamTest.mat')
X_test, y_test = vals['Xtest'], vals['ytest']
Y_test = y_test.ravel()

# linear SVM with C = 0.1
if (0):
    linear_svm = svm.SVC(C=0.1,kernel='linear',verbose=False)
    linear_svm.fit(X,Y)
    print ("num support vecs =", linear_svm.n_support_)
    print ("coefs =", linear_svm.coef_)
    print ("intercept =", linear_svm.intercept_)
    pickle.dump(linear_svm,open("linear_svm.svm","wb"))
else:
    linear_svm = pickle.load(open("linear_svm.svm","rb"))

score = computeAccuracy(linear_svm,X,Y)
score = computeAccuracy(linear_svm,X_test,Y_test)

# Determine which words indicate spam
# Find the indices of the most negative support vector coefficients
svecs = linear_svm.coef_.flatten()
sorted_indices = np.argsort(svecs)
#print sorted_indices[0:15]
#print svecs[sorted_indices[0:15]]

# Do inverse look up of Vocab_set
# Flop (key,value) pair in Vocab_set to (value,key) -- it is a one-to-one mapping
Vocab_Set_inv = dict( (v, k) for (k, v) in Vocab_set.items() )
for i in sorted_indices[0:15]:
	print (Vocab_Set_inv[i])
print ("")

# Do end-to-end spam classification of an email
with open('./Data/spamSample2.txt', 'r' ) as f:
    emails = f.read()

processed = processEmail(emails)
raw_features = emailFeatures(processed,Vocab_set)
# Create a [single sample X num_features] matrix
features = raw_features.reshape(1,len(raw_features))

print (emails) 
print (processed)

prediction = linear_svm.predict(features)
print ("prediction=", prediction)
