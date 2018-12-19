# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 13:28:39 2016

@author: Kimberly
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

COMMISSION = 0.03

def plot_income(mon,sales,color):
    plt.plot(sales/1000,mon,color,markersize=5 )
    plt.ylabel('Income ($)')
    plt.xlabel('Total Closed Sales ($K)')


def compute_compensation(percentage,threshold,sales):
    if sales < threshold:
        commish = sales * COMMISSION * percentage
    else:
        commish = threshold * COMMISSION * percentage
        commish += ( (sales-threshold) * COMMISSION * 0.8)
        
    return commish
    
rates = np.array([0.45,0.5,0.55,0.6])
limit = np.array([24475,33425,40110,54075])
colors = ['r-', 'g-', 'b-', 'k-']

s = np.arange(0,3.5*10**6,10**5)
money = np.zeros(s.shape)

for r in range(len(rates)):
    for i in range(len(s)):
        cutoff = limit[r] / rates[r] / COMMISSION
        money[i] = compute_compensation(rates[r],cutoff,s[i])
    
    plot_income(money,s,colors[r])

plt.savefig('income.jpg')
        