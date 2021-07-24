# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:12:24 2021

@author: home
"""

import logging
import doLogging

import numpy as np

import matplotlib.pyplot as plt

class SimpleHash:
    def __init__(self):
        return

    # Page 67 CTCI
    # Given an array of distinct integer values, count the number of pairs of integers
    # that have a difference of k.
    # Given [1, 7, 5, 9, 2, 12, 6, 3, 4] with k=2
    # return (1,3) (3,5) (5,7) (7,9) (4,6) (2,4)
    def run(self):
        logging.info(f"Starting")
        A = [1, 7, 5, 9, 2, 12, 6, 3, 4]
        k = 2
        all_nums = {}
        # Populate hash table
        for a in A:
            all_nums[a] = True;
        
        # Find pairs
        all_pairs = []
        for key, val in all_nums.items():
            if key+k in all_nums:
                all_pairs.append([key,key+k])

        # print all pairs found
        print (f"The numbers in {A} that are {k} apart are")
        for i in range(len(all_pairs)):
            print (all_pairs[i])
        
if __name__ == "__main__":
    doLogging.set_up_logger()
    A = SimpleHash()
    A.run()   
    doLogging.clean_up_logger()
    