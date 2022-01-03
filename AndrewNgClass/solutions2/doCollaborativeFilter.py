# -*- coding: utf-8 -*-
"""
Created on Sun Jan 2 18:12:24 2022

@author: home
"""

import logging
import doLogging

class CollaborativeFilter:
    def __init__(self):
        return

    def run(self)-> None:
        print (f"Completed Successfully")

# Start main program
if __name__ == "__main__":
    doLogging.set_up_logger()
    inst = CollaborativeFilter()
    inst.run()
    doLogging.clean_up_logger()