# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:53:36 2015

@author: Kimberly
"""
# 1. Create frame
# 2. Define classes
# 3. Define event handlers
# 4. Initialize global variables
# 5. Define helper functions
# 6. Register event handlers
# 7. Start frame and timers

# 4 5 2 3 1 6 7 

# Simple interactive application

try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

import time
from datetime import date

# Define globals.

now = time.time()
print "now=%f" % now

origin = date.fromtimestamp(0)
print "origin=", origin
print "min date=", date.min

# Mystery computation in Python
# Takes input n and computes output named result

# global state

result = 1
iteration = 0
max_iterations = 40

# helper functions

def init(start):
    """Initializes n."""
    global result
    result = start
    print "Input is", result
    
def get_next(current):
    """???  Part of mystery computation."""
    next_val = 0
    if (current % 2 == 0):
        next_val = current / 2
    else:
        next_val = 3 * current + 1
    return next_val

# timer callback

def update():
    """???  Part of mystery computation."""
    global iteration, result
    iteration += 1
    # Stop iterating after max_iterations
    if iteration >= max_iterations:
        timer.stop()
        print "Output is", result
    else:
        result = get_next(result)
    print "result=", result

# register event handlers

timer = simplegui.create_timer(1, update)

# start program
init(217)
timer.start()
print "\n"