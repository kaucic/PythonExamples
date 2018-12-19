# -*- coding: utf-8 -*-
"""
Created on Wed Sep 09 20:15:02 2015

@author: Kimberly
"""

import math

def next(x):
    return (x ** 2 + 79) % 997

x = 1
x_set = set([x])
for i in range(1000):
    print x
    x = next(x)
    x_set.add(x)
    
print "Number of elements is x_set", len(x_set)
   
"""    
try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui 
    
def draw_handler(canvas):
    canvas.draw_image(image, (220,100), (100, 100), (50, 50), (100, 100))

image = simplegui.load_image('http://commondatastorage.googleapis.com/codeskulptor-assets/alphatest.png')

frame = simplegui.create_frame('Testing', 300, 300)
frame.set_draw_handler(draw_handler)
frame.start()
"""
