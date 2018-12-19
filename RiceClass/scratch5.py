# -*- coding: utf-8 -*-
"""
Created on Wed Sep 09 20:15:02 2015

@author: Kimberly
"""

import math

l = list([0,1])
for i in range(40):
    l.append(l[i]+l[i+1])
    print l
   
def euclid(num1, num2):
    while num2 != 0:
        r = remainder(num1, num2)
        num1 = num2
        num2 = r
    return num1

def remainder(num1, num2):
    x = num1
    while x >= 0:
        x -= num2
    x += num2
    return x
 
def remainder2(num1,num2):
    return (num1 % num2)
   
print remainder(45,12)
print remainder2(45,12)

print euclid(79170,3465)
   
   
   
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
