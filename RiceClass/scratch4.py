# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:59:08 2015

@author: Kimberly
"""

try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

import math

pt = [4.0,7.0]
c = [2.0,9.0]
r = 2.0

d = math.sqrt((pt[0]-c[0])**2 + (pt[1]-c[1])**2) - r
print d

d = 5
count = 0

def keydown(key):
    global d,count 
    d *= 2
    count += 1
    print "count=%d,d=%d" % (count,d)
    
def keyup(key):
    global d,count
    d -= 3
    count += 1
    print "count=%d,d=%d" % (count,d)

def draw_handler(canvas):
    global pt,c
    pt = [10,20]
    c = [30,7]
    end_pt = [0,0]
    end_pt[0] = pt[0] + 9*c[0]
    end_pt[1] = pt[1] + 9*c[1]
    
    canvas.draw_polygon([(50,50), (180, 50), (180, 140), (50, 140)],1,"Red")
    canvas.draw_line(pt,end_pt,1,"Blue")

frame = simplegui.create_frame('Testing', 300, 300)
frame.set_draw_handler(draw_handler)
frame.set_keydown_handler(keydown)
frame.set_keyup_handler(keyup)

frame.start()