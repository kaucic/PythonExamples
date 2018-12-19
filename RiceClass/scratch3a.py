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

# Define globals.

message = "1lll1l1l1l1ll1l111ll1l1ll1l1ll1ll111ll1ll1ll1l1ll1ll1ll1ll1lll1l1l1l1l1l1l1l1l1l1l1l1ll1lll1l111ll1l1l1l1l1"
length = len(message)
print "length=",length

lcount = 0
onecount = 0

for letter in message:
    if (letter == "l"):
        lcount += 1
    elif (letter == "1"):
        onecount += 1
    else:
        print letter,"not a one or an l"

print "onecount=%d lcount=%d" % (onecount, lcount)

# Define event handlers.
def draw(canvas):
    canvas.draw_circle([90,200],20,10,"White")
    canvas.draw_circle([210,200],20,10,"White")
    canvas.draw_line([50,180],[250,180],40,"Red")
    canvas.draw_line([55,170],[90,120],5,"Red")
    canvas.draw_line([90,120],[120,120],5,"Red")
    canvas.draw_line([180,108],[180,160],140,"Red")
  
# Create frame and register event handlers.
frame = simplegui.create_frame("Home", 300, 300)
frame.set_draw_handler(draw)

# Start frame.
frame.start()

