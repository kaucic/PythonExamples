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

message = "Welcome!"
count = 0

# Define event handlers.

def button_handler():
    """Count number of button presses."""
    global count
    count += 1
    print message,"  You have clicked", count, "times."
    
def input_handler(text):
    """Get text to be displayed."""
    global message
    message = text

# Create frame and register event handlers.

frame = simplegui.create_frame("Home", 100, 200)
frame.add_button("Click me", button_handler)
frame.add_input("New message:", input_handler, 100)

# Start frame.

frame.start()