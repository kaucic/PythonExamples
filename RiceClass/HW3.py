# -*- coding: utf-8 -*-
"""
Create, run, and display a stopwatch using timers

Created on Sat Sep 19 22:42:04 2015

@author: Kimberly
"""

try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

# template for "Stopwatch: The Game"

# define global variables
secondsX10 = 0
MAXTIME = 10 * 600 # 9:59.9 is the largest time
running = False
correct = 0
tries = 0


# define helper function format that converts time
# in tenths of seconds into formatted string A:BC.D
def format(t):
    minutes = t / 600
    seconds = (t - 600*minutes) / 10 
    tenths = t % 10
    sec_str = "%02d" % seconds
    msg = str(minutes) + ":" + sec_str + "." + str(tenths)
    return msg
    
# define event handlers for buttons; "Start", "Stop", "Reset"
def start():
    global running
    timer.start()
    running = True

def stop():
    global running, tries, correct
    timer.stop()
    if (running):  # The player is attempting to stop on a whole second
        tries += 1
        if (secondsX10 % 10 == 0):
            correct += 1
    running = False

def reset():
    global secondsX10, tries, correct
    stop()
    secondsX10 = 0
    tries = 0
    correct = 0

# define event handler for timer with 0.1 sec interval
def tick():
    global secondsX10
    secondsX10 = (secondsX10 + 1) % MAXTIME

# define draw handler
def draw(canvas):
    canvas.draw_text(format(secondsX10), [100, 100], 36, "Red")
    canvas.draw_text("%d/%d" % (correct,tries), [250,20], 18, "Blue")
    
# create frame
frame = simplegui.create_frame("Stopwatch",300,200)

# register event handlers
frame.set_draw_handler(draw)
timer = simplegui.create_timer(100,tick)
frame.add_button("Start",start)
frame.add_button("Stop",stop)
frame.add_button("Reset",reset)

# start frame
frame.start()

# Please remember to review the grading rubric
