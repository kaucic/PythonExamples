# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:16:03 2015

@author: Kimberly
"""

# template for "Guess the number" mini-project
# input will come from buttons and an input field
# all output for the game will be printed in the console

try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

import random
import math

secret_number = 0
guesses_left = 0

# helper function to start and restart the game
def new_game(max_num=100):
    global secret_number,guesses_left
    secret_number = random.randrange(max_num)
    guesses_left = int(math.ceil(math.log(max_num,2)))
    print "\n"    
    print "Starting game: secret_number=",secret_number," guesses_left=",guesses_left,"\n"

# define event handlers for control panel
def range100():
    # button that changes the range to [0,100) and starts a new game 
    new_game(100)

def range1000():
    # button that changes the range to [0,1000) and starts a new game     
    new_game(1000)
    
def input_guess(guess):
    # main game logic goes here
    global guesses_left
    num = int(guess)
    print "Guess was ",num,"\n"
    if (secret_number > num):
        print "Higher\n"
    elif (secret_number < num):
        print "Lower\n"
    else:
        print "Correct, You win!!\n"
        new_game()
        
    guesses_left -= 1
    if (guesses_left > 0):
        print "You have ",guesses_left," guesses left\n"
    else:
        print "You exhausted your guesses.  Better luck next time\n"
        new_game()
    
# create frame
frame = simplegui.create_frame("Guess the number",200,200)

# register event handlers for control elements and start frame
frame.add_button("Range is [0,100)",range100)
frame.add_button("Range is [0,1000)",range1000)
frame.add_input("Guess",input_guess,50)
frame.start()

# call new_game 
new_game()


# always remember to check your completed program against the grading rubric
