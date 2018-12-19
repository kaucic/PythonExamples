# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:02:51 2015

@author: Kimberly
"""

# Rock-paper-scissors-lizard-Spock template


# The key idea of this program is to equate the strings
# "rock", "paper", "scissors", "lizard", "Spock" to numbers
# as follows:
#
# 0 - rock
# 1 - Spock
# 2 - paper
# 3 - lizard
# 4 - scissors

import random

Names = ('rock','Spock','paper','lizard','scissors')
Length = len(Names)

def name_to_number(name):
    idx = -1
    for i in range(len(Names)):
        if (Names[i] == name):
            idx = i
    if (idx == -1):
        print "Error: ", name, " not in ", Names
    return (idx)
    
    
def number_to_name(number):
    name = ""
    if (number < 0 or number >= len(Names)):
        print "Error: Invalid index=", number
    else:
        name = Names[number]
    return (name)
    

def rpsls(player_choice):
    player_number = name_to_number(player_choice)
    print "Player chooses ", player_choice, " with index=", player_number
    
    comp_number = random.randrange(len(Names))
    print "Computer chooses ", number_to_name(comp_number), " with index=", comp_number
    
    # Clever way to determine who wins
    # Based on ordered choices, N loses to N+1 and N+2, but beats N+3 and N+4
    # So, use modolo arithmetic with the player winning when the remainder is 3 or 4
    difference = (player_number - comp_number) % Length
    
    if (difference == 0):
        print "Player and Computer tie!"
    elif (difference < 3):
        print "Player wins!"
    else:
        print "Computer wins!"
    
    print "\n"

    
# test your code - THESE CALLS MUST BE PRESENT IN YOUR SUBMITTED CODE
rpsls("rock")
rpsls("Spock")
rpsls("paper")
rpsls("lizard")
rpsls("scissors")

# always remember to check your completed program against the grading rubric


