
"""
BonusCraps.py
Template for a Python main program.
Author: 
Date: 2025-08-09
Description: Entry point for the Bonus Craps simulation or application.
"""

import sys
import os
import random

def main() -> None:
    """
    Main entry point for the Bonus Craps program.
    """
    # Simulate millions of rolls
    print("Welcome to Bonus Craps!")

    def check_for_win(rolled: list[bool]) -> bool:
        """
        Check if the player has rolled all possible sums of two dice.
        """
        return all(rolled[2:7]) & all(rolled[8:13])
        #return all(rolled[8:13]) 
        #return all(rolled[2:7])
    
    number_of_wins = 0
    for i in range(1000000):
        win = False
        lose = False
        # Initialize a boolean array 'rolled' of length 13 to False
        rolled = [False] * 13
        while not (win or lose):
            # Simulate rolling two dice
            die1 = random.randint(1, 6)
            die2 = random.randint(1, 6)
            total = die1 + die2
            #print(f"Roll {i + 1}: You rolled {die1} and {die2} = {total}")
            if total == 7:
                lose = True
            else:
                rolled[total] = True
                win = check_for_win(rolled)
        #if lose:
            #print(f"You rolled a 7, you lose!")
        if win:
            #print(f"You rolled all numbers")
            number_of_wins += 1
    percent_win = number_of_wins / (i + 1) 
    odds = 1 / percent_win
    #expected_value = percent_win * 35 - 1
    expected_value = percent_win * 176 - 1
    print(f"{number_of_wins} wins out of {i+1} rolls = {percent_win} odds {odds}")
    print(f"Expected value = {expected_value}")
          
if __name__ == "__main__":
    main()
