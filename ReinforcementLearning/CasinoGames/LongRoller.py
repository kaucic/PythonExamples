"""
LongRoller.py
Template for a Python main program.
Author: Bob Kaucic
Date: 2025-08-10
Description: Simulation to determine typical number of dice rolls for craps players
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    """
    Main entry point for the program.
    """
    # Simulate thousands of turns
    N = 1000
    num_rolls_per_turn = [0] * N
    points_made_per_turn = [0] * N
    for i in range(N):
        seven_out = False
        have_point = False
        point = 0
        while not seven_out:
            num_rolls_per_turn[i] += 1
            # Simulate rolling two dice
            die1 = random.randint(1, 6)
            die2 = random.randint(1, 6)
            total = die1 + die2
            #print(f"Roll {num_rolls_per_turn[i]}: You rolled {die1} and {die2} = {total}")
            if have_point and total == 7:
                seven_out = True
                #print(f"Turn {i + 1} ended with a 7, number of rolls: {num_rolls_per_turn[i]}")
            elif have_point and total == point:
                points_made_per_turn[i] += 1
                have_point = False
                point = 0 
            elif not have_point and total in [4, 5, 6, 8, 9, 10]:
                have_point = True
                point = total
            else:
                # Continue rolling until a point is established
                continue

    # Calculate average number of rolls per turn
    #print(f"Number of rolls per turn: {num_rolls_per_turn}")
    #print(f"Number of points made per turn: {points_made_per_turn}")
    average_rolls = np.mean(num_rolls_per_turn)
    average_points = np.mean(points_made_per_turn)
    percentage_gt_24 = sum(1 for rolls in num_rolls_per_turn if rolls > 24) / N * 100
    print(f"Average number of rolls per turn: {average_rolls}, average points made per turn: {average_points}")
    print(f"Percentage of turns with 25 or more rolls: {percentage_gt_24}")

    # Plot histogram of the number of rolls per turn
    plt.hist(num_rolls_per_turn, bins=range(min(num_rolls_per_turn), max(num_rolls_per_turn)+2), edgecolor='black')
    plt.xlabel('Number of Rolls per Turn')
    plt.ylabel('Frequency')
    plt.title('Histogram of Number of Rolls per Turn')
    plt.show()
    
if __name__ == "__main__":
    main()