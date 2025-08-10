
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
import numpy as np

def main() -> None:
    """
    Main entry point for the Bonus Craps program.
    """
    # Simulate millions of rolls
    def check_for_win(rolled: list[bool]) -> bool:
        """
        Check if the player has rolled all possible sums of two dice.
        """
        #return all(rolled[2:7]) & all(rolled[8:13])
        #return all(rolled[8:13]) 
        return all(rolled[2:7])
        #return all(rolled[2:5])
    
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
        if win:
            #print(f"You rolled all numbers")
            number_of_wins += 1
        #if lose:
            #print(f"You rolled a 7, you lose!")
    percent_win = number_of_wins / (i + 1) 
    odds = 1 / percent_win
    expected_value = percent_win * 35 - 1
    #expected_value = percent_win * 176 - 1
    print(f"{number_of_wins} wins out of {i+1} rolls = {percent_win} odds {odds}")
    print(f"Expected value = {expected_value}")
          
def calculate_transition_probabilities234() -> float:
    # Initialize the state transition matrix to 0
    A_ft = np.zeros((9, 9), dtype=float)
    A_ft[0] = [24,  1,  2,  3,  0,  0,  0,  0,  6]
    A_ft[1] = [0,  25,  0,  0,  2,  3,  0,  0,  6]
    A_ft[2] = [0,   0, 26,  0,  1,  0,  3,  0,  6]
    A_ft[3] = [0,   0,  0, 27,  0,  1,  2,  0,  6]
    A_ft[4] = [0,   0,  0,  0, 27,  0,  0,  3,  6]
    A_ft[5] = [0,   0,  0,  0,  0, 28,  0,  2,  6]
    A_ft[6] = [0,   0,  0,  0,  0,  0, 29,  1,  6]
    A_ft[7] = [0,   0,  0,  0,  0,  0,  0, 36,  0]
    A_ft[8] = [0,   0,  0,  0,  0,  0,  0,  0, 36]
    
    # Normalize the transition probabilities
    A_ft = A_ft / 36.0
    #print("Transition probabilities matrix:")
    np.set_printoptions(precision=4, suppress=True)
    #print(A_ft)

    s_0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])  # Initial state vector
    s_1 = np.dot(A_ft.transpose(), s_0)  # State vector after one transition
    s_2 = np.dot(A_ft.transpose(), s_1)  # State vector after two transitions
    s_3 = np.dot(A_ft.transpose(), s_2)  # State vector after three transitions
    A_ft_100 = np.linalg.matrix_power(A_ft, 100)  # Raise the transition matrix to the 100th power
    s_final = np.dot(A_ft_100.transpose(), s_0)  # Final state vector
    print("State vectors")
    print(s_0)
    print(s_1)
    print(s_2)
    print(s_3)      
    print(s_final)
    return s_final[7]

def calculate_transition_probabilities() -> float:
    # Initialize the state transition matrix to 0
    A_ft = np.zeros((33, 33), dtype=float)
    A_ft[0]  = [15, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[1]  = [0, 16, 0, 0, 0, 0, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[2]  = [0, 0, 17, 0, 0, 0, 1, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[3]  = [0, 0, 0, 18, 0, 0, 0, 1, 0, 0, 2, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[4]  = [0, 0, 0, 0, 19, 0, 0, 0, 1, 0, 0, 2, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[5]  = [0, 0, 0, 0, 0, 20, 0, 0, 0, 1, 0, 0, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]

    A_ft[6]  = [0, 0, 0, 0, 0, 0, 18, 0,  0,  0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[7]  = [0, 0, 0, 0, 0, 0,  0, 19, 0,  0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[8]  = [0, 0, 0, 0, 0, 0,  0, 0, 20,  0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[9]  = [0, 0, 0, 0, 0, 0,  0, 0,  0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[10] = [0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 20, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[11] = [0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 21, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[12] = [0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 22, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0,  0, 6]
    A_ft[13] = [0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 22, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0,  0, 6]
    A_ft[14] = [0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0,  0, 6]
    A_ft[15] = [0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 1, 0, 0, 2, 3, 0, 0, 0, 0, 0,  0, 6]

    A_ft[16] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0,  0, 6]
    A_ft[17] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 5, 0,  0, 6]
    A_ft[18] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0,  0, 6]
    A_ft[19] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 2, 0, 5, 0, 0,  0, 6]
    A_ft[20] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0,  0, 6]
    A_ft[21] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 3, 2, 0,  0, 6]
    A_ft[22] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 1, 0, 0, 0, 5,  0, 6]
    A_ft[23] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 1, 0, 0, 4,  0, 6]
    A_ft[24] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 1, 3,  0, 6]
    A_ft[25] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 1, 0, 2,  0, 6]
    
    A_ft[26] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0,  0,  0,  0, 5, 6]
    A_ft[27] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26,  0,  0,  0, 4, 6]
    A_ft[28] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 27,  0,  0, 3, 6]
    A_ft[29] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 28,  0, 2, 6]
    A_ft[30] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 29, 1, 6]

    A_ft[31] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 36, 0]
    A_ft[32] = [0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 36]
     
    # Normalize the transition probabilities
    A_ft = A_ft / 36.0
    #print("Transition probabilities matrix:")
    np.set_printoptions(precision=4, suppress=True)
    #print(A_ft)

    s_0 = np.zeros(33); s_0[0] = 1.0  # Initial state vector
    s_1 = np.dot(A_ft.transpose(), s_0)  # State vector after one transition
    s_2 = np.dot(A_ft.transpose(), s_1)  # State vector after two transitions
    s_3 = np.dot(A_ft.transpose(), s_2)  # State vector after three transitions
    A_ft_100 = np.linalg.matrix_power(A_ft, 100)  # Raise the transition matrix to the 100th power
    s_final = np.dot(A_ft_100.transpose(), s_0)  # Final state vector
    print("State vectors")
    print(s_0)
    print(s_1)
    print(s_2)
    print(s_3)      
    print(s_final)
    return s_final[31]

if __name__ == "__main__":
    main()
    #prob_win = calculate_transition_probabilities234()
    #print(f"Probability of winning 234: {prob_win}")
    prob_win = calculate_transition_probabilities()
    odds = 1 / prob_win
    print(f"Probability of winning all small: {prob_win}, Odds: {odds}")

