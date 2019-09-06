# This is a simulation of multiple strategies in a game of Two-Coin Farkle.
#
# The game is played with two coins.
#
# Players play in turns. A turn consists of a sequence of flips of one or two
# coins. A turn begins with two coin flips, resulting in three possible outcomes:
#
# * An award of m1 is temporarily awarded if the two coin flips result in a single heads.
# * An award of m2 is temporarily awarded if the two coin flips result in two heads.
# * A "Farkle" occurs if the flips result in no heads. In this case, the player's turn is
#   over, and the player forfeits all the temporary rewards accumulated during his or her
#   turn.
#
# If a "Farkle" did not happen after the initial flip of two coins, the player may
# choose to stop and collect his or her reward, which then becomes his final reward for
# that turn. Alternatively, if the inital flip of two coins resulted in a single heads,
# the player may choose to flip the coin that resulted in a tails. If the player so
# chooses, two outcomes are possible:
#
# * If the new flip of the coin results in heads, an award of m1 is temporarily added to
#   the player's current temporary award.
# * If the flip results in tails, a "Farkle" occurs. The player's turn is over, and the
#   player forfeits all the temporary rewards accumulated during his or her turn.
#
# If a "Farkle" did not after the flip of the single coin, the player may choose to stop
# and collect his or her reward, which then becomes his final reward for that turn.
# Alternatively, the player may choose to flip two coins again, following the pattern
# started at the beginning of his or her turn.
#
# If the initial flip of two coins results in two heads, the player may choose to stop
# and collect the reward accumulated thus far in his or her turn. Otherwise, the player
# may choose to flip the two coins again, following the pattern started at the beginning
# of his or her turn.

import numpy as np
import pdb

class Coin:
    def __init__(self, seed = 0, p = 0.5):
        self._p = p
        if seed is not None:
            self.Reset(seed)

    def Flip(self):
        p = self._p
        state = np.random.choice([1, 0], p = [p, 1-p]).astype(int)
        return state

    def Reset(self, seed):
        self._seed = seed
        np.random.seed(self._seed)

def EvaluateResult(result_a, result_b = 0):
    return result_a + result_b

class Player():
    def __init__(self, m1, m2, debug = False):
        self._m1 = m1
        self._m2 = m2
        self._earnings_in_turn = 0
        self._total_reward = 0
        self._debug = debug
        
class OptimalPlayer(Player):
    def __init__(self, m1 = 1, m2 = 100):
        Player.__init__(self, m1, m2)
        
    def Simulate(self, p, p_1_0, p_2_0, M_1_0, M_2_0, max_iterations):
        iterations = 0
        initial_reward = p_1_0*M_1_0 + p_2_0*M_2_0
        if self._debug:
            print("\tp1 = %f, p2 = %f, M1 = %f, M2 = %f" % (p_1_0, p_2_0, M_1_0, M_2_0))
            print("\tinitial reward = %f" % initial_reward)

        while iterations < max_iterations:
            iterations = iterations + 1
            
            # Update state probabilities
            p_1_1 = 2*p*(1 - p)*p_2_0
            aux_a = p*p_1_0
            aux_b = p*p*p_2_0
            p_2_1 = aux_a + aux_b
            
            p_1_0 = p_1_1
            p_2_0 = p_2_1            
            
            # Update state rewards
            M_1_1 = M_2_0 + self._m1
            M_2_1 = (aux_a*(M_1_0 + self._m1) + aux_b*(M_2_0 + self._m2))/p_2_1
            
            M_1_0 = M_1_1
            M_2_0 = M_2_1
            
            new_reward = p_1_0*M_1_0 + p_2_0*M_2_0
            if self._debug:
                print("\tp1 = %f, p2 = %f, M1 = %f, M2 = %f" % (p_1_0, p_2_0, M_1_0, M_2_0))
                print("\tnew reward = %f" % new_reward)
            if new_reward > initial_reward:
                return True
            
        return False
    
    def Turn(self, coin, max_iterations):
        self._earnings_in_turn = 0
        
        # Start the turn flipping two coins.
        state = 2
        # Start turn with zero reward
        reward = 0
        
        iterations = 0
        while True:
            result_a = coin.Flip()
            if state == 2:
                # Two-coin flip.
                result_b = coin.Flip()
            else:
                # state == 1
                # Single coin flip.
                result_b = 0
                
            # Evaluate the outcome
            new_state = EvaluateResult(result_a, result_b)
            
            if new_state == 0:
                # Farkle
                return 0
            
            if new_state == 1:
                p_1_0 = 1
                p_2_0 = 0
                
                reward = reward + self._m1
                M_1_0 = reward
                M_2_0 = 0
                
            else:
                p_1_0 = 0
                p_2_0 = 1
                
                reward = reward + self._m2
                M_1_0 = 0
                M_2_0 = reward
                
            # Evaluate future reward to determine whether turn should continue
            keep_going = self.Simulate(coin._p, p_1_0, p_2_0, M_1_0, M_2_0, max_iterations)
            if keep_going == False:
                self._earnings_in_turn = reward
                self._total_reward = self._total_reward + self._earnings_in_turn
                return new_state
            
            else:
                if self._debug:
                    print("Smarty player has a reward of %f, and decided to keep going" % reward)
                if state == 1 and new_state == 1:
                    # We flipped a single coin, and now we have two heads.
                    state = 2
                elif state == 2 and new_state == 1:
                    # We flipped two coins, and now we have a single head
                    state = 1

class GreedyPlayer(Player):
    def __init__(self, m1 = 1, m2 = 10):
        Player.__init__(self, m1, m2)
        
    def Turn(self, coin):
        self._earnings_in_turn = 0
        
        # Start turn by flipping two coins
        state = 2
        # Start turn with zero reward
        reward = 0
        
        while True:
            result_a = coin.Flip()
            if state == 2:
                # Two-coin flip
                result_b = coin.Flip()
            else:
                # State == 1
                result_b = 0

            # Evaluate the outcome
            new_state = EvaluateResult(result_a, result_b)

            if new_state == 0:
                return 0

            if new_state == 2:
                self._earnings_in_turn = reward + self._m2
                self._total_reward = self._total_reward + self._earnings_in_turn
                return 2

            # New state == 1
            reward = reward + self._m1
            if state == 1:
                # A single coin was flipped. We are back to state 2
                state = 2
            else:
                # Two coins were flipped. We move on on state 1.
                state = 1


class ConservativePlayer(Player):
    def __init__(self, m1 = 1, m2 = 10):
        Player.__init__(self, m1, m2)
        
    def Turn(self, coin):
        self._earnings_in_turn = 0
        
        result_a = coin.Flip()
        result_b = coin.Flip()
        new_state = EvaluateResult(result_a, result_b)
        
        if new_state == 0:
            return 0
        
        if new_state == 2:
            self._earnings_in_turn =  self._m2
            self._total_reward = self._total_reward + self._earnings_in_turn
            return 2
        
        # New state == 1
        self._earnings_in_turn = self._m1
        self._total_reward = self._total_reward + self._earnings_in_turn
        return 1
        
                
# Dice
p = 1.0/6
# Coins
# p = 0.5

seed = 0
coin = Coin(seed = seed, p = p)
m1 = 1
m2 = 100

consvy = ConservativePlayer(m1, m2)
greedy = GreedyPlayer(m1, m2)
smarty = OptimalPlayer(m1, m2)

debug = False

max_iterations = 20
# Play a few turns
N_turns = 50000
for i in range(N_turns):
    if debug:
        print("Turn %d" % (i + 1))

        print("Greedy player has started")

    greedy_state = greedy.Turn(coin)
    
    if debug:
        print("Earnings in turn: %d" % greedy._earnings_in_turn)
        if greedy_state == 0:
            print("Greedy player has farkled. Total reward is %d" % greedy._total_reward)
        else:
            print("Greedy player stopped. Total reward is %d" % greedy._total_reward)
            
        print("")

        print("Smarty player has started")
        
    smarty_state = smarty.Turn(coin, max_iterations)

    if debug:
        print("Earnings in turn: %d" % smarty._earnings_in_turn)
        if smarty_state == 0:
            print("Smarty player has farkled. Total reward is %d" % smarty._total_reward)
        elif smarty_state == 1:
            print("Smarty player stopped on a single flip. Total reward is %d" % smarty._total_reward)
        else:
            print("Smarty player stopped on a double flip. Total reward is %d" % smarty._total_reward)

    consvy_state = consvy.Turn(coin)

    if debug:
        print("Earnings in turn: %d" % consvy._earnings_in_turn)
        if consvy_state == 0:
            print("Consvy player has farkled. Total reward is %d" % consvy._total_reward)
        elif consvy_state == 1:
            print("Consvy player stopped on a single flip. Total reward is %d" % consvy._total_reward)
        else:
            print("Consvy player stopped on a double flip. Total reward is %d" % consvy._total_reward) 

        print("******************")

print("")
#print("Greedy player payout: %d" % greedy._total_reward)
#print("Optimal player payout: %d" % smarty._total_reward)
#print("Conservative player payout: %d" % consvy._total_reward)

print("Greedy player point's per roll: %f" % (1.0*greedy._total_reward/N_turns))
print("Optimal player point's per roll: %f" % (1.0*smarty._total_reward/N_turns))
print("Conservative player point's per roll: %f" % (1.0*consvy._total_reward/N_turns))

print("Big payout: %f" % (m2/36.0))
print("Conservative payout: %f" % (m2/36.0 + 10*m1/36.0)) 

delta = 0
number_of_turns = 0
try:
    delta
except NameError:
    delta = 0

try:
    number_of_turns
except NameError:
    number_of_turns = 0

number_of_turns = number_of_turns + N_turns
delta = delta + smarty._total_reward - greedy._total_reward
    
print("Cummulative advantage of optimal strategy after %d turns: %d" % (number_of_turns, delta))
