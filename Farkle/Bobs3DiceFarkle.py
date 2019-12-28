# This is a simulation of multiple strategies in a game of N = Three Dice Farkle.
#
# The game is played with N dice.
#
# Players play in turns. A turn consists of a sequence of one or more rolls of dice. A turn
# begins with a roll of all dice.  Outcomes:
#
# 1. Points can be scored in multiple ways and players can continue rolling
# as long as one or more dice score points
# 
# 2. A "Farkle" occurs if no dice score points. In this case, the player's turn is
#  over and the player forfeits all the temporary rewards accumulated during his or her
#  turn.
#
# 3. If a "Farkle" did not happen after the initial roll, the player may
# choose to stop and collect his or her reward, which then becomes his final reward for
# that turn. Alternatively, if one or more dice resulted in points,
# the player may choose to roll again excluding the dice that scored points.
# 
# 4. If the player successfully scores points will all N dice, he/she can
# roll all N dice again.
#
# Probabilites for Farkleing with X Dice
# 6 --  2.31%
# 5 --  7.72%
# 4 -- 15.74%
# 3 -- 27.78%
# 2 -- 44.44%
# 1 -- 66.67%

import numpy as np
# import pdb

# Works for 3 or less dice
# dice_vals are a vector of 1-N dice, all containing a value of 1 to 6
def EvaluateResult(dice_vals):
    N = dice_vals.size
    #print (N, " dice evaluated with roll of ", dice_vals)
    score = 0.0
    
    if N == 3:
        # find triples
        if (dice_vals[0] == dice_vals[1]) and (dice_vals[1] == dice_vals[2]):
            if dice_vals[0] == 1:
                score += 300.0;
            else:
                score += (dice_vals[0] * 100.0)
        else:
            # score 50 for a 5 or 100 for a 1
            if (dice_vals[0] == 1) or (dice_vals[0] == 5):
                score += (-12.5*dice_vals[0] + 112.5)
            if (dice_vals[1] == 1) or (dice_vals[1] == 5):
                score += (-12.5*dice_vals[1] + 112.5)
            if (dice_vals[2] == 1) or (dice_vals[2] == 5):
                score += (-12.5*dice_vals[2] + 112.5)
            
    elif N == 2:
        # score 50 for a 5 or 100 for a 1
        if (dice_vals[0] == 1) or (dice_vals[0] == 5):
            score += (-12.5*dice_vals[0] + 112.5)
        if (dice_vals[1] == 1) or (dice_vals[1] == 5):
            score += (-12.5*dice_vals[1] + 112.5)
            
    elif N == 1:
        # score 50 for a 5 or 100 for a 1
        if (dice_vals[0] == 1) or (dice_vals[0] == 5):
            score += (-12.5*dice_vals[0] + 112.5)
            
    else:
        print ("ERROR, In EvaluateResult num dice is ", N)
        
    #print ("Score is ", score )
    #if (score == 0):
        #print ("You Farkled!")
        
    return score
    
class Dice:
    def __init__(self, N = 3, seed = None):
        self._N = N
        if seed is not None:
            self.Reset(seed)

    def Roll(self):
        dice_vals = np.random.randint(1,6,self._N)
        if debug:
            print ("You rolled ", dice_vals)
        return dice_vals

    def Reset(self, seed):
        self._seed = seed
        np.random.seed(self._seed)

class Player():
    def __init__(self, debug = False):
        self._earnings_in_turn = 0
        self._total_reward = 0
        self._debug = debug
    
    # Create a table of the expected value for possibilities of
    # [states] and [actions]  for Ndice number of dice
    # Actions are keep_first_die, keep_second_die, ... and roll_again     
    def CreateQTables(self, Ndice):
        if Ndice == 3:
            Q_sa = np.zeros((6,), dtype='f,f,f,f')
        elif Ndice == 2:
             Q_sa = np.zeros(6,6,Ndice+1)
        elif Ndice == 1:
             Q_sa = np.zeros(6,Ndice+1)
        else:
            print ("ERROR, In CreateQTables num dice is ", Ndice)
            
        return Q_sa

# Determine the optimal play through experimentation        
class OptimalPlayer(Player):
    def __init__(self, debug = False):
        Player.__init__(self, debug)
        #for n in range(self._N,0,-1):
            #self._Q_sa[n] = self.CreateQTables(n)
            
#        self._Q3_sa = self.CreateQTables(3)
#        self._Q2_sa = self.CreateQTables(2)
#        self._Q1_sa = self.CreateQTables(1)
#        
#        if debug:
#            print ("Q3_sa = ", self._Q3_sa)
#            print ("Q2_sa = ", self._Q2_sa)
#            print ("Q1_sa = ", self._Q1_sa)
        
    def Simulate(self):
        return False
    
    # Updates _earnings_in_turn as by product
    def SelectAction(self,dice_vals):
        keep_dice = [True, True, True]
        self._earnings_in_turn = 96.0
        keep_rolling = False
        return keep_rolling, dice_vals
        
    def DoTurn(self):
        self._earnings_in_turn = 0
        
        dice = Dice(3)
        dice_vals = dice.Roll()
        # Check for Farkle
        if EvaluateResult(dice_vals) == 0:
            self._total_reward = 0
            print ("Optimal player Farkled with %d dice" % dice_vals.size)
            return 0.0
        
        keep_rolling, kept_dice =  self.SelectAction(dice_vals)
        self._total_reward = self._earnings_in_turn
        return self._total_reward 
    
# Keep rolliing until you use up all dice (and then roll once more)        
class GreedyPlayer(Player):
    def __init__(self, debug = False):
        Player.__init__(self, debug)
        
    # Keep rolliing until you use up all dice (and then roll once more)
    # Updates _earnings_in_turn as by product
    def SelectAction(self,dice_vals):
        keep_rolling = False
        N = dice_vals.size
        
        if N == 3:
            keep_dice = [False, False, False]
            
            # find triples
            if (dice_vals[0] == dice_vals[1]) and (dice_vals[1] == dice_vals[2]):
                keep_dice = [True, True, True]
            # look for single dice scoring
            else:
                if (dice_vals[0] == 1) or (dice_vals[0] == 5):
                    keep_dice[0] = True
                if (dice_vals[1] == 1) or (dice_vals[1] == 5):
                    keep_dice[1] = True
                if (dice_vals[2] == 1) or (dice_vals[2] == 5):
                    keep_dice[2] = True
            
            dice_to_keep = np.multiply(dice_vals,keep_dice)
            #print ("dice_to_keep", dice_to_keep)
            kept_dice = dice_to_keep[np.nonzero(dice_to_keep)]
            if self._debug:
                print ("kept_dice", kept_dice)
            self._earnings_in_turn += EvaluateResult(kept_dice)
            if kept_dice.size != 3:
                keep_rolling = True
            return keep_rolling, kept_dice
        
        elif N == 2:
            keep_dice = [False, False]
            # look for single dice scoring
            if (dice_vals[0] == 1) or (dice_vals[0] == 5):
                keep_dice[0] = True
            if (dice_vals[1] == 1) or (dice_vals[1] == 5):
                keep_dice[1] = True
            dice_to_keep = np.multiply(dice_vals,keep_dice)
            #print ("dice_to_keep", dice_to_keep)
            kept_dice = dice_to_keep[np.nonzero(dice_to_keep)]
            if self._debug:
                print ("kept_dice", kept_dice)
            self._earnings_in_turn += EvaluateResult(kept_dice)
            if kept_dice.size != 2:
                keep_rolling = True
            return keep_rolling, kept_dice
                
        elif N == 1:
            self._earnings_in_turn += EvaluateResult(dice_vals)
            keep_rolling = False
            return keep_rolling, dice_vals
        
        else:
            print ("ERROR, In SelectAction num dice is ", N)
            
    # Do a turn and return the score accumulated    
    def DoTurn(self):
        self._earnings_in_turn = 0
        
        dice = Dice(3)
        dice_vals = dice.Roll()
        # Check for Farkle
        if EvaluateResult(dice_vals) == 0:
            self._total_reward = 0
            print ("Greedy player Farkled with %d dice" % dice_vals.size)
            return 0.0        
        
        keep_rolling, kept_dice =  self.SelectAction(dice_vals)
        if keep_rolling == True:
            if kept_dice.size == 1:
                dice = Dice(2)
                dice_vals = dice.Roll()
                # Check for Farkle
                if EvaluateResult(dice_vals) == 0:
                    self._total_reward = 0
                    print ("Greedy player Farkled with %d dice" % dice_vals.size)
                    return 0.0
                keep_rolling, kept_dice =  self.SelectAction(dice_vals)
                if keep_rolling == True:
                    dice = Dice(1)
                    dice_vals = dice.Roll()
                     # Check for Farkle
                    if EvaluateResult(dice_vals) == 0:
                        self._total_reward = 0
                        print ("Greedy player Farkled with %d die" % dice_vals.size)
                        return 0.0
                    else:
                        keep_rolling, kept_dice =  self.SelectAction(dice_vals)
                        
            elif kept_dice.size == 2:
                dice = Dice(1)
                dice_vals = dice.Roll()
                # Check for Farkle
                if EvaluateResult(dice_vals) == 0:
                    self._total_reward = 0
                    print ("Greedy player Farkled with %d die" % dice_vals.size)
                    return 0.0
                keep_rolling, kept_dice = self.SelectAction(dice_vals)
                
        #print ("GreedyPlayer turn scored %d points" % self._earnings_in_turn)
        self._total_reward = self._earnings_in_turn
        return self._total_reward 

# Always stop after one roll
class ConservativePlayer(Player):
    def __init__(self, debug = False):
        Player.__init__(self, debug)
    
    # Keep all dice and stop, return False meaning to stop rolling
    # Updates _earnings_in_turn as by product
    def SelectAction(self,dice_vals):
        keep_dice = [True, True, True]
        if self._debug:
            print ("kept_dice", dice_vals)
        self._earnings_in_turn += EvaluateResult(dice_vals)
        keep_rolling = False
        return keep_rolling, dice_vals
    
    # Do a turn and return the score accumulated    
    def DoTurn(self):
        self._earnings_in_turn = 0
        
        dice = Dice(3)
        dice_vals = dice.Roll()
        # Check for Farkle
        if EvaluateResult(dice_vals) == 0:
            self._total_reward = 0
            print ("Conservative player Farkled with %d dice" % dice_vals.size)
            return 0.0
        
        keep_rolling, dice_to_keep =  self.SelectAction(dice_vals)
        self._total_reward = self._earnings_in_turn
        
        #print ("ConversativePlayer turn scored %d points" % self._earnings_in_turn)
        return self._total_reward

# Start of main program        
debug = False     
        
consvy = ConservativePlayer(debug)
greedy = GreedyPlayer(debug)
smarty = OptimalPlayer(debug)

consvy_tp = 0.0
greedy_tp = 0.0
smarty_tp = 0.0

# Play a few turns
N_turns = 10
for i in range(N_turns):
    if True:
        print("Turn %d" % (i + 1))

    consvy_score = consvy.DoTurn()
    consvy_tp += consvy_score

    if debug:
        print("Consvy Total reward is %d" % consvy._total_reward)
        print("******************")

    greedy_score = greedy.DoTurn()
    greedy_tp += greedy_score

    if debug:
        print ("Greedy Total reward is %d" % greedy._total_reward)
        print("******************")

    smarty_score = smarty.DoTurn()
    smarty_tp += smarty_score

    if debug:
        print ("Smarty Total reward is %d" % smarty._total_reward)
        print("******************")

print("")
print("Greedy player point's per roll: %f" % (1.0*greedy_tp/N_turns))
print("Optimal player point's per roll: %f" % (1.0*smarty_tp/N_turns))
print("Conservative player point's per roll: %f" % (1.0*consvy_tp/N_turns))

