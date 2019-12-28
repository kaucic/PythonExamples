# This is a simulation of multiple strategies in a game of N = Three Dice Farkle.
#
# The game is played with N dice.
#
# Players play in turns. A turn consists of a sequence of one or more rolls of dice. A turn
# begins with a roll of all dice.  Outcomes:
#
# * An award of m1 is temporarily awarded if the two coin flips result in a single heads.
# * An award of m2 is temporarily awarded if the two coin flips result in two heads.
# 
#  A "Farkle" occurs if no dice score points. In this case, the player's turn is
#  over, and the player forfeits all the temporary rewards accumulated during his or her
#  turn.
#
# If a "Farkle" did not happen after the initial roll, the player may
# choose to stop and collect his or her reward, which then becomes his final reward for
# that turn. Alternatively, if one or more dice resulted in points,
# the player may choose to roll again excluding the dice scoring points
# two outcomes are possible:
#

import numpy as np
# import pdb

# dice_vals are a vector of 1-6, keep_dice is a binary vector True or False
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
        
class OptimalPlayer(Player):
    def __init__(self):
        Player.__init__(self)
        
    def Simulate(self):
        return False
    
    def DoTurn(self, max_iterations):
        self._earnings_in_turn = 0

# Keep rolliing until you use up all dice and then roll once more        
class GreedyPlayer(Player):
    def __init__(self, debug = False):
        Player.__init__(self, debug = False)
        
    # Keep rolliing until you use up all dice and then roll once more  
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
        keep_rolling = True
        
        dice = Dice(3)
        dice_vals = dice.Roll()
        # Check for Farkle
        if EvaluateResult(dice_vals) == 0:
            self._total_reward = 0
            keep_rolling = False
            print ("Greedy player Farkled with 3 dice")
            return 0.0        
        
        keep_rolling, kept_dice =  self.SelectAction(dice_vals)
        if keep_rolling == True:
            if kept_dice.size == 1:
                dice = Dice(2)
                dice_vals = dice.Roll()
                # Check for Farkle
                if EvaluateResult(dice_vals) == 0:
                    self._total_reward = 0
                    keep_rolling = False
                    print ("Greedy player Farkled with 2 dice")
                    return 0.0
                keep_rolling, kept_dice =  self.SelectAction(dice_vals)
                if keep_rolling == True:
                    dice = Dice(1)
                    dice_vals = dice.Roll()
                     # Check for Farkle
                    if EvaluateResult(dice_vals) == 0:
                        self._total_reward = 0
                        keep_rolling = False
                        print ("Greedy player Farkled with 1 die")
                        return 0.0
                    else:
                        keep_rolling, kept_dice =  self.SelectAction(dice_vals)
            elif kept_dice.size == 2:
                dice = Dice(1)
                dice_vals = dice.Roll()
                # Check for Farkle
                if EvaluateResult(dice_vals) == 0:
                    self._total_reward = 0
                    keep_rolling = False
                    print ("Greedy player Farkled with 1 die")
                    return 0.0
                keep_rolling, kept_dice = self.SelectAction(dice_vals)
                
        #print ("GreedyPlayer turn scored %d points" % self._earnings_in_turn)
        self._total_reward = self._earnings_in_turn
        return self._total_reward 

# Always stop after one roll
class ConservativePlayer(Player):
    def __init__(self, debug = False):
        Player.__init__(self, debug = False)
    
    # Keep all dice and stop, return False meaning to stop rolling
    def SelectAction(self,dice_vals):
        keep_dice = [True, True, True]
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
            keep_rolling = False
            print ("Conservative player Farkled with 3 dice")
            return 0.0
        
        keep_rolling, dice_to_keep =  self.SelectAction(dice_vals)
        self._total_reward = self._earnings_in_turn
        
        #print ("ConversativePlayer turn scored %d points" % self._earnings_in_turn)
        return self._total_reward

# Start of main program        
debug = False      
        
consvy = ConservativePlayer(debug)
greedy = GreedyPlayer(debug)
#smarty = OptimalPlayer()

consvy_tp = 0.0
greedy_tp = 0.0

# Play a few turns
N_turns = 1000
for i in range(N_turns):
    if debug:
        print("Turn %d" % (i + 1))
        
        print("Greedy player has started")
        
    greedy_score = greedy.DoTurn()
    greedy_tp += greedy_score

    if True:
        print ("Greedy Total reward is %d" % greedy._total_reward)
        print("******************")
    
    if debug:
        print("")

        print("Smarty player has started")
        
    #smarty_state = smarty.Turn(coin, max_iterations)

    if debug:
        print("")

        print("Conservative player has started")

    consvy_score = consvy.DoTurn()
    consvy_tp += consvy_score

    if True:
        print ("Consvy Total reward is %d" % consvy._total_reward)
        print("******************")


print("Greedy player point's per roll: %f" % (1.0*greedy_tp/N_turns))
#print("Optimal player point's per roll: %f" % (1.0*smarty._total_reward/N_turns))
print("Conservative player point's per roll: %f" % (1.0*consvy_tp/N_turns))

