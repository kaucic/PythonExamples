# Probabilites for Farkleing with X Dice
# 6 --  2.31%
# 5 --  7.72%
# 4 -- 15.74%
# 3 -- 27.78%
# 2 -- 44.44%
# 1 -- 66.67%

import numpy as np

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import matplotlib.pyplot as plt

def throw(i = None):
    if i == None or i == 3:
        result = (((3,  0), 60./216), \
                  ((3,  1), 48./216), \
                  ((3,  2), 48./216), \
                  ((3,  3), 12./216), \
                  ((3,  4), 24./216), \
                  ((3,  5), 12./216), \
                  ((3,  6),  1./216), \
                  ((3,  7),  3./216), \
                  ((3,  8),  3./216), \
                  ((3,  9),  1./216), \
                  ((3, 10),  1./216), \
                  ((3, 11),  1./216), \
                  ((3, 12),  1./216), \
                  ((3, 13),  1./216))
    if i == 2:
        result = (((2, 0), 16./36), \
                  ((2, 1),  8./36), \
                  ((2, 2),  8./36), \
                  ((2, 3),  1./36), \
                  ((2, 4),  2./36), \
                  ((2, 5),  1./36))
    if i == 1:
        result = (((1, 0), 4./6), \
                  ((1, 1), 1./6), \
                  ((1, 2), 1./6))

    return result
    
# Min greedy policy
def find_next_action_min_greedy(state, reward, sigma):
    if state[0] == 1:
        return (0, 50 * state[1])
    
    elif state[0] == 2:
        if state[1] < 3:
            return (1, 50 * state[1])
        else:
            return (0, 50 * (state[1] - 1))
    
    else:  # state[0] == 3 dice
        if state[1] < 3:
            return (2, 50 * state[1])
        elif state[1] < 6:
            return (1, 50 * (state[1] - 1))
        elif state[1] < 8:
            return (0, 200)
        elif state[1] < 10:
            return (0, 50 * (state[1] - 3))
        else:
            return (0, 100 * (state[1] - 7))
        
# Min greedy policy
def find_next_action_max_greedy(state, reward, sigma):
    if state[0] == 1:
        return (0, 50 * state[1])
    
    elif state[0] == 2:
        if state[1] < 3:
            return (1, 50 * state[1])
        else:
            return (0, 50 * (state[1] - 1))
    
    else:  # state[0] == 3 dice
        if state[1] < 3:
            return (2, 50 * state[1])
        elif state[1] < 5:
            return (2, 50 * (state[1] - 2))
        elif state[1] == 5:
            return (2, 100)
        elif state[1] < 8:
            return (0, 200)
        elif state[1] < 10:
            return (0, 50 * (state[1] - 3))
        else:
            return (0, 100 * (state[1] - 7))

    
def find_next_action_conservative(state, reward, sigma):
    if state[0] == 1:
        return (0, 50 * state[1])
    
    elif state[0] == 2:
        if state[1] < 3:
            return (0, 50 * state[1])
        else:
            return (0, 50 * (state[1] - 1))
    
    else:  # state[0] == 3 dice
        if state[1] < 3:
            return (0, 50 * state[1])
        elif state[1] < 6:
            return (0, 50 * (state[1] - 1))
        elif state[1] < 8:
            return (0, 200)
        elif state[1] < 10:
            return (0, 50 * (state[1] - 3))
        else:
            return (0, 100 * (state[1] - 7))


global p1, p2, pnf3, R1, R2, R3

pnf1 = 1./3
p1 = 0.31739496
R1 = 35.24041445
def compute_E1_total(R):
    E = 1./6*max(R+50,compute_E3_total(R+50)) + 1./6*max(R+100,compute_E3_total(R+100))
    return E

pnf2 = 5./9
p2 = 0.55218238
R2 = 52.12393241
def compute_E2_total(R):
    E = ( 2./9*max(R+50,compute_E1_total(R+50)) + 2./9*max(R+100,compute_E1_total(R+100)) +
    1./36*max(R+100,compute_E3_total(R+100)) +  1./18*max(R+150,compute_E3_total(R+150)) +
    1./36*max(R+200,compute_E3_total(R+200)) )
    return E

pnf3 = 156./216
R3 = 83.56481

R3 = 92.72940957933241
pnf3 = 1 - 0.5006001371742113
R3 = 91.23085276634657
pnf3 = 1 - 0.477023319615912
R3 = 200.0
pnf3 = 0.1
#R3 = 92.7258373342478
#pnf3 = 1 - 0.49631344307270236
def compute_E3_total(R):
    return pnf3*R + R3


# plot the expected value function for i dice
def plot_expected_value_function(i):
    rewards = list( range(0,901,50) )
    if i == 1:
        E = [compute_E1_total(r) for r in rewards]
             
    elif i == 2:
        E = [compute_E2_total(r) for r in rewards]
    
    else:
        E = [compute_E3_total(r) for r in rewards]
        
    # Use least squares to solve for reward function linear coefficients
    m = len(rewards)
    y = np.array(E)
    A = np.c_[np.ones((m,1)), rewards]
    coeffs = np.dot(np.linalg.pinv(A), y)
    print (coeffs)
    
    text = "Expected Value function for %d dice" % i
    plt.figure(text, figsize=(13, 8))
    plt.yscale("linear")
    plt.title(text)
    plt.xlabel("$R$")
    plt.xticks(rewards)
    plt.ylabel("Expected Value")
    plt.plot(rewards, E)
    plt.plot(rewards, rewards)
    plt.show()

    return rewards,E 


# One final throw if after the first throw the reward is 50,
# or if three dice are available to throw and the reward is not greater than 300
def find_next_action_manual_best(state, reward, sigma):
    
    if state[0] == 1:
        if state[1] == 1:
            if reward + 50 <= compute_E3_total(reward + 50):
                return (3, 50)
            else:
                return (0, 50)
        if state[1] == 2:
            if reward + 100 <= compute_E3_total(reward + 100):
                return (3, 100)
            else:
                return (0, 100)
        
    elif state[0] == 2:
        if state[1] == 1:
            if reward + 50 <= compute_E1_total(reward + 50):
                return (1, 50)
            else:
                return (0, 50)

        if state[1] == 2:
            if reward + 100 <= compute_E1_total(reward + 100):
                return (1, 100)
            else:
                return (0, 100)

        if state[1] == 3:
            if reward + 100 <= compute_E3_total(reward + 100):
                return (3, 100)
            else:
                return (0, 100)

        if state[1] == 4:
            if reward + 150 <= compute_E3_total(reward + 150):
                return (3, 150)
            else:
                return (0, 150)

        if state[1] == 5:
            if reward + 200 <= compute_E3_total(reward + 200):
                return (3, 200)
            else:
                return (0, 200)
    
    else: # state[0] == 3 dice
        if state[1] == 1:
            if reward + 50 <= compute_E2_total(reward + 50):
                return (2, 50)
            else:
                return (0, 50)

        elif state[1] == 2:
            if reward + 100 <= compute_E2_total(reward + 100):
                return (2, 100)
            else:
                return (0, 100)

        elif state[1] == 3:
            E2 = compute_E2_total(reward + 50)
            E1 = compute_E1_total(reward + 100)
            if E2 > E1:
                if reward + 100 <= E2:
                    return (2, 50)
                else:
                    return (0, 100)
            else:
                if reward + 100 <= E1:
                    return (1, 100)
                else:
                    return(0, 100)

        elif state[1] == 4:
            return (0, 150)

        elif state[1] == 5:
            return (0, 200)

        elif state[1] == 6:
            if reward + 200 <= compute_E3_total(reward + 200):
                return (3, 200)
            else:
                return (0, 200)

        elif state[1] == 7:
            if reward + 200 <= compute_E3_total(reward + 200):
                return (3, 200)
            else:
                return (0, 200)    

        elif state[1] == 8:
            if reward + 250 <= compute_E3_total(reward + 250):
                return (3, 250)
            else:
                return (0, 250)

        elif state[1] == 9:
            if reward + 300 <= compute_E3_total(reward + 300):
                return (3, 300)
            else:
                return (0, 300)    

        elif state[1] == 10:
            if reward + 300 <= compute_E3_total(reward + 300):
                return (3, 300)
            else:
                return (0, 300)

        elif state[1] == 11:
            return (0, 400)

        elif state[1] == 12:
            return (0, 500)

        else: # state[1] == 13
            return (0, 600)
    
# Navigate outcomes
def navigate(i, probability, reward, sigma = (), histogram = None, policy = find_next_action_min_greedy):
    if histogram == None:
        histogram = dict()
    outcomes = throw(i)
    for outcome in outcomes:
        new_state, p = outcome
        new_probability = probability * p
        if new_state[1] == 0:
            # Farkle
            if 0 in histogram.keys():
                histogram[0] += new_probability
            else:
                histogram[0] = new_probability
        else:
            k, R = policy(new_state, reward, sigma)
            new_reward = reward + R
            if k == 0:
                # Stop
                if new_reward in histogram.keys():
                    histogram[new_reward] += new_probability
                else:
                    histogram[new_reward] = new_probability
            else:
                new_sigma = sigma + (i,)
                histogram = navigate(k, new_probability, new_reward, new_sigma, histogram, policy)
    
    return histogram

def plot_reward_distribution(rewards,histogram):
    text = "Log of Reward Distribution for ""Conservative Plus"" Policy"
    plt.figure(text, figsize=(13, 8))
    plt.bar(histogram.keys(), histogram.values(), width = 40, label = "$log(P)$")
    plt.yscale("log")
    plt.title(text)
    plt.xlabel("$R$")
    plt.xticks(rewards)
    plt.ylim((1e-5, 1))
    plt.ylabel("$\log(P(R))$")
    plt.grid("on")
    plt.plot(E, plt.ylim()[0], "r*", clip_on = False, label = "$E[R] = %2.2f$" % E)
    plt.legend()
    plt.savefig(text + (".%s" % "pdf"), bbox_inches='tight')
    plt.show()

# Start of main program
histogram = navigate(3, 1, 0, policy = find_next_action_conservative)

rewards = np.sort([key for key in histogram.keys()])
print("Rewards: ", rewards)
print("Probability of Farkle: ", histogram[0])
E = 0
for reward in histogram:
    E += reward * histogram[reward]
print("Expected reward: ", E)

plot_reward_distribution(rewards,histogram)



