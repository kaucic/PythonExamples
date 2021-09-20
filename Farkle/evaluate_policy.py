# Probabilites for Farkleing with X Dice
# 6 --  2.31%
# 5 --  7.72%
# 4 -- 15.74% -- 204/1296
# 3 -- 27.78% --  60/216
# 2 -- 44.44% --  16/36
# 1 -- 66.67% --   4/6

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


global p1, p2, p3, R1, R2, R3

pnf1 = 1./3
p1 = 0.31739496
R1 = 35.24041445
cross1 = 56.31
def compute_E1_total(R):
    #E = 1./6*max(R+50,compute_E3_total(R+50)) + 1./6*max(R+100,compute_E3_total(R+100))
    E = p1*R + R1
    return E

pnf2 = 5./9
p2 = 0.54878706
R2 = 52.91624684
cross2 = 117.58
def compute_E2_total(R):
    #E = ( 2./9*max(R+50,compute_E1_total(R+50)) + 2./9*max(R+100,compute_E1_total(R+100)) +
    #1./36*max(R+100,compute_E3_total(R+100)) +  1./18*max(R+150,compute_E3_total(R+150)) +
    #1./36*max(R+200,compute_E3_total(R+200)) )
    E = p2*R + R2
    return E

pnf3 = 156./216
p3 = 0.71516241
R3 = 86.53020591
cross3 = 303.84
def compute_E3_total(R):
    #E = ( 2./9*max(R+50,compute_E2_total(R+50)) + 2./9*max(R+100,compute_E2_total(R+100)) +
    #1./18* max([R+100,compute_E1_total(R+100),compute_E2_total(R+50)])  + 1./9*max([R+150,compute_E1_total(R+150),compute_E2_total(R+100)]) +
    #1./18* max([R+200,compute_E1_total(R+200),compute_E2_total(R+100)]) + 1./216*max(R+200,approximate_E3_total(R+200)) +
    #1./72* max([R+200,compute_E1_total(R+150),compute_E2_total(R+100),approximate_E3_total(R+200)]) +
    #1./72* max([R+250,compute_E1_total(R+200),compute_E2_total(R+100),approximate_E3_total(R+250)]) +
    #1./216*max([R+300,compute_E1_total(R+200),compute_E2_total(R+100),approximate_E3_total(R+300)]) +
    #1./216*max(R+300,approximate_E3_total(R+300)) + 1./216*max(R+400,approximate_E3_total(R+400)) +
    #1./216*max(R+500,approximate_E3_total(R+500)) + 1./216*max(R+600,approximate_E3_total(R+600)) )

    E = p3*R + R3
    return E

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
    print (f"linear coefficients for {i} dice are {coeffs}")
    crossover = coeffs[0] / (1 - coeffs[1])
    print (f"crossover point is {crossover}")

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


# if three dice are available to throw and the reward is <= 300 or 
# after the first throw the reward <= 50
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
def navigate(i, probability, reward, sigma = (), histogram = None, policy = find_next_action_conservative):
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
histogram = navigate(3, 1, 0, policy =  find_next_action_manual_best)

rewards = np.sort([key for key in histogram.keys()])
print("Rewards: ", rewards)
print("Probability of Farkle: ", histogram[0])
E = 0
for reward in histogram:
    E += reward * histogram[reward]
print("Expected reward: ", E)

plot_reward_distribution(rewards,histogram)



