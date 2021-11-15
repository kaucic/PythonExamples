import numpy as np
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
                  ((2, 4),  1./36), \
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
    
    else:
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
    
    else:
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
    
    else:
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

# One final throw if after the first throw the reward is 50,
# or if three dice are available to throw and the reward is not greater than 300
def find_next_action_conservative_plus(state, reward, sigma):
    if state[0] == 1:
        if state[1] == 1:
            return (k, 50)
        if state[1] == 2:
            return (k, 100)
        
    elif state[0] == 2:
        if state[1] == 1:
            return (0, 50)
        if state[1] == 2:
            return (0, 100)
        if state[1] == 3:
            return (0, 100)
        if state[1] == 4:
            return (0, 150)
        if state[1] == 5:
            return (0, 200)
    
    else: # state[0] == 3:
        if state[1] == 1:
            if reward > 0:
                return (0, 50)
            else:
                return (2, 50)
        elif state[1] == 2:
            return (0, 100)
        elif state[1] == 3:
            return (0, 100)
        elif state[1] == 4:
            return (0, 150)
        elif state[1] == 5:
            return (0, 200)
        elif state[1] == 6:
            if reward > 0:
                return (0, 200)
            else:
                return (3, 200)
        elif state[1] == 7:
            if reward > 0:
                return (0, 200)
            else:
                return (3, 200)
        elif state[1] == 8:
            if reward > 0:
                return (0, 250)
            else:
                return (3, 250)
        elif state[1] == 9:
            if reward > 0:
                return (0, 300)
            else:
                return (3, 300)
        elif state[1] == 10:
            if reward > 0:
                return (0, 300)
            else:
                return (3, 300)
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

histogram = navigate(None, 1, 0, policy = find_next_action_conservative_plus)

rewards = np.sort([key for key in histogram.keys()])
print("Rewards: ", rewards)
#probabilities = [histogram[key] for key in rewards]
#print("Probabilities: ", probabilities)
print("Probability of Farkle: ", histogram[0])
E = 0
for reward in histogram:
    E += reward * histogram[reward]
print("Expected reward: ", E)

text = "Log of Reward Distribution for Policy"
plt.figure(text, figsize=(13, 8))
plt.bar(histogram.keys(), histogram.values(), width = 40)
plt.yscale("log")
plt.title(text)
plt.xlabel("Reward")
plt.ylabel("log(P(Reward))")
plt.grid("on")
plt.show()
