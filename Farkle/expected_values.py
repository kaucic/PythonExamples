# Probabilites for Farkleing with X Dice
# 6 --  2.31%
# 5 --  7.72%
# 4 -- 15.74% -- 204/1296
# 3 -- 27.78% --  60/216
# 2 -- 44.44% --  16/36
# 1 -- 66.67% --   4/6


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

global p1, p2, p3, pnf3, R1, R2, R3

pnf3 = 156./216
p3 = pnf3
R3 = 83.56481
def approximate_E3_total(R):
    E = p3*R + R3
    return E
    
pnf1 = 1./3
p1 = 0.31739496
R1 = 35.24041445
cross1 = 56.31
def compute_E1_total(R):
    E = 1./6*max(R+50,approximate_E3_total(R+50)) + 1./6*max(R+100,approximate_E3_total(R+100))   
    return E

pnf2 = 5./9
p2 = 0.54878706
R2 = 52.91624684
cross2 = 117.58
def compute_E2_total(R):
    E = ( 2./9*max(R+50,compute_E1_total(R+50)) + 2./9*max(R+100,compute_E1_total(R+100)) +
    1./36*max(R+100,approximate_E3_total(R+100)) +  1./18*max(R+150,approximate_E3_total(R+150)) +
    1./36*max(R+200,approximate_E3_total(R+200)) )
    
    return E

pnf3 = 156./216
p3 = 0.71516241
R3 = 86.53020591 
cross3 = 303.84
def compute_E3_total(R):
    E = ( 2./9*max(R+50,compute_E2_total(R+50)) + 2./9*max(R+100,compute_E2_total(R+100)) +
    1./18* max([R+100,compute_E1_total(R+100),compute_E2_total(R+50)])  + 1./9*max([R+150,compute_E1_total(R+150),compute_E2_total(R+100)]) +
    1./18* max([R+200,compute_E1_total(R+200),compute_E2_total(R+100)]) + 1./216*max(R+200,approximate_E3_total(R+200)) +
    1./72* max([R+200,compute_E1_total(R+150),compute_E2_total(R+100),approximate_E3_total(R+200)]) +
    1./72* max([R+250,compute_E1_total(R+200),compute_E2_total(R+100),approximate_E3_total(R+250)]) +
    1./216*max([R+300,compute_E1_total(R+200),compute_E2_total(R+100),approximate_E3_total(R+300)]) +
    1./216*max(R+300,approximate_E3_total(R+300)) + 1./216*max(R+400,approximate_E3_total(R+400)) +
    1./216*max(R+500,approximate_E3_total(R+500)) + 1./216*max(R+600,approximate_E3_total(R+600)) )

    return E

# plot the expected value function for i dice
def plot_expected_value_function(i):
    rewards = list( range(0,601,50) )
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

# Start main program
if __name__ == "__main__":
    reward,E = plot_expected_value_function(1)
    reward,E = plot_expected_value_function(2)
    reward,E = plot_expected_value_function(3)

    