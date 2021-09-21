# Probabilites for Farkleing with X Dice
# 6 --  2.31% -- 1080/46656
# 5 --  7.72% --  600/7776 
# 4 -- 15.74% --  204/1296
# 3 -- 27.78% --   60/216
# 2 -- 44.44% --   16/36
# 1 -- 66.67% --    4/6

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

global p13, p23, p33, ap33, R13, R23, R33, aR33
global p14, p24, p34, p44, ap44, R14, R24, R34, R44, aR44

pnf3 = 156./216
ap33 = pnf3
aR33 = 83.56481
def approximate_E33_total(R):
    E = ap33*R + aR33
    return E
    
pnf1 = 1./3
p13 = 0.31739496
R13 = 35.24041445
cross13 = 56.31
def compute_E13_total(R):
    E = 1./6*max(R+50,approximate_E33_total(R+50)) + 1./6*max(R+100,approximate_E33_total(R+100))   
    return E

pnf2 = 5./9
p23 = 0.54878706
R23 = 52.91624684
cross23 = 117.58
def compute_E23_total(R):
    E = ( 2./9*max(R+50,compute_E13_total(R+50)) + 2./9*max(R+100,compute_E13_total(R+100)) +
    1./36*max(R+100,approximate_E33_total(R+100)) +  1./18*max(R+150,approximate_E33_total(R+150)) +
    1./36*max(R+200,approximate_E33_total(R+200)) )
    
    return E

pnf3 = 156./216
p33 = 0.71516241
R33 = 86.53020591 
cross33 = 303.84
def compute_E33_total(R):
    E = ( 2./9*max(R+50,compute_E23_total(R+50)) + 2./9*max(R+100,compute_E23_total(R+100)) +
    1./18* max([R+100,compute_E13_total(R+100),compute_E23_total(R+50)])  + 1./9*max([R+150,compute_E13_total(R+150),compute_E23_total(R+100)]) +
    1./18* max([R+200,compute_E13_total(R+200),compute_E23_total(R+100)]) + 1./216*max(R+200,approximate_E33_total(R+200)) +
    1./72* max([R+200,compute_E13_total(R+150),compute_E23_total(R+100),approximate_E33_total(R+200)]) +
    1./72* max([R+250,compute_E13_total(R+200),compute_E23_total(R+100),approximate_E33_total(R+250)]) +
    1./216*max([R+300,compute_E13_total(R+200),compute_E23_total(R+100),approximate_E33_total(R+300)]) +
    1./216*max(R+300,approximate_E33_total(R+300)) + 1./216*max(R+400,approximate_E33_total(R+400)) +
    1./216*max(R+500,approximate_E33_total(R+500)) + 1./216*max(R+600,approximate_E33_total(R+600)) )

    return E

pnf4 = 1092./1296
ap44 = pnf4
aR44 = 160
def approximate_E44_total(R):
    E = ap44*R + aR44
    return E
    
pnf1 = 1./3
p14 = 0.2808642
R14 = 74.39814815
cross14 = 103.45
def compute_E14_total(R):
    E = 1./6*max(R+50,approximate_E44_total(R+50)) + 1./6*max(R+100,approximate_E44_total(R+100))   
    return E

pnf2 = 5./9
p24 = 0.531766
R24 = 67.78633232  
cross24 = 144.77
def compute_E24_total(R):
    E = ( 2./9*max(R+50,compute_E14_total(R+50)) + 2./9*max(R+100,compute_E14_total(R+100)) +
    1./36*max(R+100,approximate_E44_total(R+100)) +  1./18*max(R+150,approximate_E44_total(R+150)) +
    1./36*max(R+200,approximate_E44_total(R+200)) )
    
    return E

pnf3 = 156./216
p34 = 0.82105823
R34 = 148.30947003
cross34 = 828.81
def compute_E34_total(R):
    E = ( 2./9*max(R+50,compute_E24_total(R+50)) + 2./9*max(R+100,compute_E24_total(R+100)) +
    1./18* max([R+100,compute_E14_total(R+100),compute_E24_total(R+50)])  + 1./9*max([R+150,compute_E14_total(R+150),compute_E24_total(R+100)]) +
    1./18* max([R+200,compute_E14_total(R+200),compute_E24_total(R+100)]) + 1./216*max(R+200,approximate_E44_total(R+200)) +
    1./72* max([R+200,compute_E14_total(R+150),compute_E24_total(R+100),approximate_E44_total(R+200)]) +
    1./72* max([R+250,compute_E14_total(R+200),compute_E24_total(R+100),approximate_E44_total(R+250)]) +
    1./216*max([R+300,compute_E14_total(R+200),compute_E24_total(R+100),approximate_E44_total(R+300)]) +
    1./216*max(R+300,approximate_E44_total(R+300)) + 1./216*max(R+400,approximate_E44_total(R+400)) +
    1./216*max(R+500,approximate_E44_total(R+500)) + 1./216*max(R+600,approximate_E44_total(R+600)) )

    return E

pnf4 = 1092./1296
p44 = 0.83617244
R44 = 141.60817633
cross44 = 864.37
def compute_E44_total(R):
    E = ( 240./1296*max(R+50,compute_E34_total(R+50)) + 240./1296*max(R+100,compute_E34_total(R+100)) +
    96./1296* max([R+100,compute_E24_total(R+100),compute_E34_total(R+50)]) +
    192./1296*max([R+150,compute_E24_total(R+150),compute_E34_total(R+100)]) +
    96./1296*max([R+200,compute_E24_total(R+200),compute_E34_total(R+100)]) +
    48./1296* max([R+200,compute_E14_total(R+200),compute_E24_total(R+150),compute_E34_total(R+100)]) +
    12./1296* max(R+200,compute_E14_total(R+200)) +
    48./1296* max([R+250,compute_E14_total(R+250),compute_E24_total(R+200),compute_E34_total(R+100)]) +
    4./1296*  max([R+250,compute_E14_total(R+200),compute_E34_total(R+50),approximate_E44_total(R+250)]) +
    16./1296* max([R+300,compute_E14_total(R+300),compute_E24_total(R+200),compute_E34_total(R+100)]) +
    12./1296* max(R+300,compute_E14_total(R+300)) +
    6./1296*  max([R+300,compute_E14_total(R+250),compute_E24_total(R+200),compute_E34_total(R+100),approximate_E44_total(R+300)]) +
    4./1296*  max([R+300,compute_E14_total(R+200),compute_E34_total(R+100),approximate_E44_total(R+300)]) +
    4./1296*  max([R+350,compute_E14_total(R+300),compute_E24_total(R+200),compute_E34_total(R+100),approximate_E44_total(R+350)]) +
    4./1296*  max([R+350,compute_E14_total(R+300),compute_E34_total(R+50),approximate_E44_total(R+350)]) +
    12./1296* max(R+400,compute_E14_total(R+400)) +
    4./1296*  max([R+400,compute_E14_total(R+300),compute_E34_total(R+100),approximate_E44_total(R+400)]) +
    4./1296*  max([R+450,compute_E14_total(R+400),compute_E34_total(R+50),approximate_E44_total(R+450)]) +
    4./1296*  max([R+500,compute_E14_total(R+400),compute_E34_total(R+100),approximate_E44_total(R+500)]) +
    16./1296* max(R+500,compute_E14_total(R+500)) +
    12./1296* max(R+600,compute_E14_total(R+600)) +
    4./1296*  max([R+600,compute_E14_total(R+500),compute_E34_total(R+100),approximate_E44_total(R+600)]) +
    4./1296*  max([R+650,compute_E14_total(R+600),compute_E34_total(R+50),approximate_E44_total(R+650)]) +
    4./1296*  max([R+700,compute_E14_total(R+600),compute_E34_total(R+100),approximate_E44_total(R+700)]) +
    6./1296*  max(R+1000,approximate_E44_total(R+1000)) )

    return E

# plot the expected value function for i dice in an N dice Farkle game
def plot_expected_value_function(i,N,rewards):
    if N == 3:
        if i == 1:
            E = [compute_E13_total(r) for r in rewards]
                
        elif i == 2:
            E = [compute_E23_total(r) for r in rewards]
        
        else:
            E = [compute_E33_total(r) for r in rewards]

    elif N == 4:
        if i == 1:
            E = [compute_E14_total(r) for r in rewards]
        
        elif i == 2:
            E = [compute_E24_total(r) for r in rewards]
        
        elif i == 2:
            E = [compute_E34_total(r) for r in rewards]
            
        else:
            E = [compute_E44_total(r) for r in rewards]

    else:
        print (f"{N} dice Farkle game not implemented")
                 
    # Use least squares to solve for reward functions linear coefficients
    m = len(rewards)
    y = np.array(E)
    A = np.c_[np.ones((m,1)), rewards]
    coeffs = np.dot(np.linalg.pinv(A), y)
    print (f"linear coefficients for {i} dice in {N} dice game are {coeffs}")
    crossover = coeffs[0] / (1 - coeffs[1])
    print (f"crossover point is {crossover}")
    
    text = "Expected Value function for %d dice in %d dice game" % (i, N)
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

def throw(i = None):
    if i == None or i == 4:
        result = (((4,  0), 204./1296), \
                  ((4,  1), 240./1296), \
                  ((4,  2), 240./1296), \
                  ((4,  3),  96./1296), \
                  ((4,  4), 192./1296), \
                  ((4,  5),  96./1296), \
                  ((4,  6),  48./1296), \
                  ((4,  7),  12./1296), \
                  ((4,  8),  48./1296), \
                  ((4,  9),   4./1296), \
                  ((4, 10),  16./1296), \
                  ((4, 11),  12./1296), \
                  ((4, 12),   6./1296), \
                  ((4, 13),   4./1296), \
                  ((4, 14),   4./1296), \
                  ((4, 15),   4./1296), \
                  ((4, 16),  12./1296), \
                  ((4, 17),   4./1296), \
                  ((4, 18),   4./1296), \
                  ((4, 19),   4./1296), \
                  ((4, 20),  16./1296), \
                  ((4, 21),  12./1296), \
                  ((4, 22),   4./1296), \
                  ((4, 23),   4./1296), \
                  ((4, 24),   4./1296), \
                  ((4, 25),   6./1296))
    if i == 3:
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

# Start main program
if __name__ == "__main__":
    if False:
        rewards = list( range(0,301,25) )    
        reward,E = plot_expected_value_function(1,3,rewards)

        rewards = list( range(0,401,50) ) 
        reward,E = plot_expected_value_function(2,3,rewards)

        rewards = list( range(0,601,50) ) 
        reward,E = plot_expected_value_function(3,3,rewards)
    else:
        rewards = list( range(50,401,25) ) 
        reward,E = plot_expected_value_function(1,4,rewards)

        rewards = list( range(50,401,50) ) 
        reward,E = plot_expected_value_function(2,4,rewards)

        rewards = list( range(50,601,50) ) 
        reward,E = plot_expected_value_function(3,4,rewards)

        rewards = list( range(0,1001,50) ) 
        reward,E = plot_expected_value_function(4,4,rewards)


    