# Probabilites for Farkleing with X Dice
# 6 --  2.31% -- 1080/46656
# 5 --  7.72% --  600/7776
# 4 -- 15.74% --  204/1296
# 3 -- 27.78% --   60/216
# 2 -- 44.44% --   16/36
# 1 -- 66.67% --    4/6

from constants import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optim


def approximate_E33_total(R):
    pnf3 = 156. / 216
    ap33 = pnf3
    aR33 = 83.56481

    E = ap33 * R + aR33
    return E


def compute_E13_total(R):
    pnf1 = 1. / 3
    p13 = 0.25753799
    R13 = 44.53050607
    cross13 = 59.97

    E = 1. / 6 * max(R + 50, approximate_E33_total(R + 50)) + \
        1. / 6 * max(R + 100, approximate_E33_total(R + 100))
    return E


def compute_E23_total(R):
    pnf2 = 5. / 9
    p23 = 0.54270048
    R23 = 53.85035555
    cross23 = 117.58

    E = (8. / 36 * max(R + 50, compute_E13_total(R + 50)) + 8. / 36 * max(R + 100, compute_E13_total(R + 100)) +
         1. / 36 * max([R + 100, compute_E14_total(R + 50), approximate_E33_total(R + 100)]) +
         2. / 36 * max([R + 150, compute_E14_total(R + 100), approximate_E33_total(R + 150)]) +
         1. / 36 * max([R + 200, compute_E14_total(R + 100), approximate_E33_total(R + 200)]))
    return E


def compute_E33_total(R):
    pnf3 = 156. / 216
    p33 = 0.72222222
    R33 = 83.56481481
    cross33 = 300.83

    E = (48. / 216 * max(R + 50, compute_E23_total(R + 50)) + 48. / 216 * max(R + 100, compute_E23_total(R + 100)) +
         12. / 216 * max([R + 100, compute_E13_total(R + 100), compute_E23_total(R + 50)]) +
         24. / 216 * max([R + 150, compute_E13_total(R + 150), compute_E23_total(R + 100)]) +
         12. / 216 * max([R + 200, compute_E13_total(R + 200), compute_E23_total(R + 100)]) +
         1. / 216 * max(R + 200, approximate_E33_total(R + 200)) +
         3. / 216 * max([R + 200, compute_E13_total(R + 150), compute_E23_total(R + 100), approximate_E33_total(R + 200)]) +
         3. / 216 * max([R + 250, compute_E13_total(R + 200), compute_E23_total(R + 100), approximate_E33_total(R + 250)]) +
         1. / 216 * max([R + 300, compute_E13_total(R + 200), compute_E23_total(R + 100), approximate_E33_total(R + 300)]) +
         1. / 216 * max(R + 300, approximate_E33_total(R + 300)) + 1. / 216 * max(R + 400, approximate_E33_total(R + 400)) +
         1. / 216 * max(R + 500, approximate_E33_total(R + 500)) + 1. / 216 * max(R + 600, approximate_E33_total(R + 600)))

    return E


def approximate_E44_total(R):
    pnf4 = 1092. / 1296
    ap44 = pnf4
    aR44 = 132.716

    E = ap44 * R + aR44
    return E


def compute_E14_total(R):
    pnf1 = 1. / 3
    p14 = 0.2808642
    R14 = 65.30348148
    cross14 = 90.80

    E = 1. / 6 * max(R + 50, approximate_E44_total(R + 50)) + \
        1. / 6 * max(R + 100, approximate_E44_total(R + 100))
    return E


def compute_E24_total(R):
    pnf2 = 5. / 9
    p24 = 0.53806584
    R24 = 62.12276543
    cross24 = 134.48

    E = (8. / 36 * max(R + 50, compute_E14_total(R + 50)) + 8. / 36 * max(R + 100, compute_E14_total(R + 100)) +
         1. / 36 * max([R + 100, compute_E14_total(R + 50), approximate_E44_total(R + 100)]) +
         2. / 36 * max([R + 150, compute_E14_total(R + 100), approximate_E44_total(R + 150)]) +
         1. / 36 * max([R + 200, compute_E14_total(R + 100), approximate_E44_total(R + 200)]))

    return E


def compute_E34_total(R):
    pnf3 = 156. / 216
    p34 = 0.82854233
    R34 = 142.9135599
    cross34 = 833.52

    E = (48. / 216 * max(R + 50, compute_E24_total(R + 50)) + 48. / 216 * max(R + 100, compute_E24_total(R + 100)) +
         12. / 216 * max([R + 100, compute_E14_total(R + 100), compute_E24_total(R + 50)]) +
         24. / 216 * max([R + 150, compute_E14_total(R + 150), compute_E24_total(R + 100)]) +
         12. / 216 * max([R + 200, compute_E14_total(R + 200), compute_E24_total(R + 100)]) +
         1. / 216 * max(R + 200, approximate_E44_total(R + 200)) +
         3. / 216 * max([R + 200, compute_E14_total(R + 150), compute_E24_total(R + 100), approximate_E44_total(R + 200)]) +
         3. / 216 * max([R + 250, compute_E14_total(R + 200), compute_E24_total(R + 100), approximate_E44_total(R + 250)]) +
         1. / 216 * max([R + 300, compute_E14_total(R + 200), compute_E24_total(R + 100), approximate_E44_total(R + 300)]) +
         1. / 216 * max(R + 300, approximate_E44_total(R + 300)) + 1. / 216 * max(R + 400, approximate_E44_total(R + 400)) +
         1. / 216 * max(R + 500, approximate_E44_total(R + 500)) + 1. / 216 * max(R + 600, approximate_E44_total(R + 600)))

    return E


def compute_E44_total(R):
    pnf4 = 1092. / 1296
    p44 = 0.83679228
    R44 = 137.07507444
    cross44 = 839.88

    E = (240. / 1296 * max(R + 50, compute_E34_total(R + 50)) + 240. / 1296 * max(R + 100, compute_E34_total(R + 100)) +
         96. / 1296 * max([R + 100, compute_E24_total(R + 100), compute_E34_total(R + 50)]) +
         192. / 1296 * max([R + 150, compute_E24_total(R + 150), compute_E34_total(R + 100)]) +
         96. / 1296 * max([R + 200, compute_E24_total(R + 200), compute_E34_total(R + 100)]) +
         48. / 1296 * max([R + 200, compute_E14_total(R + 200), compute_E24_total(R + 150), compute_E34_total(R + 100)]) +
         12. / 1296 * max(R + 200, compute_E14_total(R + 200)) +
         48. / 1296 * max([R + 250, compute_E14_total(R + 250), compute_E24_total(R + 200), compute_E34_total(R + 100)]) +
         4. / 1296 * max([R + 250, compute_E14_total(R + 200), compute_E34_total(R + 50), approximate_E44_total(R + 250)]) +
         16. / 1296 * max([R + 300, compute_E14_total(R + 300), compute_E24_total(R + 200), compute_E34_total(R + 100)]) +
         12. / 1296 * max(R + 300, compute_E14_total(R + 300)) +
         6. / 1296 * max([R + 300, compute_E14_total(R + 250), compute_E24_total(R + 200), compute_E34_total(R + 100), approximate_E44_total(R + 300)]) +
         4. / 1296 * max([R + 300, compute_E14_total(R + 200), compute_E34_total(R + 100), approximate_E44_total(R + 300)]) +
         4. / 1296 * max([R + 350, compute_E14_total(R + 300), compute_E24_total(R + 200), compute_E34_total(R + 100), approximate_E44_total(R + 350)]) +
         4. / 1296 * max([R + 350, compute_E14_total(R + 300), compute_E34_total(R + 50), approximate_E44_total(R + 350)]) +
         12. / 1296 * max(R + 400, compute_E14_total(R + 400)) +
         4. / 1296 * max([R + 400, compute_E14_total(R + 300), compute_E34_total(R + 100), approximate_E44_total(R + 400)]) +
         4. / 1296 * max([R + 450, compute_E14_total(R + 400), compute_E34_total(R + 50), approximate_E44_total(R + 450)]) +
         4. / 1296 * max([R + 500, compute_E14_total(R + 400), compute_E34_total(R + 100), approximate_E44_total(R + 500)]) +
         16. / 1296 * max(R + 500, compute_E14_total(R + 500)) +
         12. / 1296 * max(R + 600, compute_E14_total(R + 600)) +
         4. / 1296 * max([R + 600, compute_E14_total(R + 500), compute_E34_total(R + 100), approximate_E44_total(R + 600)]) +
         4. / 1296 * max([R + 650, compute_E14_total(R + 600), compute_E34_total(R + 50), approximate_E44_total(R + 650)]) +
         4. / 1296 * max([R + 700, compute_E14_total(R + 600), compute_E34_total(R + 100), approximate_E44_total(R + 700)]) +
         6. / 1296 * max(R + 1000, approximate_E44_total(R + 1000)))

    return E

def compute_E13_total_recursive(
    R,
    counter=0):
    """Compute the expected reward of throwing one die in a three-dice Farkle
    game starting from a reward of R.

    Args:
        R ([type]): 
        counter (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if R > CONST.CUTOFF_E13 or counter > CONST.MAX_RECURSION_DEPTH:
        return -1

    E = 1 / 6 * max(
            R + 50,
            compute_E33_total_recursive(R + 50, counter + 1)
        ) + \
        1 / 6 * max(
            R + 100,
            compute_E33_total_recursive(R + 100, counter + 1))

    return E


def compute_E23_total_recursive(R, counter=0):
    if R > CONST.CUTOFF_E23 or counter > CONST.MAX_RECURSION_DEPTH:
        return -1

    E = 8 / 36 * max(
            R + 50,
            compute_E13_total_recursive(R + 50, counter + 1)
        ) + \
        8 / 36 * max(
            R + 100,
            compute_E13_total_recursive(R + 100, counter + 1)
        ) + \
        1 / 36 * max(
            R + 100,
            compute_E33_total_recursive(R + 100, counter + 1)
        ) + \
        2 / 36 * max(
            R + 150,
            compute_E33_total_recursive(R + 150, counter + 1)
        ) + \
        1 / 36 * max(
            R + 200,
            compute_E33_total_recursive(R + 200, counter + 1))

    return E


def compute_E33_total_recursive(R, counter=0):
    if R > CONST.CUTOFF_E33 or counter > CONST.MAX_RECURSION_DEPTH:
        return -1

    E = 48 / 216 * max(
            R + 50,
            compute_E23_total_recursive(R + 50, counter + 1)) + \
        48 / 216 * max(
            R + 100,
            compute_E23_total_recursive(R + 100, counter + 1)) + \
        12 / 216 * max(
            R + 100,
            compute_E13_total_recursive(R + 100, counter + 1),
            compute_E23_total_recursive(R + 50, counter + 1)) + \
        24 / 216 * max(
            R + 150,
            compute_E13_total_recursive(R + 150, counter + 1),
            compute_E23_total_recursive(R + 100, counter + 1)) + \
        12 / 216 * max(
            R + 200,
            compute_E13_total_recursive(R + 200, counter + 1),
            compute_E23_total_recursive(R + 100, counter + 1)) + \
        1 / 216 * max(
            R + 200,
            compute_E33_total_recursive(R + 200, counter + 1)) + \
        3 / 216 * max(
            R + 200,
            compute_E33_total_recursive(R + 200, counter + 1)) + \
        3 / 216 * max(
            R + 250,
            compute_E33_total_recursive(R + 250, counter + 1)) + \
        1 / 216 * max(
            R + 300,
            compute_E33_total_recursive(R + 300, counter + 1)) + \
        1 / 216 * max(
            R + 300,
            compute_E33_total_recursive(R + 300, counter + 1)) + \
        1 / 216 * max(
            R + 400,
            compute_E33_total_recursive(R + 400, counter + 1)) + \
        1 / 216 * max(
            R + 500,
            compute_E33_total_recursive(R + 500, counter + 1)) + \
        1 / 216 * max(
            R + 600,
            compute_E33_total_recursive(R + 600, counter + 1))

    return E

# plot the expected value function for i dice in an N dice Farkle game


def plot_expected_value_function(i, N, method, rewards):
    """Plot stuff

    Args:
        i (int): ?
        N (int): ?
        method (string): method used to compute expected values.
        rewards (array-like): range of rewards to plot.
    """
    if N == 3:
        if method[:6] == "approx":
            if i == 1:
                E = [compute_E13_total(r) for r in rewards]

            elif i == 2:
                E = [compute_E23_total(r) for r in rewards]

            else:
                E = [compute_E33_total(r) for r in rewards]
        if method[:6] == "recurs":
            if i == 1:
                E = [compute_E13_total_recursive(r) for r in rewards]

            elif i == 2:
                E = [compute_E23_total_recursive(r) for r in rewards]

            else:
                E = [compute_E33_total_recursive(r) for r in rewards]

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
        print(f"{N} dice Farkle game not implemented")

    # Use least squares to solve for reward functions linear coefficients
    m = len(rewards)
    y = np.array(E)
    A = np.c_[np.ones((m, 1)), rewards]
    coeffs = np.dot(np.linalg.pinv(A), y)
    print(f"linear coefficients for {i} dice in {N} dice game are {coeffs}")
    crossover = coeffs[0] / (1 - coeffs[1])
    print(f"crossover point is {crossover}")

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

    return rewards, E


# Start main program
if __name__ == "__main__":
    if True:
        for method in ["approx", "recurse"]:
            rewards = list(range(0, 301, 25))
            reward, E = plot_expected_value_function(1, 3, method, rewards)

            rewards = list(range(0, 401, 50))
            reward, E = plot_expected_value_function(2, 3, method, rewards)

            rewards = list(range(150, 601, 50))
            reward, E = plot_expected_value_function(3, 3, method, rewards)

    else:
        rewards = list(range(0, 401, 25))
        reward, E = plot_expected_value_function(1, 4, method, rewards)

        rewards = list(range(0, 401, 50))
        reward, E = plot_expected_value_function(2, 4, method, rewards)

        rewards = list(range(50, 1001, 50))
        reward, E = plot_expected_value_function(3, 4, method, rewards)

        rewards = list(range(200, 1001, 50))
        reward, E = plot_expected_value_function(4, 4, method, rewards)
