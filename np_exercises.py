#! Users/tannerwilliams/Desktop/ME499/ME499_Lab6_NumPy_I/np_exercises.py
import numpy as np
import random as r
import matplotlib.pyplot as plt

""" References:
    [1] https://www.delftstack.com/howto/numpy/python-compare-arrays/
    [2] https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    [3] https://www.studytonight.com/post/creating-random-valuedarrays-in-numpy
    [4]
"""


def numpy_close(array_a, array_b, tol=1e-8):
    # 1. Check if arrays have the same dimensions
    # 2. Check if arrays indexes have differences of value within the tolerance
    # Return True if same size and all indexes are within tolerance of another
    return np.array_equal(array_a, array_b) and np.allclose(array_a, array_b, tol)  # [1]


def simple_minimizer(func_in, start, end, num=100):
    # make sure that prescribed start and end points move forward
    if start > end:
        raise ValueError
    else:
        # Note: make start and end floats if they aren't already
        steps = np.linspace(float(start), float(end), num)  # [2] Note: linspace auto-inclusive of start and end values
        # Make a new array of same size as steps with outputs using my_func applied to steps
        y_out = func_in(steps)
        # index of minimum output
        min_index = np.argmin(y_out)
        # print smallest output value and its corresponding input
        # form: (input, output)
        return steps[min_index], y_out[min_index]


def simulate_dice_rolls(num_rolls, iterations):
    # Playing one game of die rolling num_rolls number of times and finding the sum of the rolls
    games = np.random.randint(0, num_rolls, size=(iterations, num_rolls))  # [3]
    scores = np.sum(games, axis=1)
    n, bins, patches = plt.hist(scores, 50, facecolor='blue', alpha=0.5)
    plt.savefig("dice_{}_rolls_{}.png".format(num_rolls, iterations))
    plt.show()
    return scores


if __name__ == '__main__':
    a = np.arange(15).reshape(3, 5)
    b = np.arange(15).reshape(3, 5)

    print('numpy_close: ', numpy_close(a, b))
    my_func = lambda x: x**2
    print('simple_minimizer: ', simple_minimizer(my_func, -1.75, 2.25, num=5))

    num_rolls = 5
    iterations = 2000
    # print(np.zeros(num_rolls))
    # print(r.randint(0, num_rolls))
    # a = np.random.randint(0, num_rolls, num_rolls)  # [3]
    # print(np.sum(a))

    # print(np.array(np.sum(a), iterations))
    print(simulate_dice_rolls(num_rolls, iterations))
