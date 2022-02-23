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
    """
    :param array_a:
    :param array_b:
    :param tol:
    :return:
    """
    # 1. Check if arrays have the same dimensions
    # 2. Check if arrays indexes have differences of value within the tolerance
    # Return True if same size and all indexes are within tolerance of another
    return np.equal(array_a, array_b) and np.allclose(array_a, array_b, tol)  # [1]

def simple_minimizer(func_in, start, end, num=100):
    """
    :param func_in:
    :param start:
    :param end:
    :param num:
    :return:
    """
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
    """
    :param num_rolls:
    :param iterations:
    :return:
    """
    # Playing one game of die rolling num_rolls number of times and finding the sum of the rolls
    games = np.random.randint(0, num_rolls, size=(iterations, num_rolls))  # [3]
    # Sum each game in its respective sub-array and represent as a number in the main array
    scores = np.sum(games, axis=1)
    # Histogram plot
    n, bins, patches = plt.hist(scores, 50, facecolor='blue', alpha=0.5)
    #Saving plot
    plt.savefig("dice_{}_rolls_{}.png".format(num_rolls, iterations))
    plt.show()
    return scores


def is_transformation_matrix(trans_matrix):
    """
    :param trans_matrix:
    :return:
    """
    rot_matrix = np.delete(trans_matrix, 3, 0)  # Get rid of the last row
    rot_matrix = np.delete(rot_matrix, 3, 1)  # Get rid of the last column

    # Now we have the rotation matrix (3x3)
    trans = rot_matrix.transpose()  # Calculate the transpose of the rotation matrix
    inv = np.linalg.inv(rot_matrix)  # Calculate tbe inverse of the rotation matrix

    # If they are identical then the rotation matrix is valid
    valid = np.equal(trans, inv)  # [1]
    if valid is True:
        return True
    else:
        return False


def nearest_neighbors(points, target, cutoff_dist):
    dist = np.array(np.linalg.norm(points-target, ord=2, axis=1))
    print(dist)
    dist = dist[dist < cutoff_dist]
    # dist = np.reshape(dist, (points.shape[0], 1))
    indexes = np.argsort(dist)
    return points[indexes]



if __name__ == '__main__':
    # a = np.arange(15).reshape(3, 5)
    # b = np.arange(15).reshape(3, 5)
    #
    # print('numpy_close: ', numpy_close(a, b))
    # my_func = lambda x: x**2
    # print('simple_minimizer: ', simple_minimizer(my_func, -1.75, 2.25, num=5))

    # num_rolls = 5
    # iterations = 2000
    # print(np.zeros(num_rolls))
    # print(r.randint(0, num_rolls))
    # a = np.random.randint(0, num_rolls, num_rolls)  # [3]
    # print(np.sum(a))

    # print(np.array(np.sum(a), iterations))
    # simulate_dice_rolls(num_rolls, iterations)

    # tf_valid = np.array([[0, 0, -1, 4], [0, 1, 0, 2.4], [1, 0, 0, 3], [0, 0, 0, 1]])
    # tf_invalid = np.array([[1, 2, 3, 1], [0, 1, -3, 4], [0, 1, 1, 1], [-0.5, 4, 0, 2]])
    # print(is_transformation_matrix(tf_valid))  # True
    # print(is_transformation_matrix(tf_invalid))  # False

    array = np.array([[1, 1, 1], [2, 3, 5], [0, 1, 1], [1.5, 1, 1], [10, 9, 9]])
    target_pt = np.array([0, 1, 1])
    print(nearest_neighbors(array, target_pt, 3.0))
