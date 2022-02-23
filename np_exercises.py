#! Users/tannerwilliams/Desktop/ME499/ME499_Lab6_NumPy_I/np_exercises.py
import numpy as np
import random
import matplotlib.pyplot as plt

""" References:
    [1] https://www.delftstack.com/howto/numpy/python-compare-arrays/
    [2] https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    [3] https://www.studytonight.com/post/creating-random-valuedarrays-in-numpy
"""


def numpy_close(array_a, array_b, tol=1e-8):
    """
    :param array_a: array of size mxn
    :param array_b: array of size mxn
    :param tol: tolerance for differences in array values
    :return: if array_a and array_b are within tolerance of another at every location
    """

    # 1. Check if arrays have the same dimensions
    # 2. Check if arrays indexes have differences of value within the tolerance
    # Return True if same size and all indexes are within tolerance of another
    return np.array_equal(array_a, array_b) and np.allclose(array_a, array_b, tol)  # [1]


def simple_minimizer(func_in, start, end, num=100):
    """
    :param func_in: function reference for 1-D array manipulated by a function defined by user
    :param start: where to start searching minimum in func_in
    :param end: where to end searching for minimum in func_in
    :param num: number of steps to take between the start and end parameters searching for minimum
    :return: the original value from 1-D array before func_in was applied and the corresponding minimum value from
             func_in
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
    :param num_rolls: number of rolls of a 6-sided die per iteration (game)
    :param iterations: number of times to roll num_rolls (how many games)
    :return: a) 1-D array of the sum of the dice rolls from each game (length=iterations)
             b) Histogram of results
    """
    # Playing one game of die rolling num_rolls number of times and finding the sum of the rolls
    games = np.random.randint(0, num_rolls, size=(iterations, num_rolls))  # [3]
    # Sum each game in its respective sub-array and represent as a number in the main array
    scores = np.sum(games, axis=1)
    # Histogram plot
    plt.hist(scores)
    # Saving plot
    plt.savefig("dice_{}_rolls_{}.png".format(num_rolls, iterations))
    plt.show()
    return scores


def is_transformation_matrix(trans_matrix):
    """
    :param trans_matrix: transformation matrix of size 4x4
    :return: if the rotation matrix within transformation matrix is true or not
    """
    # print('trans_matrix')
    # print(trans_matrix)
    rot_matrix = np.delete(trans_matrix, 3, 0)  # Get rid of the last row
    rot_matrix = np.delete(rot_matrix, 3, 1)  # Get rid of the last column
    # print('rotation matrix')
    # print(rot_matrix)

    # Now we have the rotation matrix (I = R * R^T)
    transpose = rot_matrix.transpose()  # Calculate the transpose of rotation matrix (R^T)
    # print('transpose')
    # print(transpose)
    inverse = np.linalg.inv(rot_matrix)  # Calculate the inverse of rotation matrix (R^-1)
    # print('inverse')
    # print(inverse)
    # determinant = np.linalg.det(rot_matrix)
    # print('determinant')
    # print(determinant)
    # identity = np.identity(rot_matrix.shape[0])  # Calculate the identity matrix of rotation matrix (I)
    # print('identity')
    # print(identity)

    # If they are identical then the rotation matrix is valid
    valid_rot = np.allclose(transpose, inverse)  # [1]
    # If the form of a transformation matrix is valid
    valid_trans = ((trans_matrix, 3, 0) == [0, 0, 0, 1])

    if valid_rot and valid_trans is True:
        return True
    else:
        return False


def nearest_neighbors(points, target, cutoff_dist):
    """
    :param points: array of points in 3-D euclidean space
    :param target: reference point in 3-D euclidean
    :param cutoff_dist: if a point is further than cutoff do not include it in output
    :return: array of points in order of closest to furthest from the target
    """
    # use np.array to find the absolute distance of each point in 3-D euclidean space from the target
    distances = np.array(np.linalg.norm(points-target, ord=2, axis=1))
    # print('distances:')
    # print(distances)

    # Sort points according to distances
    indexes = np.argsort(distances)
    ordered_points = points[indexes]
    # print('ordered_points')
    # print(ordered_points)

    # Filter points according to cutoff_distance
    ordered_dist = np.array(np.linalg.norm(ordered_points-target, ord=2, axis=1))
    # print('ordered_dist')
    # print(ordered_dist)

    cutoff_index = np.argmax(ordered_dist > cutoff_dist)
    # print('cutoff_index')
    # print(cutoff_index)

    # Include only points within the cutoff_dist
    filtered_points = ordered_points[:cutoff_index]

    return filtered_points


# if __name__ == '__main__':
    # a = np.arange(15).reshape(3, 5)
    # b = np.arange(15).reshape(3, 5)
    #
    # print('numpy_close: ', numpy_close(a, b))
    # my_func = lambda x: x**2
    # print('simple_minimizer: ', simple_minimizer(my_func, -1.75, 2.25, num=5))

    # nu_rolls = 5
    # it = 2000
    # print(np.zeros(nu))
    # print(r.randint(0, nu))
    # a = np.random.randint(0, nu, nu)  # [3]
    # print(np.sum(a))

    # print(np.array(np.sum(5), 2000))
    # simulate_dice_rolls(1, 2000)

    # tf_valid = np.array([[1, 0, 0, 9.1], [0, -0.50485, -0.86321, 0], [0, 0.86321, -0.50485, 4], [0, 0, 0, 1]])
    # tf_invalid = np.array([[0.70711, -0.70711, 0, 0], [0.70711, 0.70711, 0, 0], [0, 0, 1, 0], [0, 0, 0, 5]])
    # print(is_transformation_matrix(tf_valid))  # True
    # print(is_transformation_matrix(tf_invalid))  # False
    #
    # array = np.array([[1, 1, 1], [2, 3, 5], [0, 1, 1], [1.5, 1, 1], [10, 9, 9]])
    # target_pt = np.array([0, 1, 1])
    # print(nearest_neighbors(array, target_pt, 3.0))
