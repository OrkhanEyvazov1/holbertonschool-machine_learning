#!/usr/bin/env python3
"""2-shuffle_data.py"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Args:
        X: the first numpy.ndarray of shape (m, nx) to shuffle
            m: the number of data points
            nx: the number of features
        Y: the second numpy.ndarray of shape (m, classes) to shuffle
            m: the number of data points
            classes: the number of classes
    Returns:
        The shuffled X and Y matrices, respectively
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y
