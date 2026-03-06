#!/usr/bin/env python3
"""24.One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """one_hot_encode - converts a numeric label
    vector into a one-hot matrix
    Args:
        Y: is a numpy.ndarray with shape (m,) containing
        the numeric class
        labels for the data, where m is the number of examples
        classes: is the maximum value of a class
    Returns:
        A one-hot matrix with shape (classes, m)
    """
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy.ndarray")
    if not isinstance(classes, int):
        raise TypeError("classes must be an integer")
    if classes <= 0:
        raise ValueError("classes must be a positive integer")

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1

    return one_hot
