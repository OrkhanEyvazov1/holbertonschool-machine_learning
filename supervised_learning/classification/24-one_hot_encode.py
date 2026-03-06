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
    if (not isinstance(Y, np.ndarray) or Y.ndim != 1 or
            not isinstance(classes, int) or classes <= 0):
        return None

    m = Y.shape[0]
    if np.any(Y < 0) or np.any(Y >= classes):
        return None

    try:
        one_hot = np.zeros((classes, m))
        one_hot[Y.astype(int), np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
