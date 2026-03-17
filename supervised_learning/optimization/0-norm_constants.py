#!/usr/bin/env python3
"""0-norm_constants.py"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization constants of a matrix.

    Args:
        X: the numpy.ndarray of shape (m, nx) to normalize
            m: the number of data points
            nx: the number of features
    Returns:
        The mean and standard deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    return mean, stddev
