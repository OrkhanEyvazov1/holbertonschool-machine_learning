#!/usr/bin/env python
"""1-normalize.py"""
import numpy as np


def normalize(X, m, s):
    """Normalizes a matrix using the standard score.

    Args:
        X: the numpy.ndarray of shape (m, nx) to normalize
            m: the number of data points
            nx: the number of features
        m: the mean of each feature, respectively
        s: the standard deviation of each feature, respectively
    Returns:
        The normalized X matrix
    """
    return (X - m) / s
