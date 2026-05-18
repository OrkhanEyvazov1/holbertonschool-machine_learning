#!/usr/bin/env python3
"""4-initialize.py
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None, None
    d = X.shape[1]
    pi = np.full(shape=(k,), fill_value=1/k)
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m, S
