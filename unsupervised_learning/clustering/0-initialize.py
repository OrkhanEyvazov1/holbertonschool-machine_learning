#!/usr/bin/env python3
"""0-initialize.py
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)

    return np.random.uniform(low=low, high=high, size=(k, X.shape[1]))
