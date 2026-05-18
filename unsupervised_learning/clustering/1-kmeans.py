#!/usr/bin/env python3
"""1-kmeans.py
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)

    C = np.random.uniform(low=low, high=high, size=(k, d))

    for _ in range(iterations):
        C_prev = np.copy(C)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        clss = np.argmin(distances, axis=-1)

        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                C[j] = np.random.uniform(low=low, high=high, size=(1, d))
            else:
                C[j] = points.mean(axis=0)

        if np.all(C_prev == C):
            return C, clss

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
    clss = np.argmin(distances, axis=-1)

    return C, clss
