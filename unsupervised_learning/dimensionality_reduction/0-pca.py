#!/usr/bin/env python3
"""0-pca.py
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    # Computes SVD
    u, s, vh = np.linalg.svd(X)
    # Calculate the cumulative sum of the variance
    cumulative_variance = np.cumsum(s) / np.sum(s)
    # Find the minimum number of dimensions needed to maintain `var`
    nd = np.argwhere(cumulative_variance >= var)[0, 0] + 1
    # Return the weights matrix, W
    W = vh[:nd].T
    return W
