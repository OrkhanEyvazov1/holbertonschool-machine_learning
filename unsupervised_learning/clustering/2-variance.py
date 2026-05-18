#!/usr/bin/env python3
"""2-variance.py
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(C, np.ndarray) or len(C.shape) != 2:
            return None
        if X.shape[1] != C.shape[1]:
            return None

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        min_distances = np.min(distances, axis=-1)
        
        return np.sum(min_distances ** 2)
    except Exception:
        return None
