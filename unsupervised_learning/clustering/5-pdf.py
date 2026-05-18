#!/usr/bin/env python3
"""5. PDF"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density
    function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    cond1 = X.shape[1] != m.shape[0]
    cond2 = X.shape[1] != S.shape[0]
    cond3 = S.shape[0] != S.shape[1]
    if cond1 or cond2 or cond3:
        return None
    n, d = X.shape
    det_S = np.linalg.det(S)
    inv_S = np.linalg.inv(S)
    den = np.sqrt(((2 * np.pi) ** d) * det_S)
    X_m = X - m
    exponent = np.sum(np.matmul(X_m, inv_S) * X_m, axis=1)
    P = (1 / den) * np.exp(-0.5 * exponent)
    return np.maximum(P, 1e-300)
