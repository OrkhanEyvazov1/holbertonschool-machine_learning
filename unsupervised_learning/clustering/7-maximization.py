#!/usr/bin/env python3
"""7-maximization.py"""
import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm for a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
            n: number of data points
            d: number of dimensions in each data point
        g: numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
        k: number of clusters
    Returns:
        pi: numpy.ndarray of shape (k,) containing the updated priors for
        each cluster
        m: numpy.ndarray of shape (k, d) containing the updated centroid means
        for each cluster
        S: numpy.ndarray of shape (k, d, d) containing the updated covariance
        matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None
    Nk = np.sum(g, axis=1)
    pi = Nk / n
    m = np.dot(g, X) / Nk[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        X_centered = X - m[i]
        S[i] = np.dot(g[i] * X_centered.T, X_centered) / Nk[i]
    return pi, m, S
