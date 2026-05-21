#!/usr/bin/env python3
'''6-expectation.py'''
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    '''calculates the expectation step in the EM algorithm for a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
            n: number of data points
            d: number of dimensions in each data point
        pi: numpy.ndarray of shape (k,) containing the priors for each cluster
            k: number of clusters
        m: numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster
    Returns:
        g: numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
        l: the total log likelihood
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None
    if np.any(np.linalg.det(S) <= 0):
        return None, None
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    g_sum = np.sum(g, axis=0)
    g_sum[g_sum == 0] = 1e-300
    log_likelihood = np.sum(np.log(g_sum))
    g /= g_sum
    return g, log_likelihood
