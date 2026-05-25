#!/usr/bin/env python3
"""
0-gp.py
"""
import numpy as np


class GaussianProcess():
    """
    GaussianProcess class represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, ll=1, sigma_f=1):
        """
        Class constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.ll = ll
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        calculating the covariance kernel
        which is basically the eucleudian distance
        """
        sqdist1 = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        k = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
        return k
