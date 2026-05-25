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
        calculates the covariance kernel matrix
        between two matrices
        """
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sqdist += np.sum(X2 ** 2, axis=1)
        sqdist -= 2 * np.matmul(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)
