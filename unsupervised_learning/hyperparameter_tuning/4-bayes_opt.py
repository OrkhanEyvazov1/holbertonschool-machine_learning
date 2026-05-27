#!/usr/bin/env python3
"""
Performs Bayesian optimization on a noiseless 1D Gaussian process
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    A class that performs Bayesian optimization on a Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, ell=1,
                 sigma_f=1, xsi=0.01, minimize=True, **kwargs):
        """
        A function that initializes the class BayesianOptimization
        :param f: the black-box function to be optimized
        :param X_init: numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        :param Y_init: numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        :param bounds: tuple of (min, max) representing the bounds of the
        space in which to look for the optimal point
        :param ac_samples: number of samples that should be analyzed during
        acquisition
        :param l: length parameter for the kernel
        :param sigma_f: standard deviation given to the output of the
        black-box function
        :param xsi: exploration-exploitation factor for acquisition
        :param minimize: bool determining whether optimization should be
        performed for minimization (True) or maximization (False)
        """
        length = kwargs.get('l', ell)
        self.f = f
        self.gp = GP(X_init, Y_init, length, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        :return: X_next, EI
            X_next is a numpy.ndarray of shape (1,) representing the next best
            sample point
            EI is a numpy.ndarray of shape (ac_samples,) containing the
            expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is False:
            mu_sample_opt = np.amax(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        else:
            mu_sample_opt = np.amin(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi

        Z = np.zeros(sigma.shape)
        nonzero = sigma != 0
        Z[nonzero] = imp[nonzero] / sigma[nonzero]

        abs_z = np.abs(Z)
        t = 1.0 / (1.0 + 0.2316419 * abs_z)
        poly = t * (
            0.319381530 + t * (
                -0.356563782 + t * (
                    1.781477937 + t * (
                        -1.821255978 + t * 1.330274429
                    )
                )
            )
        )
        cdf = 1.0 - (np.exp(-0.5 * abs_z ** 2) / np.sqrt(2.0 * np.pi)) * poly
        cdf = np.where(Z < 0, 1.0 - cdf, cdf)
        pdf = np.exp(-0.5 * Z ** 2) / np.sqrt(2.0 * np.pi)
        ei = imp * cdf + sigma * pdf
        ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei
