#!/usr/bin/env python3
'''
Docstring for math tasks
'''


class Binomial:
    '''
    Docstring for Binomial
    '''
    e = 2.7182818285

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - variance / mean
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        Args:
            k: number of successes
        Returns: PMF value for k
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        factorial_k = 1
        factorial_n_k = 1
        for i in range(1, k + 1):
            factorial_k *= i
        for j in range(1, self.n - k + 1):
            factorial_n_k *= j
        coefficient = (factorial_k * factorial_n_k) / (
            self.n * (self.n - 1) // 2)
        pmf = coefficient * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes
        Args:
            k: number of successes
        Returns: CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k + 1):
            factorial_i = 1
            factorial_n_i = 1
            for j in range(1, i + 1):
                factorial_i *= j
            for m in range(1, self.n - i + 1):
                factorial_n_i *= m
            coefficient = (factorial_i * factorial_n_i) / (
                self.n * (self.n - 1) // 2)
            pmf_i = (coefficient * (self.p ** i) *
                     ((1 - self.p) ** (self.n - i)))
            cdf += pmf_i
        return cdf
