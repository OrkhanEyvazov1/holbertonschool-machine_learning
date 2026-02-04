#!/usr/bin/env python3
'''
Docstring for math tasks
'''


def likelihood(x, n, p):
    """
    Calculates the likelihood of obtaining this data given various
    probabilities of developing severe side effects
    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        p: numpy.ndarray of various probabilities
    Returns: numpy.ndarray containing the likelihood of obtaining the data
             for each probability in p
    """
    if not isinstance(p, np.ndarray):
        raise TypeError("p must be a numpy.ndarray")
    if n <= 0:
        raise ValueError("n must be a positive value")
    if x < 0:
        raise ValueError("x must be an integer that is greater than or "
                         "equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    factorial_x = 1
    factorial_n_x = 1
    factorial_n = 1
    for i in range(1, x + 1):
        factorial_x *= i
    for j in range(1, n - x + 1):
        factorial_n_x *= j
    for m in range(1, n + 1):
        factorial_n *= m
    coefficient = factorial_n / (factorial_x * factorial_n_x)

    likelihoods = coefficient * (p ** x) * ((1 - p) ** (n - x))
    return likelihoods
