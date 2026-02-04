#!/usr/bin/env python3
'''
Docstring for math tasks
'''


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    probabilities of developing severe side effects
    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing the various hypothetical probabilities
           of developing severe side effects
    Returns:
    a 1D numpy.ndarray containing the likelihood of obtaining the data,
             x and n, for each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
                "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
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

    likelihoods = coefficient * (P ** x) * ((1 - P) ** (n - x))
    return likelihoods
