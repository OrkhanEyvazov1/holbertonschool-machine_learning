#!/usr/bin/env python3
"""6-momentum.py"""


def momentum(alpha, beta, var, grad, v):
    """Updates a variable using the gradient
    descent with momentum optimization algorithm.

    Args:
        alpha: the learning rate
        beta: the momentum weight
        var: a numpy.ndarray containing the variable to be updated
        grad: a numpy.ndarray containing the gradient of var
        v: a numpy.ndarray containing the previous velocity
    Returns:
        The updated variable and the new velocity, respectively
    """
    v = beta * v + (1 - beta) * grad
    var = var - alpha * v
    return var, v
