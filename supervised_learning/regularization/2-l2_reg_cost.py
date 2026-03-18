#!/usr/bin/env python3
"""2-l2_reg_cost.py"""
import numpy as np


def l2_reg_cost(cost, weights, lambtha, L):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: cost of the network without L2 regularization
        weights: dictionary of weights and biases
        lambtha: L2 regularization parameter
        L: number of layers in the network

    Returns:
        The cost of the network with L2 regularization
    """
    l2_cost = 0
    for layer in range(1, L + 1):
        l2_cost += np.sum(np.square(weights[f'W{layer}']))
    return cost + (lambtha / (2 * L)) * l2_cost
