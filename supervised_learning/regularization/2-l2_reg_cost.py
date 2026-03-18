#!/usr/bin/env python3
"""2-l2_reg_cost.py"""
import tensorflow as tf


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
    for i in range(1, L + 1):
        l2_cost += tf.reduce_sum(tf.square(weights[f'W{i}']))
    l2_cost *= lambtha / (2 * cost.shape[0])
    return cost + l2_cost
