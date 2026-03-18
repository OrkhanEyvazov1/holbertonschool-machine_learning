#!/usr/bin/env python3
"""0-l2_reg_cost.py"""
import tensorflow as tf


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: a list of the weights matrices of the network
        L: the number of layers in the network
        m: the number of data points used in the training
    Returns:
        The cost of the network accounting for L2 regularization
    """
    l2_cost = 0
    for i in range(L):
        l2_cost += tf.reduce_sum(tf.square(weights[i]))
    l2_cost *= lambtha / (2 * m)
    return cost + l2_cost
