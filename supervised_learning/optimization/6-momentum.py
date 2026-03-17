#!/usr/bin/env python3
"""6-momentum.py"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a network in using
    the gradient descent with momentum optimization algorithm.

    Args:
        loss: the loss of the network’s prediction
        alpha: the learning rate
        beta1: the momentum weight
    Returns:
        The momentum optimization operation
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
