#!/usr/bin/env python3
"""2-l2_reg_cost.py"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: tensor containing the cost without L2 regularization
        model: Keras model that includes layers with L2 regularization

    Returns:
        Tensor containing the cost accounting for L2 regularization
    """
    if not model.losses:
        return cost
    return cost + tf.convert_to_tensor(model.losses)
