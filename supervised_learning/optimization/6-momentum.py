#!/usr/bin/env python3
"""6-momentum.py"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Sets up gradient descent with momentum in TensorFlow.

    Args:
        alpha: the learning rate
        beta1: the momentum weight
    Returns:
        A TensorFlow optimizer configured for momentum
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
