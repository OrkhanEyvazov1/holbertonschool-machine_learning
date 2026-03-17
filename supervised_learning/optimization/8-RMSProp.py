#!/usr/bin/env python3
"""8-RMSProp.py"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Sets up RMSProp optimization in TensorFlow.

    Args:
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero
    Returns:
        A TensorFlow optimizer configured for RMSProp
    """
    return tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                       rho=beta2,
                                       epsilon=epsilon)
