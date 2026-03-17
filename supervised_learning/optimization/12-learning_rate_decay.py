#!/usr/bin/env python3
"""12-learning_rate_decay.py"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Creates a learning rate decay operation in TensorFlow using
    inverse time decay in a stepwise fashion.

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate at which alpha decays
        decay_step: the number of passes of gradient descent that should
                    occur before alpha is decayed further
    Returns:
        The learning rate decay operation
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
