#!/usr/bin/env python3
"""12-learning_rate_decay.py"""
import tensorflow as tf


def create_learning_rate_decay_op(alpha, decay_rate, global_step, decay_step):
    """Sets up learning rate decay in TensorFlow.

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate of decay
        global_step: the number of passes of gradient descent that have
                     elapsed
        decay_step: the number of passes of gradient descent that should
                    occur before alpha is decayed further
    Returns:
        A TensorFlow learning rate decay operation
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)(global_step)
