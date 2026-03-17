#!/usr/bin/env python3
"""11-learning_rate_decay.py"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation in TensorFlow.

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate of decay
        global_step: the number of passes of gradient descent that have
                     elapsed. Should be cast to an int.
        decay_step: the number of passes of gradient descent that should
                    occur before alpha is decayed. Should be cast to an int.
    Returns:
        The learning rate decay operation
    """
    return tf.compat.v1.train.inverse_time_decay(alpha,
                                                  global_step,
                                                  decay_step,
                                                  decay_rate)
