#!/usr/bin/env python3
"""11-learning_rate_decay.py"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using stepwise inverse time decay.

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate of decay
        global_step: the number of passes of gradient descent that have
                     elapsed
        decay_step: the number of passes of gradient descent that should
                    occur before alpha is decayed further
    Returns:
        The updated learning rate
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
