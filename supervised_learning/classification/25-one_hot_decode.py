#!/usr/bin/env python3
"""25.One-Hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """one_hot_decode - converts a one-hot matrix into a vector of
    numeric labels
    Args:
        one_hot: is a numpy.ndarray with shape (classes, m) containing
        the one-hot encoded labels, where classes is the number of
        classes and m is the number of examples
    Returns:
        A numpy.ndarray with shape (m,) containing the numeric class
        labels for each example, or None on failure
    """
    if (not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2):
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
