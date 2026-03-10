#!/usr/bin/env python3
"""3-one_hot.py
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """one_hot - converts a label vector into a one-hot matrix
    Args:
        labels: is a numpy.ndarray containing the labels to convert, shape
            (m,) where m is the number of labels
        classes: is the maximum number of classes found in labels, if None
            then set classes to the maximum label in labels
    Returns:
        Matrix : a one-hot encoding of labels
    """
    if classes is None:
        classes = max(labels) + 1

    return K.utils.to_categorical(labels, num_classes=classes)
