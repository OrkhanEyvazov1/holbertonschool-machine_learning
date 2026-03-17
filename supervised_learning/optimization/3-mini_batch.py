#!/usr/bin/env python3
"""3-mini_batch.py"""
import numpy as np


def mini_batch(X, Y, batch_size=32):
    """Divides the data into batches.

    Args:
        X: the numpy.ndarray of shape (m, nx) to divide into batches
            m: the number of data points
            nx: the number of features
        Y: the numpy.ndarray of shape (m, classes) to divide into batches
            m: the number of data points
            classes: the number of classes
        batch_size: the number of data points in each batch
    Returns:
        A list of tuples, where each tuple is (batch_X, batch_Y)
    """
    m = X.shape[0]
    mini_batches = []
    for i in range(0, m, batch_size):
        batch_X = X[i:i + batch_size]
        batch_Y = Y[i:i + batch_size]
        mini_batches.append((batch_X, batch_Y))
    return mini_batches
