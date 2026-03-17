#!/usr/bin/env python3
"""3-mini_batch.py"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches for mini-batch gradient descent.

    Args:
        X: the numpy.ndarray of shape (m, nx) containing input data
            m: the number of data points
            nx: the number of features
        Y: the numpy.ndarray of shape (m, ny) containing labels
            m: the number of data points
            ny: the number of classes
        batch_size: the number of data points in each batch
    Returns:
        A list of tuples, where each tuple is (X_batch, Y_batch)
    """
    m = X.shape[0]
    mini_batches = []
    shuffled_X, shuffled_Y = shuffle_data(X, Y)
    for i in range(0, m, batch_size):
        batch_X = shuffled_X[i:i + batch_size]
        batch_Y = shuffled_Y[i:i + batch_size]
        mini_batches.append((batch_X, batch_Y))
    return mini_batches
