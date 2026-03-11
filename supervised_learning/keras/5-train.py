#!/usr/bin/env python3
"""5-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """train_model - trains a model using mini-batch gradient descent
    Args:
        network: is the model to train
        data: is a numpy.ndarray of shape (m, nx) containing the input data
            where m is the number of data points and nx is the number of
            features
        labels: is a one-hot numpy.ndarray of shape (m, classes) containing
            the labels of data where classes is the number of classes
        batch_size: is the number of data points in a batch
        epochs: is the number of passes through data for mini-batch gradient
            descent
        validation_data: is the data to validate the model with, if not None
            the model should be validated against this data at the end of
            each epoch. validation_data is a tuple (val_data, val_labels)
            where val_data is the input data to validate the model with and
            val_labels are the labels of val_data
    Returns:
        the History object generated after training the model
    """
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
