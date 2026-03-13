#!/usr/bin/env python3
"""13-predict.py"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network.

    Args:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose: a boolean that determines if output should be printed
    Returns:
        the prediction for the data
    """
    return network.predict(data, verbose=verbose)
