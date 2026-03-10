#!/usr/bin/env python3
"""2-optimize.py
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """optimize_model - sets up Adam optimization for a keras model with
        categorical crossentropy loss and accuracy metrics
    Args:
        network: is the model to optimize
        alpha: is the learning rate
        beta1: is the beta1 parameter for the Adam optimization
        beta2: is the beta2 parameter for the Adam optimization
    Returns:
        the optimized model
    """
    network.compile(
        optimizer=K.optimizers.Adam(alpha, beta1, beta2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return network
