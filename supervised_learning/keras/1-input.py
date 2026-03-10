#!/usr/bin/env python3
"""1-input.py
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """build_model - builds a neural network with the Keras library
    Args:
        nx: is the number of input features to the network
        layers: is a list containing the number of nodes in each layer
        activations: is a list containing the activation functions used for
            each layer of the network
        lambtha: is the L2 regularization parameter
        keep_prob: is the probability that a node will be kept for dropout
    Returns:
        the keras model
    """
    #Sequential model not allowed
    input_layer = K.Input(shape=(nx,))
    x = input_layer
    for i, nodes in enumerate(layers):
        x = K.layers.Dense(
            nodes,
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    return K.Model(inputs=input_layer, outputs=x)
