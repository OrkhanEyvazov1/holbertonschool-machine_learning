#!/usr/bin/env python3
"""L2 regularized layer creation in TensorFlow."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer with L2 regularization.

    Args:
        prev: tensor output from the previous layer
        n: number of nodes in the new layer
        activation: activation function for the new layer
        lambtha: L2 regularization parameter

    Returns:
        Tensor output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )
    regularizer = tf.keras.regularizers.l2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
