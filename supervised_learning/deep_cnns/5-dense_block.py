#!/usr/bin/env python3
"""5-dense_block.py"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    a function that builds a dense block as described in Densely Connected
    Convolutional Networks (2017)
    :param X: the output from the previous layer
    :param nb_filters: an integer representing the number of filters in X
    :param growth_rate: the growth rate for the dense block
    :param layers: the number of layers in the dense block
    :return: the concatenated output of each layer in the dense block and
             the number of filters within the concatenated outputs, respectively
    """
    initializer = K.initializers.HeNormal(seed=0)
    for i in range(layers):
        bn = K.layers.BatchNormalization(axis=3)(X)
        act = K.layers.Activation('relu')(bn)
        conv_1x1 = K.layers.Conv2D(filters=4 * growth_rate, kernel_size=1,
                                   strides=1, padding='same',
                                   kernel_initializer=initializer)(act)
        bn = K.layers.BatchNormalization(axis=3)(conv_1x1)
        act = K.layers.Activation('relu')(bn)
        conv_3x3 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                   strides=1, padding='same',
                                   kernel_initializer=initializer)(act)
        X = K.layers.Concatenate()([X, conv_3x3])
        nb_filters += growth_rate
    return X, nb_filters
