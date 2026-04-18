#!/usr/bin/env python3
"""6-transition_layer.py"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    a function that builds a transition
    layer as described in Densely Connected
    Convolutional Networks (2017)
    :param X: the output from the previous layer
    :param nb_filters: an integer representing the number of filters in X
    :param compression: the compression factor for the transition layer
    :return: the output of the transition layer and the number of filters
             within the output, respectively
    """
    initializer = K.initializers.HeNormal(seed=0)
    bn = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(bn)
    conv_1x1 = K.layers.Conv2D(filters=int(nb_filters * compression),
                               kernel_size=1, strides=1, padding='same',
                               kernel_initializer=initializer)(act)
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding='valid')(conv_1x1)
    nb_filters = int(nb_filters * compression)
    return avg_pool, nb_filters
