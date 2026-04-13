#!/usr/bin/env python3
"""0-inception_block.py
"""
import tensorflow as tf


def inception_block(A_prev, filters):
    """
    a function that builds an inception block as described in
    Going Deeper with Convolutions (2014)
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F1, F3R, F3, F5R,
    F5, FPP, respectively:
        - F1 is the number of filters in the 1x1 convolution
        - F3R is the number of filters in the 1x1 convolution before
        the 3x3 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F5R is the number of filters in the 1x1 convolution before
        the 5x5 convolution
        - F5 is the number of filters in the 5x5 convolution
        - FPP is the number of filters in the 1x1 convolution after
        the max pooling
    :return: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1 = tf.keras.layers.Conv2D(filters=F1,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    activation='relu')(A_prev)
    conv_3 = tf.keras.layers.Conv2D(filters=F3R,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    activation='relu')(A_prev)
    conv_3 = tf.keras.layers.Conv2D(filters=F3,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation='relu')(conv_3)
    conv_5 = tf.keras.layers.Conv2D(filters=F5R,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    activation='relu')(A_prev)
    conv_5 = tf.keras.layers.Conv2D(filters=F5,
                                    kernel_size=(5, 5),
                                    padding='same',
                                    activation='relu')(conv_5)
    max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                        strides=(1, 1),
                                        padding='same')(A_prev)
    max_pool = tf.keras.layers.Conv2D(filters=FPP,
                                        kernel_size=(1, 1),
                                        padding='same',
                                        activation='relu')(max_pool)
    return tf.keras.layers.concatenate([conv_1, conv_3, conv_5, max_pool])
