#!/usr/bin/env python3
"""7-densenet121.py"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growsth_rate=32, compression=1.0):
    """
    a function that builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks (2017)
    :param growth_rate: the growth rate for the dense blocks
    :return: the keras model
    """
    X_input = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal(seed=0)
    conv_7x7 = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                               padding='same', kernel_initializer=initializer)(
        X_input)
    norm_1 = K.layers.BatchNormalization(axis=3)(conv_7x7)
    act_1 = K.layers.Activation('relu')(norm_1)
    max_pool = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                     padding='same')(act_1)
    nb_filters = 64
    layers = [6, 12, 24, 16]
    dense_block_1, nb_filters = dense_block(max_pool, nb_filters,
                                           growsth_rate, layers[0])
    transition_layer_1, nb_filters = transition_layer(dense_block_1,
                                                     nb_filters,
                                                     compression)
    dense_block_2, nb_filters = dense_block(transition_layer_1, nb_filters,
                                           growsth_rate, layers[1])
    transition_layer_2, nb_filters = transition_layer(dense_block_2,
                                                     nb_filters,
                                                     compression)
    dense_block_3, nb_filters = dense_block(transition_layer_2, nb_filters,
                                           growsth_rate, layers[2])
    transition_layer_3, nb_filters = transition_layer(dense_block_3,
                                                     nb_filters,
                                                     compression)
    dense_block_4, nb_filters = dense_block(transition_layer_3, nb_filters,
                                           growsth_rate, layers[3])
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                         padding='valid')(dense_block_4)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=initializer)(avg_pool)
    return K.Model(inputs=X_input, outputs=softmax)
