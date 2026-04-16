#!/usr/bin/env python3
"""1-inception_network.py
"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    a function that builds the inception network as described in
    Going Deeper with Convolutions (2014)
    :return: the keras model
    """
    X_input = K.Input(shape=(224, 224, 3))
    conv_7x7 = K.layers.Conv2D(filters=64, kernel_size=7,
                               strides=2, padding='same',
                               activation='relu')(X_input)
    max_pool_1 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(conv_7x7)
    conv_1x1 = K.layers.Conv2D(filters=64, kernel_size=1,
                               padding='same', activation='relu')(max_pool_1)
    conv_3x3 = K.layers.Conv2D(filters=192, kernel_size=3,
                               padding='same', activation='relu')(conv_1x1)
    max_pool_2 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(conv_3x3)
    filters = (64, 96, 128, 16, 32, 32)
    incp_3a = inception_block(max_pool_2, filters)
    filters = (128, 128, 192, 32, 96, 64)
    incp_3b = inception_block(incp_3a, filters)
    max_pool_3 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(incp_3b)
    filters = (192, 96, 208, 16, 48, 64)
    incp_4a = inception_block(max_pool_3, filters)
    filters = (160, 112, 224, 24, 64, 64)
    incp_4b = inception_block(incp_4a, filters)
    filters = (128, 128, 256, 24, 64, 64)
    incp_4c = inception_block(incp_4b, filters)
    filters = (112, 144, 288, 32, 64, 64)
    incp_4d = inception_block(incp_4c, filters)
    filters = (256, 160, 320, 32, 128, 128)
    incp_4e = inception_block(incp_4d, filters)
    max_pool_4 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(incp_4e)
    filters = (256, 160, 320, 32, 128, 128)
    incp_5a = inception_block(max_pool_4, filters)
    filters = (384, 192, 384, 48, 128, 128)
    incp_5b = inception_block(incp_5a, filters)
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                         padding='valid')(incp_5b)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)
    softmax = K.layers.Dense(units=1000, activation='softmax')(dropout)
    return K.Model(inputs=X_input, outputs=softmax)
