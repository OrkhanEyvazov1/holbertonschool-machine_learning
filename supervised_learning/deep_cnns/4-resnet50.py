#!/usr/bin/env python3
"""4-resnet50.py"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    a function that builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
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
    filters = (64, 64, 256)
    proj_block_1 = projection_block(max_pool, filters, s=1)
    filters = (64, 64, 256)
    id_block_1a = identity_block(proj_block_1, filters)
    id_block_1b = identity_block(id_block_1a, filters)
    filters = (128, 128, 512)
    proj_block_2 = projection_block(id_block_1b, filters)
    filters = (128, 128, 512)
    id_block_2a = identity_block(proj_block_2, filters)
    id_block_2b = identity_block(id_block_2a, filters)
    id_block_2c = identity_block(id_block_2b, filters)
    filters = (256, 256, 1024)
    proj_block_3 = projection_block(id_block_2c, filters)
    filters = (256, 256, 1024)
    id_block_3a = identity_block(proj_block_3, filters)
    id_block_3b = identity_block(id_block_3a, filters)
    id_block_3c = identity_block(id_block_3b, filters)
    id_block_3d = identity_block(id_block_3c, filters)
    id_block_3e = identity_block(id_block_3d, filters)
    filters = (512, 512, 2048)
    proj_block_4 = projection_block(id_block_3e, filters)
    id_block_4a = identity_block(proj_block_4, filters)
    id_block_4b = identity_block(id_block_4a, filters)
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(id_block_4b)
    softmax = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=initializer)(avg_pool)
    return K.Model(inputs=X_input, outputs=softmax)
