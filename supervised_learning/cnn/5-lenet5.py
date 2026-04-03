#!/usr/bin/env python3
"""5-lenet5.py
"""
from tensorflow import keras as K


def lenet5(X):
    """
    a function that builds a modified version of the LeNet-5 architecture
    using Keras
    :param X: is the K.Input of shape (m, 28, 28, 1) containing the input
    images for the network
    :return: a K.Model compiled to use Adam optimization and accuracy
    metrics with categorical crossentropy loss
    """
    initializer = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding="same",
                            activation="relu",
                            kernel_initializer=initializer)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding="valid",
                            activation="relu",
                            kernel_initializer=initializer)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    flatten = K.layers.Flatten()(pool2)

    dense1 = K.layers.Dense(units=120,
                            activation="relu",
                            kernel_initializer=initializer)(flatten)

    dense2 = K.layers.Dense(units=84,
                            activation="relu",
                            kernel_initializer=initializer)(dense1)

    output = K.layers.Dense(units=10,
                            activation="softmax",
                            kernel_initializer=initializer)(dense2)

    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
