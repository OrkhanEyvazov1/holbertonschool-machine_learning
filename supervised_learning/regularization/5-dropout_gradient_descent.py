#!/usr/bin/env python3
"""5-dropout_gradient_descent.py"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout
    regularization using gradient descent.
    
    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases of the neural network
        cache: dictionary of outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
    """
    m = Y.shape[1]
    
    dZ = cache[f"A{L}"] - Y
    
    for layer in range(L, 0, -1):
        A_prev = cache[f"A{layer - 1}"]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        weights[f"W{layer}"] -= alpha * dW
        weights[f"b{layer}"] -= alpha * db
        if layer > 1:
            dA_prev = np.dot(weights[f"W{layer}"].T, dZ)
            dA_prev *= cache[f"D{layer - 1}"]
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - cache[f"A{layer - 1}"] ** 2)
