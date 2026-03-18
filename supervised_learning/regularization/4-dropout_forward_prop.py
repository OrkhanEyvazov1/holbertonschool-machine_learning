#!/usr/bin/env python3
"""4-dropout_forward_prop.py"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing input data
        weights: dictionary of weights and biases
        L: number of layers in the network
        keep_prob: probability that a node will be kept

    Returns:
        dictionary containing outputs of each layer and dropout masks
    """
    cache = {}
    A = X
    cache['A0'] = A

    for layer in range(1, L + 1):
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']
        Z = np.dot(W, A) + b
        if layer == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache[f'D{layer}'] = D
        cache[f'A{layer}'] = A
    return cache
