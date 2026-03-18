#!/usr/bin/env python3
"""1-l2_reg_gradient_descent.py"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient descent with L2 regularization.
    
    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of outputs from each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers in the network
    """
    m = Y.shape[1]
    
    # Calculate initial gradient for output layer (softmax)
    dZ = cache[f'A{L}'] - Y
    
    # Backpropagate through each layer
    for layer in range(L, 0, -1):
        # Calculate weight and bias gradients
        dW = (1 / m) * np.dot(dZ, cache[f'A{layer - 1}'].T) + (lambtha / m) * weights[f'W{layer}']
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Calculate gradient for previous layer
        if layer > 1:
            dA_prev = np.dot(weights[f'W{layer}'].T, dZ)
            dZ = dA_prev * (1 - np.power(cache[f'A{layer - 1}'], 2))  # tanh derivative
        
        # Update weights and biases
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db
        