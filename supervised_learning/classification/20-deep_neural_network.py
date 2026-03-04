#!/usr/bin/env python3
"""14.Dee[] Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification"""

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for intermediary values cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights and biases"""
        return self.__weights

    def __init__(self, nx, layers):
        """Constructor method
        Args:
            nx: number of input features to the neural network
            layers: list representing the number of nodes in each layer of the
                network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights["W" + str(i + 1)] = (np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i + 1)] = (np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]))

            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
        Returns:
            The output of the neural network and the cache containing the
            intermediary values of the network
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            W = self.__weights["W" + str(i + 1)]
            b = self.__weights["b" + str(i + 1)]
            A_prev = self.__cache["A" + str(i)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(i + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
                of the neuron for each example
        Returns:
            The cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
                of the neuron for each example
        Returns:
            The cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
