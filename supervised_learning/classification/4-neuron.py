#!/usr/bin/env python3
"""3. Neuron"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor method
        Args:
            nx: number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = __import__('numpy').random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
        Returns:
            Activated output of the neuron (prediction)
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

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

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
        Returns:
            The neuron’s prediction and the cost of the network, respectively
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
