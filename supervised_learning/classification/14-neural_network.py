#!/usr/bin/env python3
"""13. Neural Network"""
import numpy as np


class NeuralNetwork:
    """Neural network with one hidden layer for binary classification"""

    def __init__(self, nx, nodes):
        """Constructor method
        Args:
            nx: number of input features to the neural network
            nodes: number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for hidden layer weights"""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer bias"""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output"""
        return self.__A1

    @property
    def W2(self):
        """Getter for output neuron weights"""
        return self.__W2

    @property
    def b2(self):
        """Getter for output neuron bias"""
        return self.__b2

    @property
    def A2(self):
        """Getter for output neuron activated output"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
        Returns:
            The activated output for the hidden and output layers,
            respectively
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

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
        """Evaluates the neural network's predictions
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
        Returns:
            The neural network's prediction and
            the cost of the network, respectively
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            A1: numpy.ndarray with shape (nodes, m)
            containing the activated output
                of the hidden layer for each example
            A2: numpy.ndarray with shape (1, m) containing the activated output
                of the output neuron for each example
            alpha: learning rate
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            iterations: number of iterations to train over
            alpha: learning rate
        Returns:
            The evaluation of the training data after iterations of training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
