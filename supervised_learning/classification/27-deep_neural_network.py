#!/usr/bin/env python3
"""27.Deep Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
            if i == self.__L - 1:
                T = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = T / np.sum(T, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(i + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Args:
            Y: numpy.ndarray with shape (classes, m) that contains the
                one-hot labels for the input data
                for the input data
            A: numpy.ndarray with shape (classes, m) containing the activated
                output of the network for each example
        Returns:
            The cost of the model using multiclass cross-entropy
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
            Y: numpy.ndarray with shape (classes, m) that contains the one-hot
                labels for the input data
        Returns:
            The neural network’s prediction and the cost of the network,
            respectively
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
        Args:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            cache: dictionary containing the intermediary values of the network
            alpha: learning rate
        Returns:
            None
        """
        m = Y.shape[1]
        A_last = cache["A" + str(self.__L)]
        dZ = A_last - Y
        for i in reversed(range(self.__L)):
            A_prev = cache["A" + str(i)]
            W = self.__weights["W" + str(i + 1)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 0:
                dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)
            self.__weights["W" + str(i + 1)] -= alpha * dW
            self.__weights["b" + str(i + 1)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: if True, prints information about training
            graph: if True, graphs information about training
            step: interval of iterations to print/graph training data
        Returns:
            The evaluation of the training data after iterations of
            training have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        x_data = []
        y_data = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    x_data.append(i)
                    y_data.append(cost)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(x_data, y_data, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
