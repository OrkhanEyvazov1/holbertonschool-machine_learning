#!/usr/bin/env python3
"""0. Neuron"""
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
        self.W = __import__('numpy').random.randn(1, nx)
        self.b = 0
        self.A = 0
