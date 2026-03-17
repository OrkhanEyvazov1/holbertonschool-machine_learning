#!/usr/bin/env python3
"""4-moving_average.py"""
import numpy as np


def moving_average(data, beta):
    '''Calculates the moving average of a data set.'''
    v = 0
    averages = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        averages.append(v / (1 - beta ** (i + 1)))
    return averages
